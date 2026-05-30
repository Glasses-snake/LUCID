"""
LUCID — Lightweight Uncertainty-Calibrated Imaging via Deep learning

A compact reconstruction network with dual heads:
  - mu (image)         : final reconstructed FPM image
  - log_var (per-pixel): aleatoric uncertainty in log-variance form

A Bayesian last convolution enables epistemic uncertainty via Monte-Carlo
sampling at inference time.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Basic building blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Channel-first LayerNorm."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class CCM(nn.Module):
    """Convolutional Channel Mixer."""

    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden = int(dim * growth_rate)
        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return self.ccm(x)


class FreqAFM(nn.Module):
    """Frequency-Adaptive Feature Modulation."""

    def __init__(self, dim):
        super().__init__()
        self.n_levels = 4
        chunk = dim // self.n_levels

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chunk * 2, chunk * 2, 1),
                nn.GELU(),
                nn.Conv2d(chunk * 2, chunk * 2, 1),
            )
            for _ in range(self.n_levels)
        ])
        self.aggr = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

    def _gate(self, freq, level):
        C = freq.shape[1]
        ri = torch.cat([freq.real, freq.imag], dim=1)
        gated = self.gates[level](ri)
        return torch.complex(gated[:, :C], gated[:, C:])

    def forward(self, x):
        B, C, H, W = x.shape
        chunks = x.chunk(self.n_levels, dim=1)
        out = []

        for i in range(self.n_levels):
            freq = torch.fft.fftshift(
                torch.fft.fft2(chunks[i], norm='ortho'), dim=(-2, -1),
            )
            if i > 0:
                rH = max(H // (2 ** i), 2)
                rW = max(W // (2 ** i), 2)
                h0, w0 = (H - rH) // 2, (W - rW) // 2
                low = freq[:, :, h0:h0 + rH, w0:w0 + rW]
                gated_low = self._gate(low, i)
                freq = torch.zeros_like(freq)
                freq[:, :, h0:h0 + rH, w0:w0 + rW] = gated_low
            else:
                freq = self._gate(freq, i)

            s = torch.fft.ifft2(
                torch.fft.ifftshift(freq, dim=(-2, -1)), norm='ortho',
            ).real
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        return self.act(out) * x


class AttBlock(nn.Module):
    """FreqAFM + CCM with pre-norm residual shortcuts."""

    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.freq_afm = FreqAFM(dim)
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.freq_afm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


# ---------------------------------------------------------------------------
#  Heads
# ---------------------------------------------------------------------------

class UncertaintyHead(nn.Module):
    """Per-pixel log-variance prediction head."""

    def __init__(self, in_dim, out_channels=3, n_layers=3):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv2d(in_dim, in_dim, 3, 1, 1), nn.ELU()]
        self.feat_refine = nn.Sequential(*layers)
        self.upsample = nn.Sequential(nn.Conv2d(in_dim, out_channels, 3, 1, 1))

    def forward(self, feat):
        return self.upsample(self.feat_refine(feat))


class BayesianConv2d(nn.Module):
    """3x3 Bayesian conv: w = mu_w + softplus(rho_w) * eps, eps ~ N(0, 1)."""

    def __init__(self, in_c, out_c, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight_mu = nn.Parameter(torch.empty(out_c, in_c, kernel_size, kernel_size))
        self.weight_rho = nn.Parameter(torch.empty(out_c, in_c, kernel_size, kernel_size))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_c))
            self.bias_rho = nn.Parameter(torch.empty(out_c))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.deterministic = False
        self._init_params()

    def _init_params(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        if self.bias_mu is not None:
            fan_in = self.in_c * self.kernel_size ** 2
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x):
        if self.deterministic:
            return F.conv2d(x, self.weight_mu, self.bias_mu, padding=self.padding)

        w_sigma = F.softplus(self.weight_rho)
        weight = self.weight_mu + w_sigma * torch.randn_like(self.weight_mu)
        bias = None
        if self.bias_mu is not None:
            b_sigma = F.softplus(self.bias_rho)
            bias = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)
        return F.conv2d(x, weight, bias, padding=self.padding)


# ---------------------------------------------------------------------------
#  LUCID — main model
# ---------------------------------------------------------------------------

class LUCID(nn.Module):
    """End-to-end LUCID model for FPM image enhancement.

    Operates at native input resolution (no spatial up/down-sampling).
    Outputs:
      forward(x)         -> (mu, log_var)
      mc_inference(x, T) -> (mu_mean, aleatoric, epistemic)
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # Network width and depth are fixed per the released configuration.
        dim = 48
        n_blocks = 12
        ffn_scale = 2.0

        self.to_feat = nn.Conv2d(in_channels, dim, 3, 1, 1)
        self.body = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        # Reconstruction head: a single Bayesian 3x3 convolution, enabling
        # cheap Monte-Carlo epistemic sampling at inference time.
        self.sr_last = BayesianConv2d(dim, in_channels, 3, 1)

        # Uncertainty head: produces per-pixel log-variance.
        self.unc_head = UncertaintyHead(dim, out_channels=in_channels, n_layers=3)

    def forward(self, x):
        feat = self.to_feat(x)
        feat = self.body(feat) + feat
        mu = self.sr_last(feat) + x             # global image residual
        log_var = self.unc_head(feat)
        return mu, log_var

    def set_deterministic(self, mode=True):
        for m in self.modules():
            if isinstance(m, BayesianConv2d):
                m.deterministic = mode

    @torch.no_grad()
    def mc_inference(self, x, n_samples=5):
        """Monte-Carlo inference: backbone runs once, only the Bayesian last
        layer is sampled n_samples times for epistemic variance.
        """
        self.eval()
        self.set_deterministic(False)

        feat = self.to_feat(x)
        feat = self.body(feat) + feat

        log_var = self.unc_head(feat)
        aleatoric = torch.exp(log_var)

        sr_stack = []
        for _ in range(n_samples):
            sr_stack.append(self.sr_last(feat) + x)
        sr_stack = torch.stack(sr_stack)

        mu_mean = sr_stack.mean(dim=0)
        epistemic = sr_stack.var(dim=0)

        self.set_deterministic(True)
        return mu_mean, aleatoric, epistemic

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    net = LUCID()
    print(f'LUCID parameters: {net.num_parameters() / 1e6:.3f} M')
    x = torch.randn(1, 3, 768, 768)
    mu, log_var = net(x)
    print(f'mu: {mu.shape}  log_var: {log_var.shape}')
