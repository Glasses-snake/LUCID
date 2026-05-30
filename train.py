"""
Training script for LUCID on the FPM-BioCell dataset.

Usage:
    python train.py
 
Data are expected at:
    FPM-BioCell/Dataset/train/{LR,HR}
    FPM-BioCell/Dataset/test/{LR,HR}
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import LUCID

# ===========================================================================
#  Hyper-parameters
# ===========================================================================

TRAIN_LR_DIR = 'FPM-BioCell/Dataset/train/LR'
TRAIN_HR_DIR = 'FPM-BioCell/Dataset/train/HR'

SAVE_DIR        = 'checkpoints'
BATCH_SIZE      = 2
NUM_WORKERS     = 4
EPOCHS          = 350
LEARNING_RATE   = 5e-4
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
SNAPSHOT_EVERY  = 50      # periodic snapshot every N epochs


# ===========================================================================
#  Dataset
# ===========================================================================

class FPMDataset(Dataset):
    """Paired LR-HR loader for the FPM-BioCell 768x768 tiles."""

    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        files = sorted(
            f for f in os.listdir(lr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        )
        hr_set = set(os.listdir(hr_dir))
        self.filenames = [f for f in files if f in hr_set]
        if not self.filenames:
            raise RuntimeError(f'No paired LR-HR files found in {lr_dir}')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = Image.open(os.path.join(self.lr_dir, fname)).convert('RGB')
        hr = Image.open(os.path.join(self.hr_dir, fname)).convert('RGB')
        return TF.to_tensor(lr), TF.to_tensor(hr), fname


# ===========================================================================
#  Loss
# ===========================================================================

def _ssim(pred, target, window_size=11, max_val=1.0):
    """Mean SSIM over the batch (channel-independent, Gaussian window)."""
    C = pred.shape[1]
    coords = torch.arange(window_size, dtype=torch.float32,
                          device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1))[None, None].expand(C, 1, -1, -1)
    pad = window_size // 2

    mu_p = F.conv2d(pred, kernel, groups=C, padding=pad)
    mu_t = F.conv2d(target, kernel, groups=C, padding=pad)
    var_p = F.conv2d(pred * pred, kernel, groups=C, padding=pad) - mu_p ** 2
    var_t = F.conv2d(target * target, kernel, groups=C, padding=pad) - mu_t ** 2
    cov = F.conv2d(pred * target, kernel, groups=C, padding=pad) - mu_p * mu_t

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / \
               ((mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))
    return ssim_map.mean()


def lucid_loss(model_output, target):
    """Composite training objective.

    Components:
        Gaussian NLL (drives mu and log_var jointly)
        FFT L1     (frequency-domain consistency)
        SSIM       (structural similarity)
    """
    mu, log_var = model_output
    log_var = log_var.clamp(-10.0, 10.0)
    precision = torch.exp(-log_var)
    nll = 0.5 * (log_var + (target - mu) ** 2 * precision).mean()

    pred_clamp = mu.clamp(0, 1)
    pred_fft = torch.fft.rfft2(pred_clamp, norm='backward')
    targ_fft = torch.fft.rfft2(target, norm='backward')
    fft = (torch.abs(pred_fft.real - targ_fft.real)
           + torch.abs(pred_fft.imag - targ_fft.imag)).mean()

    ssim = 1.0 - _ssim(pred_clamp, target)

    total = nll + 0.01 * fft + 0.1 * ssim
    return total, {'nll': nll.item(), 'fft': fft.item(),
                   'ssim': ssim.item(), 'total': total.item()}


# ===========================================================================
#  Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='LUCID Training (FPM-BioCell)')
    p.add_argument('--resume', type=str, default=None,
                   help='checkpoint path to resume from')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- Data (training split only) ----
    train_ds = FPMDataset(TRAIN_LR_DIR, TRAIN_HR_DIR)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    print(f'Train: {len(train_ds)} tiles | Device: {device}')

    # ---- Model ----
    model = LUCID().to(device)
    n_params = model.num_parameters() / 1e6
    print(f'LUCID parameters: {n_params:.3f} M')

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )

    # ---- Resume ----
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', -1) + 1
        print(f'Resumed at epoch {start_epoch}')

    # ---- Training loop ----
    print(f'Training: epoch {start_epoch + 1} -> {EPOCHS}')
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        running, n_valid = 0.0, 0
        agg_parts = {}

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=False)
        for lr, hr, _ in pbar:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(lr)
            loss, parts = lucid_loss(out, hr)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running += loss.item()
            n_valid += 1
            for k, v in parts.items():
                agg_parts[k] = agg_parts.get(k, 0.0) + v
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()
        n = max(n_valid, 1)
        avg = running / n
        parts_str = ' | '.join(f'{k}={v / n:.4f}' for k, v in agg_parts.items()
                                if k != 'total')
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f'[{epoch + 1}/{EPOCHS}] loss={avg:.4f} | {parts_str} | '
              f'lr={lr_now:.2e} | time={elapsed:.1f}s')

        # ---- Periodic snapshot ----
        if (epoch + 1) % SNAPSHOT_EVERY == 0 and epoch + 1 != EPOCHS:
            snap = os.path.join(SAVE_DIR, f'epoch_{epoch + 1}.pth')
            torch.save({'epoch': epoch, 'model': model.state_dict()}, snap)

    # ---- Final save ----
    torch.save({
        'epoch': EPOCHS - 1,
        'model': model.state_dict(),
    }, os.path.join(SAVE_DIR, 'last.pth'))
    print(f'Done.  Final weights -> {SAVE_DIR}/last.pth')


if __name__ == '__main__':
    main()
