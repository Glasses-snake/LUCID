"""
Quick Test for LUCID inference on FPM-BioCell.

For each of the 10 biological categories in the test set, a tile is randomly
sampled and run through the trained LUCID model. Ten 2x3 figures are written
to `outputs/`:

    Row 1:  LR (bicubic up)   |  HR (GT)        |  LUCID Recon
    Row 2:  Total Uncertainty |  Aleatoric      |  Epistemic

Usage:
    python test.py
    python test.py --checkpoint checkpoints/last.pth --seed 42

Tile filenames follow the FPM-BioCell convention `{cat}tile_{r}_{c}.png`,
where `cat` is the category index 1..10.
"""

import argparse
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from model import LUCID

CATEGORIES = {
    '1':  'Ascaris eggs',
    '2':  'Squamous epithelium',
    '3':  'Privet leaf',
    '4':  'Lymph node',
    '5':  'Testis section',
    '6':  'Large intestine',
    '7':  'Locust meiosis',
    '8':  'Fig fruit',
    '9':  'Fish gill',
    '10': 'Rat tail',
}

DEFAULT_LR_DIR = 'FPM-BioCell/Dataset/test/LR'
DEFAULT_HR_DIR = 'FPM-BioCell/Dataset/test/HR'


def parse_args():
    p = argparse.ArgumentParser(description='LUCID quick demo (FPM-BioCell)')
    p.add_argument('--checkpoint', type=str, default='checkpoints/last.pth',
                   help='trained LUCID weights')
    p.add_argument('--lr_dir', type=str, default=DEFAULT_LR_DIR)
    p.add_argument('--hr_dir', type=str, default=DEFAULT_HR_DIR)
    p.add_argument('--output_dir', type=str, default='outputs')
    p.add_argument('--mc_samples', type=int, default=5,
                   help='Monte-Carlo samples for the Bayesian last layer')
    p.add_argument('--seed', type=int, default=None,
                   help='random seed for tile selection')
    return p.parse_args()


def category_of(filename):
    """Extract category index from filenames like '10tile_0_4.png' -> '10'."""
    idx = filename.find('tile')
    return filename[:idx] if idx > 0 else ''


def group_by_category(lr_dir, hr_dir):
    """Return {cat_id: [filenames]} for tiles that exist in both directories."""
    lr_set = {f for f in os.listdir(lr_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))}
    hr_set = set(os.listdir(hr_dir))
    paired = sorted(lr_set & hr_set)

    groups = defaultdict(list)
    for fname in paired:
        cat = category_of(fname)
        if cat in CATEGORIES:
            groups[cat].append(fname)
    return groups


def to_uint8(t):
    """(C, H, W) tensor in [0, 1] -> (H, W, C) uint8 numpy."""
    return (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def _unc_to_2d(t):
    """(C, H, W) or (1, C, H, W) tensor -> (H, W) numpy averaged over channels."""
    arr = t.squeeze().detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    return arr


def render_panel(lr, hr, mu, aleatoric, epistemic, title, save_path):
    """2x3 figure.

    Row 1:  LR (bicubic up)   | HR (GT)    | LUCID Recon
    Row 2:  Total Uncertainty | Aleatoric  | Epistemic
    """
    h, w = hr.shape[-2:]
    lr_up = F.interpolate(lr.unsqueeze(0), size=(h, w),
                          mode='bicubic', align_corners=False).squeeze(0)

    ale = _unc_to_2d(aleatoric)
    epi = _unc_to_2d(epistemic)
    total = ale + epi

    def _vmax(arr):
        return max(float(np.percentile(arr, 99.5)), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ---- Row 1: image panels ----
    axes[0, 0].imshow(to_uint8(lr_up))
    axes[0, 0].set_title('LR (bicubic up)')

    axes[0, 1].imshow(to_uint8(hr))
    axes[0, 1].set_title('HR (GT)')

    axes[0, 2].imshow(to_uint8(mu.clamp(0, 1)))
    axes[0, 2].set_title('LUCID Recon')

    # ---- Row 2: uncertainty maps ----
    im_t = axes[1, 0].imshow(total, cmap='jet', vmin=0, vmax=_vmax(total))
    axes[1, 0].set_title('Total Uncertainty')
    plt.colorbar(im_t, ax=axes[1, 0], shrink=0.8)

    im_a = axes[1, 1].imshow(ale, cmap='jet', vmin=0, vmax=_vmax(ale))
    axes[1, 1].set_title('Aleatoric')
    plt.colorbar(im_a, ax=axes[1, 1], shrink=0.8)

    im_e = axes[1, 2].imshow(epi, cmap='jet', vmin=0, vmax=_vmax(epi))
    axes[1, 2].set_title('Epistemic')
    plt.colorbar(im_e, ax=axes[1, 2], shrink=0.8)

    for row in axes:
        for ax in row:
            ax.axis('off')

    fig.suptitle(title, fontsize=14, y=1.0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def run(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load model ----
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    model = LUCID().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    # ---- Discover tiles per category ----
    groups = group_by_category(args.lr_dir, args.hr_dir)
    if not groups:
        raise RuntimeError(f'No paired tiles found under {args.lr_dir} / {args.hr_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Found categories: {sorted(groups.keys(), key=int)}')
    print(f'Writing {len(CATEGORIES)} figures to {args.output_dir}/')

    # ---- Process each category (1..10) ----
    for cat_id in sorted(CATEGORIES.keys(), key=int):
        cat_name = CATEGORIES[cat_id]
        candidates = groups.get(cat_id, [])
        if not candidates:
            print(f'  [{cat_id:>2}] {cat_name}: no tiles, skipped')
            continue

        fname = random.choice(candidates)
        lr_pil = Image.open(os.path.join(args.lr_dir, fname)).convert('RGB')
        hr_pil = Image.open(os.path.join(args.hr_dir, fname)).convert('RGB')
        lr = TF.to_tensor(lr_pil).to(device)
        hr = TF.to_tensor(hr_pil).to(device)

        mu, aleatoric, epistemic = model.mc_inference(
            lr.unsqueeze(0), n_samples=args.mc_samples)
        mu = mu.squeeze(0)
        aleatoric = aleatoric.squeeze(0)
        epistemic = epistemic.squeeze(0)

        # PSNR for the caption.
        mse = F.mse_loss(mu.clamp(0, 1), hr).item()
        psnr = 20 * np.log10(1.0) - 10 * np.log10(max(mse, 1e-12))

        title = (f'Category {cat_id}: {cat_name}  |  '
                 f'tile: {fname}  |  PSNR={psnr:.2f} dB')
        out_path = os.path.join(args.output_dir,
                                f'{cat_id.zfill(2)}_{cat_name.replace(" ", "_")}.png')
        render_panel(lr.cpu(), hr.cpu(), mu.cpu(),
                     aleatoric.cpu(), epistemic.cpu(), title, out_path)
        print(f'  [{cat_id:>2}] {cat_name:<22} -> {os.path.basename(out_path)}  '
              f'(PSNR {psnr:.2f} dB, sample: {fname})')

    print(f'\nDone. Outputs at {os.path.abspath(args.output_dir)}/')


if __name__ == '__main__':
    run(parse_args())
