# LUCID — Lightweight Uncertainty-Calibrated Imaging via Deep learning

A compact, end-to-end deep network for Fourier Ptychographic Microscopy (FPM)
image enhancement with per-pixel uncertainty quantification.

## Highlights

- **Lightweight backbone** with multi-scale Fourier feature modulation.
- **Dual heads** producing the reconstruction `μ` and pixel-wise log-variance.
- **Bayesian last layer** for efficient Monte-Carlo epistemic uncertainty.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0, torchvision
- numpy, matplotlib, Pillow, tqdm

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

A CUDA-capable GPU is recommended for both training and inference.

## Dataset

LUCID is trained and evaluated on **FPM-BioCell**, a paired LR/HR FPM dataset
covering 10 categories of biological samples. The expected directory layout
is:

```
FPM-BioCell/
└── Dataset/
    ├── train/
    │   ├── LR/    # 768×768 RGB tiles (low resolution)
    │   └── HR/    # 768×768 RGB tiles (high resolution, paired)
    └── test/
        ├── LR/
        └── HR/
```

Tile filenames follow the pattern `{category}tile_{row}_{col}.png`, where
`category` is the index 1..10 (e.g. `4tile_2_3.png`). The 10 categories are:

| ID | Sample                | ID | Sample            |
|----|------------------------|----|-------------------|
| 1  | Ascaris eggs           | 6  | Large intestine   |
| 2  | Squamous epithelium    | 7  | Locust meiosis    |
| 3  | Privet leaf            | 8  | Fig fruit         |
| 4  | Lymph node             | 9  | Fish gill         |
| 5  | Testis section         | 10 | Rat tail          |

## Training

```bash
python train.py
```

Training only loads the **training** split — the test set is never read
during training, so no test-time signal influences the saved weights. The
final weights are saved to `checkpoints/last.pth` after the last epoch;
intermediate snapshots (`epoch_50.pth`, `epoch_100.pth`, …) are written
every 50 epochs as recovery points.

## Quick Test

After placing a trained `last.pth` in `checkpoints/`, run:

```bash
python test.py
```

This randomly samples one tile from each of the 10 categories in the test
set, performs MC inference, and writes ten 2×3 figures to `outputs/`:

```
Row 1:  LR (bicubic up)   |  HR (GT)      |  LUCID Recon
Row 2:  Total Uncertainty |  Aleatoric    |  Epistemic
```

Some examples are shown in the `outputs/`.
## License

Released under the MIT License. See `LICENSE` for details.
