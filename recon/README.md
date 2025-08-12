
# Reconstruction Training (UNet-family backbone)

This is a **minimal, ready-to-run** image reconstruction setup that uses your uploaded **UNet-family** as the backbone.

## Folder layout (example)
```
project/
├── UNet-family/                 # put the provided UNet-family folder here
└── recon/                       # this package (you can rename)
    ├── train_recon.py
    ├── eval_recon.py
    ├── datasets.py
    ├── models.py
    └── losses.py
```

## Data layout (paired by filename)
```
data/
├── train/
│   ├── defect/        # e.g., 000.png, 001.png, ...
│   └── defect_free/   # matching names 000.png, 001.png, ...
└── val/
    ├── defect/
    └── defect_free/
```

> We pair images **by identical filenames** and ignore files that do not exist on both sides.

## Quick start (Kaggle/Colab/Local)
Make sure **UNet-family** sits next to these scripts (or set env var `UNET_FAMILY_PATH` to its directory). Then:

```bash
# Train (UNet backbone, mixed precision on, save to checkpoints_recon/)
python train_recon.py \
  --train_defect data/train/defect \
  --train_clean  data/train/defect_free \
  --val_defect   data/val/defect \
  --val_clean    data/val/defect_free \
  --img_size 256 \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-3 \
  --backbone unet \
  --amp \
  --out_dir checkpoints_recon
```

The model picks **UNet** by default; use `--backbone unetpp` for **UNet++**.

### Evaluate / export reconstructions
```bash
python eval_recon.py \
  --ckpt checkpoints_recon/best.pt \
  --defect data/test/defect \
  --clean  data/test/defect_free \
  --img_size 256 \
  --batch_size 32 \
  --out_dir out/recon/test
```

- Reconstructed images are written to `out/recon/test/`.
- A `metrics.csv` with **PSNR** per-batch is also produced.

## Tips
- The loss defaults to **L1 + 0.1·(1-SSIM)**. You can tune this in `train_recon.py` (see `ReconLoss` in `losses.py`).
- If your targets are already in [0,1], keep the default **Sigmoid** output. Otherwise, add `--no_sigmoid`.
- If your images are not square, change the `Resize` transform or use `CenterCrop` instead.
- If you have only *train* and want to create a validation split, just duplicate the train paths and skip `--val_*` first, or pre-split your folders.
```

