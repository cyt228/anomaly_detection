
import os
import argparse
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

from datasets import PairedFolderDataset
from models import UNetRecon
from losses import ReconLoss, psnr

def build_transforms(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # scales to [0,1]
    ])
'''
def save_samples(x, y, yhat, names, out_dir, step_tag):
    os.makedirs(out_dir, exist_ok=True)
    # make a simple grid: [defect | recon | target]
    rows = []
    for i in range(min(6, x.size(0))):
        rows.append(torch.cat([x[i], yhat[i], y[i]], dim=2))  # along width
    grid = torch.cat(rows, dim=1)  # along height
    vutils.save_image(grid, os.path.join(out_dir, f"samples_{step_tag}.png"))
'''

def train_one_epoch(model, loader, optimizer, scaler, device, criterion, epoch, args):
    model.train()
    total_loss, total_psnr = 0.0, 0.0
    for it, (x, y, names) in enumerate(loader, 1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',  enabled=args.amp):
            yhat = model(x)
            loss = criterion(yhat, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            batch_psnr = psnr(yhat.clamp(0,1), y)
        total_loss += loss.item()
        total_psnr += batch_psnr.item()

        if it % 50 == 0:
            print(f"Epoch {epoch} | iter {it}/{len(loader)} | loss {loss.item():.4f} | PSNR {batch_psnr.item():.2f} dB")
    # save sample grid
    #with torch.no_grad():
    #    save_samples(x, y, yhat, names, os.path.join(args.out_dir, "samples"), f"epoch{epoch:03d}")
    return total_loss / len(loader), total_psnr / len(loader)

@torch.no_grad()
def validate(model, loader, device, criterion, epoch, log_dir):
    model.eval()
    total_loss, total_psnr = 0.0, 0.0
    for x, y, names in loader:
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        total_loss += criterion(yhat, y).item()
        total_psnr += psnr(yhat.clamp(0,1), y).item()
    # save samples
    #save_samples(x, y, yhat, names, os.path.join(log_dir, "samples"), f"val_epoch{epoch:03d}")
    return total_loss / len(loader), total_psnr / len(loader)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_defect", required=True, help="path to train/defect folder")
    ap.add_argument("--train_clean", required=True, help="path to train/defect_free folder")
    ap.add_argument("--val_defect", required=False, help="optional path to val/defect folder")
    ap.add_argument("--val_clean", required=False, help="optional path to val/defect_free folder")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--backbone", choices=["unet", "unetpp"], default="unet")
    ap.add_argument("--feature_scale", type=int, default=1)
    ap.add_argument("--no_sigmoid", action="store_true", help="disable sigmoid at the output")
    ap.add_argument("--amp", action="store_true", help="use mixed precision")
    ap.add_argument("--out_dir", default="checkpoints_recon")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = build_transforms(args.img_size)

    ds_train = PairedFolderDataset(args.train_defect, args.train_clean, transform=transform)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    dl_val = None
    if args.val_defect and args.val_clean:
        ds_val = PairedFolderDataset(args.val_defect, args.val_clean, transform=transform)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNetRecon(
        backbone=args.backbone,
        feature_scale=args.feature_scale,
        out_activation="none" if args.no_sigmoid else "sigmoid",
        in_channels=3, out_channels=3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    criterion = ReconLoss(alpha=1.0, beta=0.1)  # L1 + 0.1*(1-SSIM)
    os.makedirs(args.out_dir, exist_ok=True)
    best_val = -1e9  # use PSNR for model selection

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_psnr = train_one_epoch(model, dl_train, optimizer, scaler, device, criterion, epoch, args)
        print(f"[Train] epoch {epoch}: loss={tr_loss:.4f}, PSNR={tr_psnr:.2f} dB")

        val_psnr = None
        if dl_val is not None:
            va_loss, va_psnr = validate(model, dl_val, device, criterion, epoch, args.out_dir)
            print(f"[Valid] epoch {epoch}: loss={va_loss:.4f}, PSNR={va_psnr:.2f} dB")
            val_psnr = va_psnr

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        }, os.path.join(args.out_dir, "last.pt"))

        # Save best
        score = tr_psnr if val_psnr is None else val_psnr
        if score > best_val:
            best_val = score
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "best_metric": score,
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"[*] Saved best model at epoch {epoch} with PSNR={score:.2f} dB")

if __name__ == "__main__":
    main()
