
import os, csv, argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from datasets import PairedFolderDataset
from models import UNetRecon
from losses import psnr

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--defect", required=True, help="folder with defect images")
    ap.add_argument("--clean", required=True, help="folder with clean target images")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="recon_outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])
    ds = PairedFolderDataset(args.defect, args.clean, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # load model
    ckpt = torch.load(args.ckpt, map_location=device)
    margs = ckpt["args"]
    model = UNetRecon(
        backbone=margs.get("backbone", "unet"),
        feature_scale=margs.get("feature_scale", 1),
        out_activation="none" if margs.get("no_sigmoid", False) else "sigmoid",
        in_channels=3, out_channels=3
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = [["filename", "psnr_db"]]
    for i, (x, y, names) in enumerate(dl):
        x, y = x.to(device), y.to(device)
        yhat = model(x).clamp(0,1)
        # save images
        for j in range(yhat.size(0)):
            vutils.save_image(yhat[j], os.path.join(args.out_dir, f"{names[j]}"))
        # metrics
        batch_psnr = psnr(yhat, y).item()
        for name in names:
            rows.append([name, f"{batch_psnr:.4f}"])

    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Saved recon images to {args.out_dir} and metrics.csv")

if __name__ == "__main__":
    main()
