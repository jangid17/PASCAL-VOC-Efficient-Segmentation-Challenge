"""
evaluate.py — PASCAL VOC Efficient Segmentation Challenge
==========================================================
Evaluates the best model checkpoint on the held-out validation split
(same 20 % split used in train.py) and reports:
  • Macro Dice Score (21 classes, competition metric)
  • FLOPs per forward pass (single 300×300 image)
  • Parameter count

Usage:
    python evaluate.py
    python evaluate.py --data_root /path/to/VOC2012 --checkpoint best_model.pth
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import VOCDataset
from augmentations import get_val_transforms
from model import get_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 21,
    ignore_index: int = 255,
) -> float:
    """Macro-averaged DSC over all classes (competition metric)."""
    preds = torch.argmax(preds, dim=1)
    valid = targets != ignore_index

    total = 0.0
    for cls in range(num_classes):
        pred_c   = (preds   == cls) & valid
        target_c = (targets == cls) & valid

        tp    = (pred_c & target_c).sum().float()
        fp    = (pred_c & ~target_c).sum().float()
        fn    = (~pred_c & target_c).sum().float()
        denom = 2.0 * tp + fp + fn

        total += 1.0 if denom == 0 else (2.0 * tp / denom).item()

    return total / num_classes


# ---------------------------------------------------------------------------
# FLOPs helper
# ---------------------------------------------------------------------------

def compute_flops(model: torch.nn.Module, device: torch.device):
    dummy = torch.randn(1, 3, 300, 300, device=device)

    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        print(f"  FLOPs  : {flops / 1e9:.3f} GFLOPs  ({flops:.2e} FLOPs)")
        print(f"  Params : {params / 1e6:.3f} M")
        return flops, params
    except ImportError:
        pass

    # Fallback: fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        n = flops.total()
        params = sum(p.numel() for p in model.parameters())
        print(f"  FLOPs  : {n / 1e9:.3f} GFLOPs  ({n:.2e} FLOPs)")
        print(f"  Params : {params / 1e6:.3f} M")
        return n, params
    except ImportError:
        pass

    params = sum(p.numel() for p in model.parameters())
    print(f"  Params : {params / 1e6:.3f} M")
    print("  FLOPs  : install 'thop' or 'fvcore' for FLOPs count.")
    return None, params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="VOC2012_train_val")
    p.add_argument("--checkpoint", default="best_model.pth")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers",type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---- Data ----------------------------------------------------------------
    # Use train.txt only — the VOC val split is the test set and must NOT be touched
    ids_file = os.path.join(args.data_root, "ImageSets/Segmentation/train.txt")
    with open(ids_file) as f:
        ids = f.read().splitlines()

    _, val_ids = train_test_split(ids, test_size=0.2, random_state=42)
    print(f"Val samples : {len(val_ids)}")

    val_ds     = VOCDataset(args.data_root, val_ids, transform=get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # ---- Model ---------------------------------------------------------------
    model = get_model(num_classes=21)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    # ---- Dice ----------------------------------------------------------------
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            logits = model.get_logits(images)
            total_dice += macro_dice_score(logits, masks)

    mean_dice = total_dice / len(val_loader)
    print(f"\n{'=' * 40}")
    print(f"  Macro Dice Score : {mean_dice:.4f}")

    # ---- FLOPs ---------------------------------------------------------------
    print(f"{'=' * 40}")
    compute_flops(model, device)
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
