"""
train.py — PASCAL VOC 2012 Efficient Segmentation
==================================================
Splits the official trainval set 80 / 20 (train / val) and trains
EfficientSegNet (DeepLabV3+ / MobileNetV3-Large) using Combined
CE + Dice loss.  Saves the best checkpoint and a CSV training log.

Usage:
    python train.py
    python train.py --data_root /path/to/VOC2012 --epochs 40 --batch_size 8
"""

import os
import csv
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import VOCDataset
from augmentations import get_train_transforms, get_val_transforms
from model import get_model
from losses import CombinedLoss

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 21,
    ignore_index: int = 255,
) -> float:
    """
    Macro-averaged Dice Similarity Coefficient over all `num_classes` classes.
    Pixels with label == ignore_index are excluded from numerator/denominator.
    If a class is absent in both prediction and ground-truth, its Dice = 1.0.
    """
    preds = torch.argmax(preds, dim=1)  # (B, H, W)
    valid_mask = (targets != ignore_index)

    total_dice = 0.0
    for cls in range(num_classes):
        pred_cls   = (preds   == cls) & valid_mask
        target_cls = (targets == cls) & valid_mask

        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()

        denom = 2.0 * tp + fp + fn
        if denom == 0:
            total_dice += 1.0          # class absent in pred & GT → perfect
        else:
            total_dice += (2.0 * tp / denom).item()

    return total_dice / num_classes


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="  train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model.get_logits(images)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()      # OneCycleLR steps once per batch
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, num_classes=21):
    model.eval()
    total_dice = 0.0
    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model.get_logits(images)
        total_dice += macro_dice_score(logits, masks, num_classes)
    return total_dice / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="VOC2012_train_val")
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--checkpoint", default="best_model.pth")
    p.add_argument("--log_csv",    default="training_log.csv")
    return p.parse_args()


def main():
    args = parse_args()

    ids_file = os.path.join(args.data_root, "ImageSets/Segmentation/train.txt")
    with open(ids_file) as f:
        ids = f.read().splitlines()

    train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=42)
    print(f"Train: {len(train_ids)}  |  Val: {len(val_ids)}")

    train_ds = VOCDataset(args.data_root, train_ids, transform=get_train_transforms())
    val_ds   = VOCDataset(args.data_root, val_ids,   transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4,              shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = get_model(num_classes=21).to(device)
    criterion = CombinedLoss(num_classes=21)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           steps_per_epoch=len(train_loader),
                           epochs=args.epochs, pct_start=0.1)
    scaler    = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    best_dice = 0.0
    log_rows  = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, scaler, scheduler, device)
        val_dice   = validate(model, val_loader, device)

        print(f"  Loss: {train_loss:.4f}  |  Val Dice: {val_dice:.4f}  |  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        log_rows.append([epoch, train_loss, val_dice])

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  ** Saved best model (Dice={best_dice:.4f})")

    # Save CSV log
    with open(args.log_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_dice"])
        w.writerows(log_rows)

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")
    print(f"Checkpoint: {args.checkpoint}  |  Log: {args.log_csv}")


if __name__ == "__main__":
    main()
