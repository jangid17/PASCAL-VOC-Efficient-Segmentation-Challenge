"""
visualize.py — Visualize predictions vs ground truth
======================================================
Renders a side-by-side grid: image | GT mask | predicted mask.
Useful for qualitative inspection during development.

Usage:
    python visualize.py
    python visualize.py --data_root /path/to/VOC2012 --num_samples 8 --out vis.png
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from augmentations import get_val_transforms
from dataset import VOCDataset
from model import get_model

# VOC 2012 colour palette (21 classes)
VOC_PALETTE = np.array([
    [0,   0,   0  ],  # 0  background
    [128, 0,   0  ],  # 1  aeroplane
    [0,   128, 0  ],  # 2  bicycle
    [128, 128, 0  ],  # 3  bird
    [0,   0,   128],  # 4  boat
    [128, 0,   128],  # 5  bottle
    [0,   128, 128],  # 6  bus
    [128, 128, 128],  # 7  car
    [64,  0,   0  ],  # 8  cat
    [192, 0,   0  ],  # 9  chair
    [64,  128, 0  ],  # 10 cow
    [192, 128, 0  ],  # 11 diningtable
    [64,  0,   128],  # 12 dog
    [192, 0,   128],  # 13 horse
    [64,  128, 128],  # 14 motorbike
    [192, 128, 128],  # 15 person
    [0,   64,  0  ],  # 16 pottedplant
    [128, 64,  0  ],  # 17 sheep
    [0,   192, 0  ],  # 18 sofa
    [128, 192, 0  ],  # 19 train
    [0,   64,  128],  # 20 tvmonitor
], dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map (H, W) class-index mask to (H, W, 3) RGB image."""
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(len(VOC_PALETTE)):
        rgb[mask == cls] = VOC_PALETTE[cls]
    return rgb


def denormalize(tensor):
    """Reverse ImageNet normalisation and return (H, W, 3) uint8."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = (img * std + mean) * 255
    return img.clip(0, 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="VOC2012_train_val")
    p.add_argument("--checkpoint",  default="best_model.pth")
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--out",         default="visualisation.png")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ids_file = os.path.join(args.data_root, "ImageSets/Segmentation/trainval.txt")
    with open(ids_file) as f:
        ids = f.read().splitlines()

    _, val_ids = train_test_split(ids, test_size=0.2, random_state=42)
    val_ids    = val_ids[: args.num_samples]

    dataset = VOCDataset(args.data_root, val_ids, transform=get_val_transforms())

    model = get_model(num_classes=21)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    n   = len(val_ids)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    with torch.no_grad():
        for i, (image_t, mask_t) in enumerate(dataset):
            inp  = image_t.unsqueeze(0).to(device)
            pred = model(inp).squeeze(0).cpu().numpy().astype(np.uint8)

            img_np  = denormalize(image_t)
            gt_np   = colorize(mask_t.numpy().astype(np.uint8))
            pred_np = colorize(pred)

            axes[i][0].imshow(img_np);   axes[i][0].set_title("Image")
            axes[i][1].imshow(gt_np);    axes[i][1].set_title("Ground Truth")
            axes[i][2].imshow(pred_np);  axes[i][2].set_title("Prediction")
            for ax in axes[i]:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
