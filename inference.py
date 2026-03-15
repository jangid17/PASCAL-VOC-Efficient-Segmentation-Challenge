"""
inference.py — PASCAL VOC Efficient Segmentation Challenge (Group 30)
======================================================================
Runs the trained model on all images in a folder and writes 21-class
segmentation masks (pixel values = class index 0-20) to an output folder.

Output filename format: {original_name}_mask.png
Output folder:         30_output/  (group number_output)

Usage (as per submission guidelines):
    python inference.py --in_dir=/path/to/test_images/ --out_dir=30_output/
    python inference.py --in_dir=/path/to/test_images/
"""

import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_transform():
    return A.Compose([
        A.Resize(192, 192),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = get_model(num_classes=21)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run segmentation inference on a folder of images.")
    p.add_argument("--in_dir",     required=True,
                   help="Folder containing test images (.jpg / .jpeg / .png)")
    p.add_argument("--out_dir",    default="30_output",
                   help="Output folder for segmentation masks (default: 30_output)")
    p.add_argument("--model_path", default="best_model.pth",
                   help="Path to trained model checkpoint")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---- Output folder -------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output : {args.out_dir}/")

    # ---- Model ---------------------------------------------------------------
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.model_path}\n"
            "Run train.py first to produce best_model.pth."
        )
    model     = load_model(args.model_path, device)
    transform = get_transform()

    # ---- Image list ----------------------------------------------------------
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in os.listdir(args.in_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not image_files:
        print("No images found in the specified folder.")
        return

    print(f"Images : {len(image_files)}\n")

    # ---- Inference -----------------------------------------------------------
    with torch.no_grad():
        for fname in image_files:
            img_path = os.path.join(args.in_dir, fname)

            # Read & preprocess
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [SKIP] Cannot read {fname}")
                continue
            orig_h, orig_w = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            tensor = transform(image=image)["image"].unsqueeze(0).to(device)

            # Forward pass — returns integer class mask (H, W), values 0-20
            class_mask = model(tensor).squeeze(0).cpu().numpy().astype(np.uint8)

            # Resize mask back to original image dimensions (nearest-neighbour
            # preserves integer class indices exactly)
            class_mask = cv2.resize(class_mask, (orig_w, orig_h),
                                    interpolation=cv2.INTER_NEAREST)

            # Output filename: {original_stem}_mask.png
            stem     = os.path.splitext(fname)[0]
            out_name = f"{stem}_mask.png"
            out_path = os.path.join(args.out_dir, out_name)
            cv2.imwrite(out_path, class_mask)

    print(f"\nDone. {len(image_files)} masks saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
