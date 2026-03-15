# PASCAL VOC Efficient Segmentation Challenge — Group 30

Efficient semantic segmentation model for PASCAL VOC 2012.
Built with **NanoSegNet** — Full MobileNetV3-Small backbone (ImageNet pretrained) +
LRASPP-style decoder, trained at 192×192 input to maximise the Dice/GFLOPs ratio.

---

## Results

| Metric | Value |
|---|---|
| **Macro Dice Score (Val, 21 classes)** | **0.6421** |
| **FLOPs per image** | **0.050 GFLOPs** (at 192×192 input) |
| **Dice / GFLOPs ratio** | **12.84** |
| **Parameters** | **1.079 M** |
| Input → Output | `(3, 192, 192)` → `(H, W)` integer mask (class 0–20) |
| Classes | 21 (background + 20 VOC objects) |
| Best epoch | 147 / 150 |
| GPU inference | ~3 ms / image |

---

## Dataset

Downloaded from Kaggle:
**https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset**

- `VOC2012_train_val/` — Training data (1464 images with segmentation masks)
- `VOC2012_test/` — Test data (16135 images, no masks — competition test set)

> **Important:** Only `ImageSets/Segmentation/train.txt` (1464 images) is used for
> training. It is split 80/20 → 1171 train / 293 val. The VOC validation split
> (`val.txt`) is the competition test set and is **never touched during training**.

---

## Project Structure

```
PASCAL-VOC-Efficient-Segmentation-Challenge-/
│
├── model.py               # NanoSegNet architecture
├── dataset.py             # VOCDataset — loads images + masks
├── augmentations.py       # Train & val transform pipelines (192×192)
├── losses.py              # Combined CrossEntropy + Dice loss
│
├── train.py               # STEP 1 — Train the model
├── evaluate.py            # STEP 2 — Evaluate Dice + FLOPs
├── inference.py           # STEP 3 — Generate masks for submission
│
├── requirements.txt       # Python dependencies
├── best_model.pth         # Saved best model checkpoint
├── training_log.csv       # Epoch-wise loss and dice log
├── training_stdout.log    # Full training output log
│
├── VOC2012_train_val/     # Training dataset (from Kaggle)
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   └── ImageSets/Segmentation/
│       ├── train.txt         # 1464 training IDs (used for training)
│       └── val.txt           # competition TEST SET — do not use
│
├── VOC2012_test/          # Test dataset (images only)
│   └── JPEGImages/
│
└── 30_output/             # OUTPUT — predicted masks for submission
    └── <name>_mask.png    # One mask per test image
```

---

## Setup

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Running the Code

### STEP 1 — Train

```bash
python3 train.py --epochs 150 --batch_size 16 --lr 1e-3 --backbone_lr 1e-4
```

Output: `best_model.pth`, `training_log.csv`, `training_stdout.log`

### STEP 2 — Evaluate

```bash
python3 evaluate.py --checkpoint best_model.pth
```

Expected output:
```
========================================
  Macro Dice Score : 0.6421
========================================
  FLOPs  : 0.050 GFLOPs
  Params : 1.079 M
========================================
```

### STEP 3 — Inference (Submission)

```bash
python3 inference.py --in_dir=VOC2012_test/JPEGImages/ --out_dir=30_output/
```

Output: `30_output/<name>_mask.png` — 16135 integer class masks (0–20)

---

## Model Architecture

**NanoSegNet** — Full MobileNetV3-Small + LRASPP decoder

```
Encoder (ImageNet pretrained):
  enc_low  : features[0:4]  → 24ch  @ 1/8  (24×24 at 192 input)
  enc_high : features[4:13] → 576ch @ 1/32 ( 6×6  at 192 input)

Decoder (trained from scratch):
  spatial branch : Conv 576→128, BN, Hardswish        @ 6×6
  global branch  : AdaptiveAvgPool + Conv 576→128 + Hardsigmoid → 1×1
  attention      : spatial × global → 128ch           @ 6×6
  upsample 4×    :                                    @ 24×24
  low_conv       : Conv 24→32, BN, ReLU               @ 24×24
  concat         : [128, 32] = 160ch                  @ 24×24
  classifier     : Conv 160→21                        @ 24×24
  upsample 8×    :                                    → 192×192
```

- `model.train()` → logits `(B, 21, H, W)`
- `model.eval()` → integer mask `(B, H, W)` (values 0–20)
- Backbone: ImageNet pretrained (MobileNet_V3_Small_Weights.IMAGENET1K_V1)
- Decoder: trained from scratch with 10× higher LR

---

## Training Details

| Setting | Value |
|---|---|
| Input size | 192 × 192 |
| Batch size | 16 |
| Optimizer | AdamW (weight decay 1e-4) |
| Backbone LR | 1e-4 |
| Decoder LR | 1e-3 |
| Scheduler | CosineAnnealingLR (T_max=150, eta_min=1e-6) |
| Loss | 0.5 × CrossEntropy (class-weighted) + 0.5 × Dice |
| Epochs | 150 |
| Mixed precision | AMP (torch.amp.autocast) |
| Gradient clipping | max_norm=5.0 |
| Best Val Dice | 0.6421 |

**Augmentations (training):**
- RandomResizedCrop (scale 0.5–1.0) @ 192×192
- HorizontalFlip
- ShiftScaleRotate
- GaussNoise / ISONoise
- GaussianBlur / MotionBlur / MedianBlur
- JPEG Compression (quality 40–100)
- ColorJitter + RandomGamma
- ImageNet Normalize

---

## Important Notes

- **Dataset:** `train.txt` only (1464 images) — split 80/20
- **`val.txt` is the competition test set** — never used during training
- Void/boundary pixels (label=255) are ignored in loss and metric
- Output masks are saved as **PNG** (lossless, preserves class indices exactly)
- Output filename format: `{original_stem}_mask.png`
- Masks are resized to **original image dimensions** (nearest-neighbour)

---

## Quick Reference

```bash
# 1. Train
python3 train.py --epochs 150 --batch_size 16 --lr 1e-3 --backbone_lr 1e-4

# 2. Evaluate
python3 evaluate.py

# 3. Generate submission masks
python3 inference.py --in_dir=VOC2012_test/JPEGImages/ --out_dir=30_output/
```
