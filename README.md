# PASCAL VOC Efficient Segmentation Challenge — Group 30

Efficient semantic segmentation model for PASCAL VOC 2012.
Built with **NanoSegNet** — Full MobileNetV3-Small backbone (ImageNet pretrained) +
LRASPP-style decoder, trained at 192×192 input to maximise the Dice/GFLOPs ratio.

---

## Results

| Metric | Value |
|---|---|
| **Macro Dice Score (Val, 21 classes)** | **0.6421** |
| **FLOPs per image** | **0.050 GFLOPs** (at 192×192 inference) |
| **Dice / GFLOPs ratio** | **12.84** |
| **Parameters** | **1.079 M** |
| Input → Output | `(3, H, W)` → `(H, W)` binary mask (0=background, 255=foreground) |
| Classes | 21 internally (background + 20 VOC objects) → combined to binary output |
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
├── inference.py           # STEP 3 — Generate binary masks for submission
│
├── requirements.txt       # Python dependencies
├── best_model.pth         # Saved best model checkpoint
├── training_log.csv       # Epoch-wise loss and dice log
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
└── 30_output/             # OUTPUT — predicted binary masks for submission
    └── <name>.jpg         # Same filename as input (binary: 0=black, 255=white)
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

Output: `best_model.pth`, `training_log.csv`

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

Output: `30_output/<original_filename>` — 16135 binary masks
- Background pixels → **0 (black)**
- Foreground pixels (any of classes 1–20) → **255 (white)**
- Output filename is **identical** to input filename

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
- `model.eval()` → integer mask `(B, H, W)` (values 0–20), converted to binary in inference
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

**Augmentations (training — includes robustness to noise/corruption):**
- RandomResizedCrop (scale 0.5–1.0) @ 192×192
- HorizontalFlip
- ShiftScaleRotate
- GaussNoise / ISONoise
- GaussianBlur / MotionBlur / MedianBlur
- JPEG Compression (quality 40–100)
- ColorJitter + RandomGamma
- ImageNet Normalize

---

## Competition Compliance

| Requirement | Status |
|---|---|
| PyTorch framework only | ✅ |
| Input `(3, 300, 300)` → Output `(300, 300)` | ✅ Model handles any input size |
| Binary output mask (0=background, 255=foreground) | ✅ |
| Output filename identical to input filename | ✅ |
| `--in_dir` / `--out_dir` arguments | ✅ |
| Output folder `30_output/` | ✅ |
| 80/20 split from `train.txt` only | ✅ |
| `val.txt` never used during training | ✅ |
| Robustness to noise / blur / JPEG / contrast | ✅ (augmentation pipeline) |

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
