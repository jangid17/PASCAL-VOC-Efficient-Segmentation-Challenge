# PASCAL VOC Efficient Segmentation Challenge — Group 30

Efficient semantic segmentation model for PASCAL VOC 2012.
Built with **NanoSegNet** — Full MobileNetV3-Small backbone (ImageNet pretrained) +
LRASPP-style decoder, trained at 192×192 input to maximise the Dice/GFLOPs ratio.

---

## Results

| Metric | Value |
|---|---|
| **Macro Dice Score (Val, 21 classes)** | **0.6421** |
| **FLOPs per image** | **0.128 GFLOPs** (at 300×300 inference) |
| **Dice / GFLOPs ratio** | **5.02** |
| **Parameters** | **1.079 M** |
| Input → Output | `(3, 300, 300)` → `(300, 300)` binary mask (0=background, 255=foreground) |
| Best epoch | 147 / 150 |
| GPU inference | ~5 ms / image |

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
├── best_model.pth         # Saved best model checkpoint (excluded from git)
├── training_log.csv       # Epoch-wise loss and dice log
│
├── VOC2012_train_val/     # Training dataset (from Kaggle, excluded from git)
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   └── ImageSets/Segmentation/
│       ├── train.txt         # 1464 training IDs (used for training)
│       └── val.txt           # competition TEST SET — do not use
│
├── VOC2012_test/          # Test dataset (images only, excluded from git)
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
  FLOPs  : 0.128 GFLOPs
  Params : 1.079 M
========================================
```

### STEP 3 — Inference (Submission)

```bash
python3 inference.py --in_dir=VOC2012_test/JPEGImages/ --out_dir=30_output/
```

- Input images resized to **300×300** before the forward pass
- Model forward pass **directly** outputs `(300, 300)` integer class mask (0–20)
- Per-class predictions combined into one **binary mask**: foreground=255, background=0
- Output saved with **identical filename** as input (e.g. `2008_000001.jpg`)

---

## Model Architecture

**NanoSegNet** — Full MobileNetV3-Small + LRASPP decoder

```
Encoder (ImageNet pretrained):
  enc_low  : features[0:4]  → 24ch  @ 1/8  (37×37 at 300 input)
  enc_high : features[4:13] → 576ch @ 1/32 ( 9×9  at 300 input)

Decoder (trained from scratch):
  spatial branch : Conv 576→128, BN, Hardswish        @ 9×9
  global branch  : AdaptiveAvgPool + Conv 576→128 + Hardsigmoid → 1×1
  attention      : spatial × global → 128ch           @ 9×9
  upsample 4×    :                                    @ 37×37
  low_conv       : Conv 24→32, BN, ReLU               @ 37×37
  concat         : [128, 32] = 160ch                  @ 37×37
  classifier     : Conv 160→21                        @ 37×37
  upsample 8×    :                                    → 300×300
```

- `model.train()` → logits `(B, 21, H, W)`
- `model.eval()` → integer class mask `(B, H, W)` values 0–20
- Backbone: ImageNet pretrained (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`)
- Decoder: trained from scratch with 10× higher LR than backbone

---

## Training Details

| Setting | Value |
|---|---|
| Training input size | 192 × 192 |
| Inference input size | 300 × 300 |
| Batch size | 16 |
| Optimizer | AdamW (weight decay 1e-4) |
| Backbone LR | 1e-4 |
| Decoder LR | 1e-3 |
| Scheduler | CosineAnnealingLR (T_max=150, eta_min=1e-6) |
| Loss | 0.5 × CrossEntropy (class-weighted) + 0.5 × Dice |
| Epochs | 150 |
| Mixed precision | AMP (torch.amp.autocast) |
| Best Val Dice | 0.6421 |

**Augmentations (training — robustness to noise & corruption):**
- RandomResizedCrop (scale 0.5–1.0) @ 192×192
- HorizontalFlip, ShiftScaleRotate
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
| End-to-end: input `(3, 300, 300)` → output `(300, 300)` via forward pass | ✅ |
| No external post-processing | ✅ resize happens before forward pass |
| Integer class labels 0–20 (model output) | ✅ |
| Binary output mask: background=black, foreground=white | ✅ |
| Per-class masks combined into one final binary mask | ✅ |
| Output filename identical to input filename | ✅ |
| `--in_dir` / `--out_dir` arguments | ✅ |
| Output folder `30_output/` | ✅ |
| 80/20 split from `train.txt` only | ✅ |
| `val.txt` (competition test) never used during training | ✅ |
| Robustness to noise / blur / JPEG / contrast | ✅ augmentation pipeline |

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
