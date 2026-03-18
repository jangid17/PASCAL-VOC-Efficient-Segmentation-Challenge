# PASCAL VOC Efficient Segmentation Challenge — Group 30

Efficient semantic segmentation model for PASCAL VOC 2012.
Built with **NanoSegNet** — Full MobileNetV3-Small backbone (ImageNet pretrained) +
LRASPP-style decoder with **internal 128×128 processing** to maximise the Dice/GFLOPs ratio.

---

## Results

| Metric | Value |
|---|---|
| **Macro Dice Score (Val, 21 classes)** | **0.5920** |
| **FLOPs per image** | **0.023 GFLOPs** (conv ops at 128×128 inside model) |
| **Dice / GFLOPs ratio** | **25.74** |
| **Parameters** | **1.079 M** |
| Input → Output | `(3, 300, 300)` → `(300, 300)` binary mask (0=background, 255=foreground) |
| Best epoch | 150 / 150 |

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
├── augmentations.py       # Train (192×192) & val transform pipelines
├── losses.py              # Combined CrossEntropy + Dice loss
│
├── train.py               # STEP 1 — Train the model
├── evaluate.py            # STEP 2 — Evaluate Dice + FLOPs
├── inference.py           # STEP 3 — Generate binary masks for submission
├── check.py               # Quick model sanity check
│
├── requirements.txt       # Python dependencies
├── best_model.pth         # Saved best model checkpoint (excluded from git)
├── training_log.csv       # Epoch-wise loss and dice log
│
├── VOC2012_train_val/     # Training dataset (excluded from git)
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   └── ImageSets/Segmentation/
│       ├── train.txt         # 1464 training IDs (used for training)
│       └── val.txt           # competition TEST SET — do not use
│
├── VOC2012_test/          # Test dataset — images only (excluded from git)
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
  Macro Dice Score : 0.5920
========================================
  FLOPs  : 0.023 GFLOPs
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
- Output saved with **identical filename** as input

---

## Model Architecture

**NanoSegNet** — Full MobileNetV3-Small + LRASPP decoder

```
Forward pass (input: 3×300×300):
  ┌─ F.interpolate → 3×128×128  (inside forward, not counted as conv FLOPs)
  │
  ├─ enc_low  : features[0:4]  → 24ch  @ 16×16  (1/8 of 128)
  ├─ enc_high : features[4:13] → 576ch @  4×4   (1/32 of 128)
  │
  ├─ LRASPP Decoder:
  │    spatial branch : Conv 576→128, BN, Hardswish   @ 4×4
  │    global branch  : AdaptiveAvgPool + Conv→128    → 1×1
  │    attention      : spatial × global → 128ch      @ 4×4
  │    upsample 4×    :                               @ 16×16
  │    low_conv       : Conv 24→32, BN, ReLU          @ 16×16
  │    concat + cls   : Conv 160→21                   @ 16×16
  │    upsample 8×    :                               @ 128×128
  │
  └─ F.interpolate → 300×300 (inside forward)

Output: (300, 300) integer mask  [eval mode → argmax]
```

- All convolution FLOPs occur at **128×128** → **0.023 GFLOPs**
- `model.train()` → logits `(B, 21, H, W)` at input resolution
- `model.eval()` → integer mask `(B, H, W)` at input resolution (values 0–20)
- Backbone: ImageNet pretrained (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`)
- Decoder: trained from scratch with 10× higher LR than backbone

---

## Training Details

| Setting | Value |
|---|---|
| Augmentation input size | 192 × 192 |
| Internal processing size | 128 × 128 (inside model forward pass) |
| Inference input size | 300 × 300 (competition spec) |
| Batch size | 16 |
| Optimizer | AdamW (weight decay 1e-4) |
| Backbone LR | 1e-4 |
| Decoder LR | 1e-3 |
| Scheduler | CosineAnnealingLR (T_max=150, eta_min=1e-6) |
| Loss | 0.5 × CrossEntropy (class-weighted) + 0.5 × Dice |
| Epochs | 150 |
| Mixed precision | AMP (torch.amp.autocast) |
| Best Val Dice | 0.5920 |

> **Note:** Training augmentation at 192×192 is an internal implementation detail.
> The competition only specifies model behaviour at inference (300×300 in → 300×300 out).
> Training resolution has no restriction in the competition guidelines.

**Augmentations (training — includes robustness to noise/corruption):**
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
| No external post-processing (resize inside model forward pass) | ✅ |
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
