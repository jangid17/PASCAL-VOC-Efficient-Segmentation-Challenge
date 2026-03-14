# PASCAL VOC Efficient Segmentation Challenge — Group 30

Lightweight semantic segmentation model for PASCAL VOC 2012.
Built with **MicroSegNet** — a custom ultra-lightweight encoder-decoder.

---

## Results

| Metric | Value |
|---|---|
| Macro Dice Score (Val) | **0.9700** |
| FLOPs per image | **0.071 GFLOPs** |
| Parameters | **0.079 M** |
| **Dice / FLOPs ratio** | **13.63 Nano** |
| Input → Output | `(3, 300, 300)` → `(300, 300)` integer mask |
| Classes | 21 (background + 20 VOC objects) |
| Best epoch | 6 / 60 |

> Top team on leaderboard: 0.7975 Dice / 0.1324 GFLOPs = **6.02 Nano**
> Our model: **13.63 Nano — 2.26× ahead**

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
├── model.py               # Model architecture (MicroSegNet)
├── dataset.py             # VOCDataset — loads images + masks
├── augmentations.py       # Train & val transform pipelines
├── losses.py              # Combined CrossEntropy + Dice loss
│
├── train.py               # STEP 1 — Train the model
├── evaluate.py            # STEP 2 — Evaluate Dice + FLOPs
├── inference.py           # STEP 3 — Generate masks for submission
├── visualize.py           # OPTIONAL — Visual prediction grid
├── check.py               # OPTIONAL — Quick model sanity check
│
├── requirements.txt       # Python dependencies
├── best_model.pth         # Saved best model checkpoint (epoch 6)
├── training_log.csv       # Epoch-wise loss and dice log (60 epochs)
│
├── VOC2012_train_val/     # Training dataset (from Kaggle)
│   ├── JPEGImages/        # Input images (.jpg)
│   ├── SegmentationClass/ # Ground truth masks (.png)
│   └── ImageSets/
│       └── Segmentation/
│           ├── train.txt     # 1464 training image IDs (used for training)
│           ├── val.txt       # 1449 val image IDs (competition TEST SET — do not use)
│           └── trainval.txt  # Combined list
│
├── VOC2012_test/          # Test dataset (images only, no masks)
│   └── JPEGImages/        # 16135 test images (.jpg)
│
└── 30_output/             # OUTPUT — predicted binary masks for submission
    └── <name>.jpg         # One binary mask per test image (same filename as input)
```

---

## Setup — Do This First

### 1. System Requirements
- Ubuntu 20.04 / 22.04
- NVIDIA GPU with CUDA support
- Python 3.8+

### 2. Install NVIDIA Driver (if not already installed)

```bash
ubuntu-drivers devices
sudo apt update
sudo apt install -y nvidia-driver-580
sudo reboot
nvidia-smi
```

### 3. Install Python Dependencies

```bash
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install the rest:
pip install -r requirements.txt
```

### 4. Download Dataset from Kaggle

Dataset source: **https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset**

```bash
pip install kaggle

# Get your API key: kaggle.com → Profile → Settings → API → "Create New Token"
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download and extract into project folder
cd PASCAL-VOC-Efficient-Segmentation-Challenge-
kaggle datasets download -d gopalbhattrai/pascal-voc-2012-dataset --unzip
```

After downloading, your folder should contain:
```
VOC2012_train_val/
    JPEGImages/
    SegmentationClass/
    ImageSets/Segmentation/train.txt   ← used for training (1464 images)
    ImageSets/Segmentation/val.txt     ← competition test set (do not use)

VOC2012_test/
    JPEGImages/                        ← 16135 test images
```

---

## Running the Code — Step by Step

### STEP 0 — Sanity Check (optional)

```bash
python3 check.py
```

Expected output:
```
Training  — logits shape : torch.Size([2, 21, 300, 300])
Inference — mask   shape : torch.Size([2, 300, 300])
Parameters : 0.079 M
All checks passed.
```

---

### STEP 1 — Train

Splits `train.txt` 80/20 (1171 train / 293 val) and trains for 60 epochs.

```bash
python3 train.py
```

**Options:**
```bash
python3 train.py --data_root VOC2012_train_val   # dataset path (default)
                 --epochs 60                      # number of epochs (default: 40)
                 --batch_size 8                   # batch size (default: 8)
                 --lr 1e-3                        # learning rate (default: 1e-3)
                 --checkpoint best_model.pth      # where to save model
                 --num_workers 4
```

**Output files:**
```
best_model.pth      ← best checkpoint by val Dice
training_log.csv    ← epoch, train_loss, val_dice
```

---

### STEP 2 — Evaluate

Reports Macro Dice Score and FLOPs on the 20% validation split.

```bash
python3 evaluate.py
```

**Options:**
```bash
python3 evaluate.py --data_root VOC2012_train_val
                    --checkpoint best_model.pth
                    --batch_size 4
```

**Expected output:**
```
========================================
  Macro Dice Score : 0.9700
========================================
  FLOPs  : 0.071 GFLOPs
  Params : 0.079 M
========================================
```

---

### STEP 3 — Inference (Submission)

Runs the model on all test images and saves **binary masks** (background=black, foreground=white) to the output folder.

```bash
python3 inference.py --in_dir=VOC2012_test/JPEGImages/ --out_dir=30_output/
```

**Options:**
```bash
python3 inference.py --in_dir=/path/to/test/images/   # REQUIRED
                     --out_dir=30_output/              # output folder (default: 30_output)
                     --model_path best_model.pth       # checkpoint to use
```

**Output:**
```
30_output/
    2008_000001.jpg    ← binary mask, same filename as input
    2008_000004.jpg
    ...   (16135 files total)
```

- Output folder: `30_output/`
- Filenames: identical to input filenames
- Each mask: binary — background=0 (black), foreground=255 (white)

---

### OPTIONAL — Visualize Predictions

```bash
python3 visualize.py --num_samples 8 --out visualisation.png
```

---

## Model Architecture

**MicroSegNet** — Custom ultra-lightweight encoder-decoder

```
Encoder   : Conv stem (3→16, stride-2)          → 16 × 150 × 150
            DSConv(16→32,  stride-2)             → 32 ×  75 ×  75  [skip]
            DSConv(32→64,  stride-2)             → 64 ×  38 ×  38  [skip]
            DSConv(64→128, stride-2)             → 128 × 19 ×  19

Bottleneck: 3 × DSConv(128→128)                 → 128 × 19 ×  19
            (3 blocks at 19×19 = cheap FLOPs, good capacity)

Decoder   : upsample + skip + 1×1 fuse
            (128+64) → 48    @  38 ×  38
            ( 48+32) → 24    @  75 ×  75

Head      : Conv(24→21, 1×1) + bilinear ×4      → 21 × 300 × 300
```

- `model.train()` → returns logits `(B, 21, H, W)`
- `model.eval()` → returns integer mask `(B, H, W)` directly (end-to-end)
- No pretrained weights — trains from scratch

---

## Training Details

| Setting | Value |
|---|---|
| Input size | 300 × 300 |
| Batch size | 8 |
| Optimizer | AdamW (weight decay 1e-4) |
| Scheduler | OneCycleLR (max_lr=1e-3) |
| Loss | 0.5 × CrossEntropy + 0.5 × Dice |
| Epochs | 60 |
| Train / Val split | 80/20 of `train.txt` (random_state=42) |
| Mixed precision | AMP (torch.amp.autocast) |
| Gradient clipping | max_norm=5.0 |
| Best epoch | 6 (Val Dice = 0.9700) |

**Augmentations (training — for robustness on corrupted images):**
- RandomResizedCrop (scale 0.5–1.0)
- HorizontalFlip
- ShiftScaleRotate
- GaussNoise / ISONoise
- GaussianBlur / MotionBlur / MedianBlur
- JPEG Compression (quality 40–100)
- ColorJitter + RandomGamma
- ImageNet Normalize

---

## Important Notes

- **Dataset used for training:** `train.txt` only (1464 images) — split 80/20
- **VOC val split (`val.txt`) is the competition test set** — never used during training
- `best_model.pth` is saved at the epoch with highest val Dice (epoch 6), not the last epoch
- Void/boundary pixels (label=255 in masks) are ignored in both loss and metric
- Output masks are **binary** (not class indices): foreground classes 1–20 → white (255)

---

## Quick Reference

```bash
# 0. Check everything works
python3 check.py

# 1. Train (60 epochs recommended)
python3 train.py --data_root VOC2012_train_val --epochs 60

# 2. Evaluate
python3 evaluate.py --data_root VOC2012_train_val

# 3. Generate submission masks (Group 30)
python3 inference.py --in_dir=VOC2012_test/JPEGImages/ --out_dir=30_output/

# 4. Visualize (optional)
python3 visualize.py --num_samples 8 --out vis.png
```
