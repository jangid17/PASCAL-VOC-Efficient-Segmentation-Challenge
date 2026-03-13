import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization (matches MobileNetV3 pretrained weights)
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def get_train_transforms():
    """
    Heavy augmentation pipeline for training.
    Includes geometric, photometric, noise, blur and compression augmentations
    to make the model robust to real-world corruptions (required by competition).
    """
    return A.Compose([
        # --- Geometry ---
        A.RandomResizedCrop(size=(300, 300), scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, border_mode=0, p=0.4),

        # --- Robustness: noise & blur ---
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.35),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.25),

        # --- Robustness: JPEG / color ---
        A.ImageCompression(quality_range=(40, 100), p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.1, p=0.4),
        A.RandomGamma(gamma_limit=(70, 130), p=0.2),

        # --- Normalize & convert ---
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Minimal pipeline for validation / test."""
    return A.Compose([
        A.Resize(300, 300),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
