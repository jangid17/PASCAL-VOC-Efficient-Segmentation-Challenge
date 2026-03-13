import os
import cv2
import torch
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    """
    PASCAL VOC 2012 segmentation dataset.

    root_dir  : VOC dataset root (contains JPEGImages/ and SegmentationClass/)
    image_ids : list of image stem IDs (no extension)
    transform : albumentations Compose pipeline
    """

    def __init__(self, root_dir: str, image_ids: list, transform=None):
        self.root_dir = root_dir
        self.image_ids = image_ids
        self.transform = transform
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        self.masks_dir = os.path.join(root_dir, "SegmentationClass")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        img_path = os.path.join(self.images_dir, image_id + ".jpg")
        mask_path = os.path.join(self.masks_dir, image_id + ".png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 255 is the VOC "void/boundary" label; clamp anything else above 20
        mask[mask > 20] = 255

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()
