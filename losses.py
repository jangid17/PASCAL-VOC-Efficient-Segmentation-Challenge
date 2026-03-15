import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft multiclass Dice loss.
    Macro-averaged over all classes (excluding ignore_index pixels).
    """

    def __init__(self, num_classes: int = 21, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = F.softmax(preds, dim=1)
        valid_mask = (targets != self.ignore_index).float()

        total_dice = 0.0
        for cls in range(self.num_classes):
            pred_cls   = preds[:, cls] * valid_mask
            target_cls = (targets == cls).float() * valid_mask

            intersection = (pred_cls * target_cls).sum()
            union        = pred_cls.sum() + target_cls.sum()
            total_dice  += (2.0 * intersection + 1e-6) / (union + 1e-6)

        return 1.0 - total_dice / self.num_classes


class CombinedLoss(nn.Module):
    """0.5 × CrossEntropy (with class weights) + 0.5 × Dice."""

    def __init__(self, num_classes: int = 21, ignore_index: int = 255,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                         weight=class_weights)
        self.dice = DiceLoss(num_classes, ignore_index)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.ce(preds, targets) + 0.5 * self.dice(preds, targets)
