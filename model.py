import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class EfficientSegNet(nn.Module):
    """
    Lightweight segmentation model: DeepLabV3+ with MobileNetV3-Large backbone.

    - Training mode  : forward() returns logits (B, C, H, W).
    - Eval/inference : forward() returns integer class mask (B, H, W) directly,
                       satisfying the competition requirement of end-to-end output.

    Use get_logits() inside the training loop to always get raw logits.
    """

    def __init__(self, num_classes: int = 21):
        super().__init__()
        base = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
        # Replace final classifier head for 21 VOC classes
        base.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1
        )
        # Drop aux head to save FLOPs at inference
        base.aux_classifier = None
        self.base = base
        self.num_classes = num_classes

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, C, H, W). Always call this inside training loop."""
        return self.base(x)["out"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)["out"]
        if self.training:
            return logits
        # End-to-end: integer class indices produced by the network forward pass
        return torch.argmax(logits, dim=1)


def get_model(num_classes: int = 21) -> EfficientSegNet:
    return EfficientSegNet(num_classes=num_classes)
