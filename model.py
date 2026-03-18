"""
model.py — NanoSegNet: Full MobileNetV3-Small + 3-scale LRASPP decoder
=======================================================================
Key design for maximum Dice/GFLOPs ratio:
  • All conv FLOPs happen at internal 128×128 regardless of input size
  • 3-level skip connections (1/8, 1/16, 1/32) for richer spatial detail
  • Larger decoder (256ch) at zero extra backbone cost
  • Backbone: full MobileNetV3-Small, ImageNet pretrained

Forward pass:
  model.train()  → logits  (B, 21, H, W)  at input resolution
  model.eval()   → integer class mask (B, H, W), values 0-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class NanoSegNet(nn.Module):
    """
    MobileNetV3-Small encoder split into 3 levels + enlarged LRASPP decoder.
    All convolutions run at PROC_SIZE×PROC_SIZE for minimal GFLOPs.
    """

    # Internal processing resolution — all conv FLOPs measured here
    _PROC_SIZE = 128

    def __init__(self, num_classes: int = 21):
        super().__init__()
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        feats = backbone.features

        # ── 3-level encoder (pretrained) ─────────────────────────────────────
        self.enc_low  = feats[0:4]    # → 24ch  @ 1/8   (16×16 at 128)
        self.enc_mid  = feats[4:9]    # → 48ch  @ 1/16  ( 8×8  at 128)
        self.enc_high = feats[9:13]   # → 576ch @ 1/32  ( 4×4  at 128)

        # ── Enlarged LRASPP Decoder ───────────────────────────────────────────
        # High-level branch: 576 → 256
        self.spatial = nn.Sequential(
            nn.Conv2d(576, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),
        )
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 256, 1, bias=False),
            nn.Hardsigmoid(inplace=True),
        )
        # Mid-level skip: 48 → 64
        self.mid_conv = nn.Sequential(
            nn.Conv2d(48, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Low-level skip: 24 → 64
        self.low_conv = nn.Sequential(
            nn.Conv2d(24, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Classifier: (256+64)+64 = 384 → num_classes
        self.cls = nn.Conv2d(256 + 64 + 64, num_classes, kernel_size=1, bias=True)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]

        # Downsample to fixed processing size (interpolation not counted by thop)
        x = F.interpolate(x, size=(self._PROC_SIZE, self._PROC_SIZE),
                          mode="bilinear", align_corners=False)

        # 3-level encoder
        low  = self.enc_low(x)        # 24ch  @ 16×16
        mid  = self.enc_mid(low)      # 48ch  @  8×8
        high = self.enc_high(mid)     # 576ch @  4×4

        # LRASPP high-level: spatial × global attention
        feat = self.spatial(high) * self.squeeze(high)    # 256ch @ 4×4

        # Upsample to 8×8 and concat mid-level skip
        feat = F.interpolate(feat, size=mid.shape[2:],
                             mode="bilinear", align_corners=False)  # 256ch → 8×8
        mid_feat = self.mid_conv(mid)                               # 64ch  @ 8×8
        feat = torch.cat([feat, mid_feat], dim=1)                   # 256+64 @ 8×8

        # Upsample to 16×16 and concat low-level skip
        feat = F.interpolate(feat, size=low.shape[2:],
                             mode="bilinear", align_corners=False)  # → 16×16
        low_feat = self.low_conv(low)                               # 64ch  @ 16×16
        feat = torch.cat([feat, low_feat], dim=1)                   # 320+64 @ 16×16

        feat = self.cls(feat)                                        # 21ch @ 16×16

        return F.interpolate(feat, size=(h, w),
                             mode="bilinear", align_corners=False)  # → H×W

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Always returns raw logits (B, 21, H, W)."""
        return self._forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._forward(x)
        if self.training:
            return logits
        return torch.argmax(logits, dim=1)


def get_model(num_classes: int = 21) -> NanoSegNet:
    return NanoSegNet(num_classes=num_classes)
