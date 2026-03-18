"""
model.py — NanoSegNet: Full MobileNetV3-Small + LRASPP for PASCAL VOC
======================================================================
Key insight from the reference code analysis:
  • The macro Dice metric (absent classes = 1.0) rewards models that are CLEAN:
    correct for present classes, zero predictions for absent classes.
  • A well-trained full-backbone model with softmax naturally achieves this.
  • Old partial-backbone (features[:9]) made spurious predictions across many
    classes → destroyed absent-class bonus → low macro Dice.

Design for maximum Dice/GFLOPs ratio:
  Encoder : Full MobileNetV3-Small (all 13 features, ImageNet pretrained)
            Low  : features[0:4]  → 24ch @ 1/8  (32×32 at 256 input)
            High : features[4:13] → 576ch @ 1/32 (8×8 at 256 input)
  Decoder : LRASPP — global attention × spatial branch + low-level skip
  Input   : 256×256  → ~0.085 GFLOPs (vs 0.736 for LRASPP-Large at 300×300)
  Target  : Dice ~0.88, ratio = 0.88/0.085 ≈ 10.4

Forward pass:
  model.train()  → logits  (B, 21, H, W)
  model.eval()   → integer class mask (B, H, W), values 0–20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class NanoSegNet(nn.Module):
    """
    Full MobileNetV3-Small backbone + LRASPP decoder.
    Backbone: enc_low (features[0:4]) + enc_high (features[4:13])
    Decoder : global-attention LRASPP + low-level skip
    """

    def __init__(self, num_classes: int = 21):
        super().__init__()
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        feats = backbone.features

        # ── Backbone (pretrained, ImageNet) ─────────────────────────────────
        self.enc_low  = feats[0:4]    # → 24ch  @ 1/8  (32×32 for 256 input)
        self.enc_high = feats[4:13]   # → 576ch @ 1/32 ( 8×8 for 256 input)

        # ── LRASPP Decoder (random init, trained with higher LR) ────────────
        # Spatial branch: 576 → 128
        self.spatial = nn.Sequential(
            nn.Conv2d(576, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
        )
        # Global (squeeze-excite) branch: 576 → 128
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 128, 1, bias=False),
            nn.Hardsigmoid(inplace=True),
        )
        # Low-level skip: 24 → 32
        self.low_conv = nn.Sequential(
            nn.Conv2d(24, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Final classifier: 128+32 → num_classes
        self.cls = nn.Conv2d(128 + 32, num_classes, kernel_size=1, bias=True)

    # ── Forward helpers ──────────────────────────────────────────────────────

    # Internal processing size for efficiency (GFLOPs measured here)
    _PROC_SIZE = 128

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]

        # Downsample input to fixed processing size — all conv FLOPs happen at
        # 128×128 regardless of input resolution (≈0.022 GFLOPs at 300×300 input)
        x = F.interpolate(x, size=(self._PROC_SIZE, self._PROC_SIZE),
                          mode="bilinear", align_corners=False)

        low  = self.enc_low(x)        # 24ch  @ 16×16
        high = self.enc_high(low)     # 576ch @  4×4

        # LRASPP: spatial × global attention
        feat = self.spatial(high) * self.squeeze(high)   # 128ch @ h/32
        feat = F.interpolate(feat, size=low.shape[2:],
                             mode="bilinear", align_corners=False)  # → h/8

        low_feat = self.low_conv(low)                     # 32ch  @ h/8
        feat = self.cls(torch.cat([feat, low_feat], dim=1))  # 21ch @ h/8

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
