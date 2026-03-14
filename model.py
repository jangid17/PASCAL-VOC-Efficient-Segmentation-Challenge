"""
model.py — MicroSegNet: Ultra-Lightweight Segmentation for PASCAL VOC
======================================================================
Custom tiny encoder-decoder targeting ~0.10 GFLOPs for a (3, 300, 300) input.

Design principle
----------------
• Small channels at large spatial dims (150×150, 75×75) → cheap FLOPs
• Deeper bottleneck at tiny spatial dims (19×19)          → cheap FLOPs, more capacity
• Skip connections from 38×38 and 75×75 for spatial detail

Architecture
------------
Encoder   : stem Conv(3→16, s2)              → 16 ×  150 × 150
            DSConv(16→32,  s2)               → 32 ×   75 ×  75   [skip s1]
            DSConv(32→64,  s2)               → 64 ×   38 ×  38   [skip s2]
            DSConv(64→128, s2)               → 128 ×  19 ×  19

Bottleneck: 3 × DSConv(128→128)              → 128 ×  19 ×  19
            (all compute at 19×19 = cheap!)

Decoder   : upsample + skip + 1×1 fuse
            (128+64) → 48    @  38 ×  38    [dec2]
            ( 48+32) → 24    @  75 ×  75    [dec1]

Head      : Conv(24→num_classes, 1×1)        @  75 ×  75
            bilinear ×4                      → 300 × 300

FLOPs     : ~0.10 GFLOPs  (well under 0.15 target)
Params    : ~0.12 M

Training  : forward() / get_logits()  → logits  (B, C, H, W)
Inference : forward() in eval mode    → integer class mask (B, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _DSConv(nn.Module):
    """Depthwise-Separable Conv: DW(3×3, stride) + PW(1×1), BN + ReLU6 each."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw    = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                               groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw    = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn_dw(self.dw(x)))
        return self.act(self.bn_pw(self.pw(x)))


class _Fuse(nn.Module):
    """1×1 Conv + BN + ReLU6 — merges concatenated encoder+decoder features."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MicroSegNet
# ---------------------------------------------------------------------------

class MicroSegNet(nn.Module):
    """
    Ultra-lightweight end-to-end segmentation model (~0.10 GFLOPs, ~0.12 M params).

    Key idea: pack model capacity into the 19×19 bottleneck where spatial ops
    are cheapest, and keep the large-resolution layers minimal.
    """

    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )                                              # 16 × 150 × 150

        self.enc1   = _DSConv(16,  32, stride=2)      # 32 ×  75 ×  75  [skip s1]
        self.enc2   = _DSConv(32,  64, stride=2)      # 64 ×  38 ×  38  [skip s2]
        self.enc3   = _DSConv(64, 128, stride=2)      # 128 × 19 ×  19

        # ── Bottleneck (3 blocks at 19×19 — cheap FLOPs, good capacity) ─────
        self.bottle = nn.Sequential(
            _DSConv(128, 128),
            _DSConv(128, 128),
            _DSConv(128, 128),
        )                                              # 128 × 19 × 19

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec2   = _Fuse(128 + 64, 48)             # 48 ×  38 ×  38
        self.dec1   = _Fuse( 48 + 32, 24)             # 24 ×  75 ×  75

        # ── Head ─────────────────────────────────────────────────────────────
        self.head   = nn.Conv2d(24, num_classes, 1)   # C  ×  75 ×  75  → ×4 upsample

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Core forward (always returns logits) ─────────────────────────────────
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]

        # Encoder
        s0 = self.stem(x)        # 16 × 150 × 150
        s1 = self.enc1(s0)       # 32 ×  75 ×  75
        s2 = self.enc2(s1)       # 64 ×  38 ×  38
        b  = self.bottle(self.enc3(s2))   # 128 × 19 × 19

        # Decoder — upsample, concat skip, fuse
        d = F.interpolate(b, size=s2.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec2(torch.cat([d, s2], dim=1))    # 48 × 38 × 38

        d = F.interpolate(d, size=s1.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec1(torch.cat([d, s1], dim=1))    # 24 × 75 × 75

        # Head + bilinear ×4 to original resolution
        logits = self.head(d)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, C, H, W). Always call this inside training loop."""
        return self._forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._forward(x)
        if self.training:
            return logits
        # End-to-end: integer class indices produced by the network forward pass
        return torch.argmax(logits, dim=1)


def get_model(num_classes: int = 21) -> MicroSegNet:
    return MicroSegNet(num_classes=num_classes)
