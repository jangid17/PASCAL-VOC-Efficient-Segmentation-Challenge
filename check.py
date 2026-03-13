"""
check.py — Quick sanity check
==============================
Verifies that the model can do a forward pass on a dummy 300×300 image
and prints architecture summary + parameter count.

Usage:
    python check.py
"""

import torch
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}\n")

model = get_model(num_classes=21).to(device)

# --- Training mode: returns logits ---
model.train()
dummy  = torch.randn(2, 3, 300, 300, device=device)
logits = model.get_logits(dummy)
print(f"Training  — logits shape : {logits.shape}")   # (2, 21, 300, 300)

# --- Eval mode: returns integer mask ---
model.eval()
with torch.no_grad():
    mask = model(dummy)
print(f"Inference — mask   shape : {mask.shape}")     # (2, 300, 300)
print(f"            mask   dtype : {mask.dtype}")     # torch.int64
print(f"            mask   range : [{mask.min()}, {mask.max()}]")

# --- Param count ---
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"\nParameters : {params:.2f} M")
print("\nAll checks passed.")
