"""DINOv3 / DINOv2 ViT loading and patch-token extraction.

Backbone loading is family-agnostic:
    * DINOv3 (gated): clone https://github.com/facebookresearch/dinov3, accept
      Meta's license, download ViT-B/16 weights, pass repo= and weights=.
    * DINOv2 (ungated fallback): loads straight from torch.hub, so the whole
      pipeline can be built and tested before the DINOv3 license clears. Same
      forward_features dict, so nothing downstream changes.

    model, meta = load_backbone(family="dinov2")
    feats, grid = patch_features(model, "tile.jpg", meta)   # (N, D), (gh, gw)
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-family patch size and a default hub entrypoint. IMG_SIZE is snapped to a
# multiple of the patch size at preprocessing time.
FAMILY_DEFAULTS = {
    "dinov3": {"patch": 16, "arch": "dinov3_vitb16", "hub": None},
    "dinov2": {"patch": 14, "arch": "dinov2_vitb14", "hub": "facebookresearch/dinov2"},
}


# --------------------------------------------------------------------------- #
# Device + backbone
# --------------------------------------------------------------------------- #
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_backbone(family: str = "dinov3", repo: str | None = None,
                  weights: str | None = None, arch: str | None = None,
                  device: str | None = None):
    """Load a DINO ViT backbone and return (model, meta).

    meta carries the patch size and device so the rest of the pipeline stays
    family-agnostic.
    """
    if family not in FAMILY_DEFAULTS:
        raise ValueError(f"family must be one of {list(FAMILY_DEFAULTS)}")
    cfg = FAMILY_DEFAULTS[family]
    arch = arch or cfg["arch"]
    device = device or pick_device()

    if family == "dinov3":
        if not repo or not weights:
            raise ValueError(
                "DINOv3 is gated: pass --repo (local clone of facebookresearch/"
                "dinov3) and --weights (the .pth you downloaded after accepting "
                "Meta's license). To build without them today, use --family dinov2."
            )
        model = torch.hub.load(repo, arch, source="local", weights=weights)
    else:  # dinov2, ungated
        model = torch.hub.load(cfg["hub"], arch)

    model = model.to(device).eval()
    meta = {"family": family, "patch": cfg["patch"], "device": device, "arch": arch}
    return model, meta


# --------------------------------------------------------------------------- #
# Preprocessing + feature extraction
# --------------------------------------------------------------------------- #
def _snap(size: int, patch: int) -> int:
    """Round a target side length down to a whole number of patches (>= 1 patch)."""
    return max(patch, (size // patch) * patch)


def preprocess(image: Image.Image, patch: int, img_size: int = 768):
    """Resize to a multiple of `patch`, normalize. Returns (tensor, (gh, gw))."""
    side = _snap(img_size, patch)
    tx = transforms.Compose([
        transforms.Resize((side, side)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tx(image.convert("RGB")).unsqueeze(0)
    grid = side // patch
    return x, (grid, grid)


@torch.inference_mode()
def patch_features(model, image, meta, img_size: int = 768, l2: bool = True):
    """Extract per-patch feature vectors from one image.

    image: PIL.Image or path. Returns (feats [N, D] float32 numpy, (gh, gw)).
    """
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    x, grid = preprocess(image, meta["patch"], img_size)
    x = x.to(meta["device"])
    out = model.forward_features(x)
    patches = out["x_norm_patchtokens"][0]            # (N, D)
    if l2:
        patches = torch.nn.functional.normalize(patches, p=2, dim=-1)
    return patches.float().cpu().numpy(), grid
