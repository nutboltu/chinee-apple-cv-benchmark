"""Shared backbones. Pixels in, patch tokens out.

Every pipeline that needs DINO features loads them through here, so the same
preprocessing and the same token grid are used everywhere and results stay
comparable. Nothing in this package knows about clustering, probes, CVAT or
chinee apple.
"""

from .dino import (  # noqa: F401
    pick_device,
    load_backbone,
    preprocess,
    patch_features,
    FAMILY_DEFAULTS,
)
