"""Backbone-agnostic metrics shared by every pipeline.

Nothing in here knows about DINOv3, CVAT or chinee apple. The functions take
probability grids and boolean masks, so any pipeline that can produce a dense
score map can be scored with the same code and the numbers stay comparable
across experiments.
"""

from .segmentation import (  # noqa: F401
    patch_metrics,
    best_f1_threshold,
    mask_metrics,
    instance_map,
    report,
)
