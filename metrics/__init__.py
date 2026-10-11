"""Backbone-agnostic metrics shared by every pipeline.

Nothing in here knows about DINOv3, CVAT or chinee apple. The functions take
probability grids and boolean masks, so any pipeline that can produce a dense
score map can be scored with the same code and the numbers stay comparable
across experiments.

The scoring functions need numpy, scipy and scikit-learn. `render`, which turns
a finished metrics.json into a wiki page, is pure standard library, so the
heavy imports are deferred to keep it usable anywhere.
"""

_LAZY = {
    "patch_metrics": "segmentation",
    "best_f1_threshold": "segmentation",
    "mask_metrics": "segmentation",
    "instance_map": "segmentation",
    "report": "segmentation",
}

__all__ = list(_LAZY) + ["segmentation", "render"]


def __getattr__(name):
    """PEP 562 lazy import, so `metrics.render` works without numpy installed."""
    import importlib
    if name in _LAZY:
        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    if name in ("segmentation", "render"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
