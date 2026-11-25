"""Shared implementation behind the YOLO segmentation pipelines.

`pipelines/yolo11-seg` and `pipelines/yolov8-seg` are the two entry points. They
differ only in which Ultralytics checkpoint they default to, so the conversion,
training and scoring code lives here once rather than once per family. The repo
already does this for `backbones` and `metrics`.

A bug fixed here is fixed for both. That is not hypothetical: the CVAT mask
tracing that the wet season needed went into `cvat_to_yolo` after the YOLO11
pipeline was already written, and a copied directory would have silently kept
reading polygons only.
"""
