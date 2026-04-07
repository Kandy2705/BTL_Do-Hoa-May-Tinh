"""Shared constants used across the repository."""

from __future__ import annotations

CLASS_NAMES = [
    "person",
    "car",
    "bus",
    "truck",
    "motorbike",
    "traffic_light",
    "traffic_sign",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DEFAULT_SPLITS = ("train", "val")

DEFAULT_OUTPUT_SUBDIRS = (
    "images/train",
    "images/val",
    "depth/train",
    "depth/val",
    "masks/train",
    "masks/val",
    "labels_yolo/train",
    "labels_yolo/val",
    "annotations_coco",
    "metadata/train",
    "metadata/val",
    "previews",
)

IMAGE_EXT = ".png"
DEPTH_RAW_EXT = ".npy"
YOLO_EXT = ".txt"
JSON_EXT = ".json"
