"""Segmentation mapping and mask helpers."""

from __future__ import annotations

from btl2.scene.scene_object import SceneObject
from btl2.utils.colors import instance_color


def build_segmentation_mapping(objects: list[SceneObject], mask_rgb) -> dict:
    """Build a JSON-friendly mapping from mask color to class and instance id."""
    mapping = {}
    for obj in objects:
        if obj.instance_id <= 0:
            continue
        color = instance_color(obj.instance_id)
        mapping[f"{color[0]}_{color[1]}_{color[2]}"] = {
            "instance_id": obj.instance_id,
            "class_name": obj.class_name,
            "class_id": obj.semantic_id,
        }
    return mapping
