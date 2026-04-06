"""Lightweight occlusion estimation using instance-colored masks."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from btl2.scene.scene_object import SceneObject
from btl2.utils.colors import instance_color


def estimate_occlusion_ratios(mask_rgb: np.ndarray, objects: list[SceneObject]) -> dict[int, float]:
    """Approximate occlusion as 1 - visible_mask_pixels / bbox_proxy_pixels."""
    counts = defaultdict(int)
    flat = mask_rgb.reshape(-1, 3)
    for color in flat:
        key = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
        counts[key] += 1

    occlusion: dict[int, float] = {}
    for obj in objects:
        if obj.instance_id <= 0:
            continue
        key = (instance_color(obj.instance_id)[0] << 16) | (instance_color(obj.instance_id)[1] << 8) | instance_color(obj.instance_id)[2]
        visible_pixels = counts.get(key, 0)
        expected_area = max(1.0, float(obj.scale[0] * obj.scale[1] * 400.0))
        occlusion[obj.instance_id] = float(max(0.0, min(1.0, 1.0 - visible_pixels / expected_area)))
    return occlusion
