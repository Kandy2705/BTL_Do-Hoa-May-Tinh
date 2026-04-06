"""Bounding-box computation from projected 3D object bounds."""

from __future__ import annotations

import numpy as np

from btl2.renderer.camera import CameraMatrices
from btl2.scene.scene import Scene
from btl2.utils.math3d import ndc_to_screen, project_points, transform_points


def compute_bounding_boxes(scene: Scene, camera: CameraMatrices, config: dict) -> list[dict]:
    """Project each object's local AABB into screen space and clip the result."""
    results: list[dict] = []
    min_bbox_pixels = float(config["min_bbox_pixels"])

    for obj in scene.objects:
        if obj.instance_id <= 0 or obj.aabb_local is None:
            continue

        corners_world = transform_points(obj.model_matrix, obj.aabb_local.corners())
        ndc_xy, ndc_depth = project_points(corners_world, camera.view, camera.projection)

        in_front = np.any(ndc_depth <= 1.0)
        if not in_front:
            obj.visible = False
            continue

        screen = ndc_to_screen(ndc_xy, camera.width, camera.height)
        x_min = float(np.clip(np.min(screen[:, 0]), 0.0, camera.width - 1))
        y_min = float(np.clip(np.min(screen[:, 1]), 0.0, camera.height - 1))
        x_max = float(np.clip(np.max(screen[:, 0]), 0.0, camera.width - 1))
        y_max = float(np.clip(np.max(screen[:, 1]), 0.0, camera.height - 1))
        width = x_max - x_min
        height = y_max - y_min

        if width < min_bbox_pixels or height < min_bbox_pixels:
            obj.visible = False
            continue

        visibility = float((width * height) / (camera.width * camera.height))
        if visibility < float(config["visibility_threshold"]):
            obj.visible = False
            continue

        obj.visible = True
        results.append(
            {
                "instance_id": obj.instance_id,
                "class_name": obj.class_name,
                "class_id": obj.semantic_id,
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
                "bbox_xywh": [x_min, y_min, width, height],
                "visibility_ratio": visibility,
            }
        )

    return results
