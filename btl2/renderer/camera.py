"""Camera matrices derived from the scene camera state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from btl2.scene.scene import CameraState
from btl2.utils.math3d import look_at, perspective


@dataclass
class CameraMatrices:
    """View and projection matrices plus a few convenience values."""

    view: np.ndarray
    projection: np.ndarray
    position: np.ndarray
    near: float
    far: float
    width: int
    height: int


def build_camera_matrices(camera: CameraState) -> CameraMatrices:
    """Turn camera pose and intrinsics into the matrices used by shaders."""
    aspect = float(camera.image_width) / float(camera.image_height)
    return CameraMatrices(
        view=look_at(camera.position, camera.target, camera.up),
        projection=perspective(camera.fov_y_degrees, aspect, camera.near, camera.far),
        position=camera.position.astype(np.float32),
        near=float(camera.near),
        far=float(camera.far),
        width=int(camera.image_width),
        height=int(camera.image_height),
    )
