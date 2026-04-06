"""Container object for one procedurally generated frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from btl2.scene.scene_object import SceneObject


@dataclass
class DirectionalLight:
    """Simple directional light used by the RGB shading pass."""

    direction: np.ndarray
    color: np.ndarray
    intensity: float
    ambient_strength: float


@dataclass
class CameraState:
    """Camera intrinsics and pose stored alongside the scene."""

    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_degrees: float
    near: float
    far: float
    image_width: int
    image_height: int


@dataclass
class Scene:
    """Full scene description consumed by renderer and annotation modules."""

    frame_id: str
    seed: int
    split: str
    camera: CameraState
    light: DirectionalLight
    objects: list[SceneObject] = field(default_factory=list)
    background_color: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.75, 0.95], dtype=np.float32))

    def add_object(self, obj: SceneObject) -> None:
        """Append one object to the render list."""
        self.objects.append(obj)
