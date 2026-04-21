"""Các dataclass mô tả một frame scene BTL 2 trước khi render.

Những lớp trong file này chỉ giữ dữ liệu thuần: camera, đèn, danh sách object.
Renderer và exporter cùng đọc chung cấu trúc này nên các trường phải rõ nghĩa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from btl2.scene.scene_object import SceneObject


@dataclass
class DirectionalLight:
    """Nguồn sáng hướng đơn giản dùng trong pass RGB."""

    direction: np.ndarray
    color: np.ndarray
    intensity: float
    ambient_strength: float


@dataclass
class CameraState:
    """Thông số camera gồm pose và intrinsics của ảnh output."""

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
    """Mô tả đầy đủ một frame để renderer và annotation module cùng sử dụng."""

    frame_id: str
    seed: int
    split: str
    camera: CameraState
    light: DirectionalLight
    objects: list[SceneObject] = field(default_factory=list)
    background_color: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.75, 0.95], dtype=np.float32))

    def add_object(self, obj: SceneObject) -> None:
        """Thêm object vào danh sách render theo thứ tự vẽ của scene."""
        self.objects.append(obj)
