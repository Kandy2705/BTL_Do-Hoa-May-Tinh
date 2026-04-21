"""Cấu trúc dữ liệu cho từng object renderable trong scene BTL 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from btl2.utils.math3d import AABB, compose_model_matrix


@dataclass
class SceneObject:
    """Một object có transform, class/instance label và thông tin vật liệu.

    `instance_id` dùng để tách từng object trong segmentation mask. Riêng road
    thường có `instance_id = 0` vì là nền, không xuất bbox như object động.
    """

    name: str
    class_name: str
    mesh_key: str
    position: np.ndarray
    rotation_degrees: np.ndarray
    scale: np.ndarray
    base_color: np.ndarray
    instance_id: int
    semantic_id: int
    metadata: dict[str, Any] = field(default_factory=dict)
    aabb_local: AABB | None = None
    visible: bool = True

    @property
    def model_matrix(self) -> np.ndarray:
        """Tính ma trận model từ position/rotation/scale hiện tại."""
        return compose_model_matrix(self.position, self.rotation_degrees, self.scale)

    def to_metadata(self) -> dict[str, Any]:
        """Chuyển object sang dict thuần để ghi JSON metadata."""
        return {
            "name": self.name,
            "class_name": self.class_name,
            "mesh_key": self.mesh_key,
            "instance_id": self.instance_id,
            "semantic_id": self.semantic_id,
            "position": self.position.tolist(),
            "rotation_degrees": self.rotation_degrees.tolist(),
            "scale": self.scale.tolist(),
            "visible": bool(self.visible),
            "extra": self.metadata,
        }
