"""Data structures that describe renderable objects in one scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from btl2.utils.math3d import AABB, compose_model_matrix


@dataclass
class SceneObject:
    """One renderable object with transform, labels, and shading info."""

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
        """Compute the current model matrix from object transform fields."""
        return compose_model_matrix(self.position, self.rotation_degrees, self.scale)

    def to_metadata(self) -> dict[str, Any]:
        """Export the transform and identity fields to JSON-ready values."""
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
