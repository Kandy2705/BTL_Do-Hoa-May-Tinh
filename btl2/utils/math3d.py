"""Basic 3D math helpers built on NumPy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


Vec3 = np.ndarray
Mat4 = np.ndarray


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of the input vector."""
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> Mat4:
    """Create a standard OpenGL perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fov_y_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    """Create a view matrix from camera position, target, and up vector."""
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(true_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def translation_matrix(position: Vec3) -> Mat4:
    """Build a homogeneous translation matrix."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(position, dtype=np.float32)
    return matrix


def scale_matrix(scale: Vec3) -> Mat4:
    """Build a homogeneous scale matrix."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = scale
    return matrix


def rotation_matrix_xyz(rotation_degrees: Vec3) -> Mat4:
    """Create a rotation matrix from Euler angles in XYZ order."""
    rx, ry, rz = np.radians(rotation_degrees)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rot_x = np.array(
        [[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    rot_y = np.array(
        [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    rot_z = np.array(
        [[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    return rot_z @ rot_y @ rot_x


def compose_model_matrix(position: Vec3, rotation_degrees: Vec3, scale: Vec3) -> Mat4:
    """Combine translation, rotation, and scale into one model matrix."""
    return translation_matrix(position) @ rotation_matrix_xyz(rotation_degrees) @ scale_matrix(scale)


def transform_points(matrix: Mat4, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 matrix to N 3D points and return transformed 3D points."""
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.hstack((points.astype(np.float32), ones))
    transformed = (matrix @ homogeneous.T).T
    w = np.clip(transformed[:, 3:4], 1e-8, None)
    return transformed[:, :3] / w


def project_points(points_world: np.ndarray, view: Mat4, projection: Mat4) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world points to NDC and clip-space depth."""
    ones = np.ones((points_world.shape[0], 1), dtype=np.float32)
    world_h = np.hstack((points_world.astype(np.float32), ones))
    clip = (projection @ view @ world_h.T).T
    w = np.clip(clip[:, 3:4], 1e-8, None)
    ndc = clip[:, :3] / w
    return ndc[:, :2], ndc[:, 2]


def ndc_to_screen(ndc_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized device coordinates to pixel coordinates."""
    x = (ndc_xy[:, 0] * 0.5 + 0.5) * width
    y = (1.0 - (ndc_xy[:, 1] * 0.5 + 0.5)) * height
    return np.column_stack((x, y))


@dataclass
class AABB:
    """Axis-aligned bounding box stored as local-space min/max corners."""

    min_corner: np.ndarray
    max_corner: np.ndarray

    def corners(self) -> np.ndarray:
        """Return the eight corners used for projection-based 2D bboxes."""
        x0, y0, z0 = self.min_corner
        x1, y1, z1 = self.max_corner
        return np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x0, y1, z0],
                [x1, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x0, y1, z1],
                [x1, y1, z1],
            ],
            dtype=np.float32,
        )
