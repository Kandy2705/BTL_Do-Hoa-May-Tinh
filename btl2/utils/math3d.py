"""Các phép toán 3D cơ bản dùng chung trong BTL 2, viết bằng NumPy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


Vec3 = np.ndarray
Mat4 = np.ndarray


def normalize(vector: np.ndarray) -> np.ndarray:
    """Trả về vector đơn vị; nếu vector quá nhỏ thì giữ nguyên để tránh chia 0."""
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> Mat4:
    """Tạo ma trận phối cảnh chuẩn OpenGL từ FOV, aspect, near và far."""
    f = 1.0 / math.tan(math.radians(fov_y_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    """Tạo view matrix từ vị trí camera, điểm nhìn và vector up."""
    forward = normalize(target - eye)
    # right/true_up tạo hệ trục camera trực chuẩn để ảnh không bị nghiêng sai.
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
    """Tạo ma trận tịnh tiến dạng homogeneous 4x4."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(position, dtype=np.float32)
    return matrix


def scale_matrix(scale: Vec3) -> Mat4:
    """Tạo ma trận scale dạng homogeneous 4x4."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = scale
    return matrix


def rotation_matrix_xyz(rotation_degrees: Vec3) -> Mat4:
    """Tạo ma trận xoay từ góc Euler, thứ tự áp dụng XYZ."""
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
    """Ghép translation, rotation và scale thành ma trận model của object."""
    return translation_matrix(position) @ rotation_matrix_xyz(rotation_degrees) @ scale_matrix(scale)


def transform_points(matrix: Mat4, points: np.ndarray) -> np.ndarray:
    """Nhân ma trận 4x4 với N điểm 3D và trả về tọa độ 3D sau biến đổi."""
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.hstack((points.astype(np.float32), ones))
    transformed = (matrix @ homogeneous.T).T
    w = np.clip(transformed[:, 3:4], 1e-8, None)
    return transformed[:, :3] / w


def project_points(points_world: np.ndarray, view: Mat4, projection: Mat4) -> tuple[np.ndarray, np.ndarray]:
    """Chiếu điểm world-space sang NDC và trả về depth clip-space."""
    ones = np.ones((points_world.shape[0], 1), dtype=np.float32)
    world_h = np.hstack((points_world.astype(np.float32), ones))
    clip = (projection @ view @ world_h.T).T
    w = np.clip(clip[:, 3:4], 1e-8, None)
    ndc = clip[:, :3] / w
    return ndc[:, :2], ndc[:, 2]


def ndc_to_screen(ndc_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Đổi tọa độ NDC [-1, 1] sang tọa độ pixel màn hình."""
    # Trục Y của NDC hướng lên, còn ảnh 2D thường có gốc trên-trái nên cần đảo Y.
    x = (ndc_xy[:, 0] * 0.5 + 0.5) * width
    y = (1.0 - (ndc_xy[:, 1] * 0.5 + 0.5)) * height
    return np.column_stack((x, y))


@dataclass
class AABB:
    """Bounding box song song trục, lưu bằng góc min/max trong local space."""

    min_corner: np.ndarray
    max_corner: np.ndarray

    def corners(self) -> np.ndarray:
        """Trả về 8 góc dùng để chiếu AABB 3D thành bbox 2D."""
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
