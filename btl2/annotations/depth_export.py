"""Chuyển đổi và lưu depth map cho dataset BTL 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from btl2.utils.image import save_grayscale


def linearize_depth(depth_buffer: np.ndarray, near: float, far: float) -> tuple[np.ndarray, np.ndarray]:
    """Đổi depth phi tuyến của OpenGL sang depth tuyến tính và ảnh preview 8-bit."""
    # OpenGL depth buffer nằm trong [0, 1] nhưng không tuyến tính theo khoảng cách.
    # Đưa về NDC z [-1, 1] rồi áp công thức nghịch đảo projection perspective.
    z = depth_buffer * 2.0 - 1.0
    linear = (2.0 * near * far) / np.clip(far + near - z * (far - near), 1e-6, None)

    # PNG preview không cần giữ toàn bộ dải far plane rất xa; dùng percentile 95
    # để ảnh depth nhìn rõ object/đường gần camera hơn.
    finite = linear[np.isfinite(linear)]
    scene_far = float(np.percentile(finite, 95.0)) if finite.size else float(far)
    scene_far = min(scene_far, 40.0)
    visual_far = min(float(far), max(float(near) + 1.0, scene_far))
    normalized = (linear - near) / max(visual_far - near, 1e-6)

    # Giữ cho depth preview dễ nhìn và dễ đọc trong các cảnh đường phố thay vì
    # mapping các vật gần thành màu gần đen so với mặt phẳng xa rất lớn.
    grayscale = np.clip(92.0 + normalized * 140.0, 0.0, 245.0).astype(np.uint8)
    return linear.astype(np.float32), grayscale


def save_depth_outputs(
    png_path: str | Path,
    npy_path: str | Path,
    depth_gray: np.ndarray,
    depth_linear: np.ndarray,
    save_npy: bool,
) -> None:
    """Ghi PNG depth đã visualize và tùy chọn mảng depth tuyến tính `.npy`."""
    save_grayscale(png_path, depth_gray)
    if save_npy:
        np.save(npy_path, depth_linear.astype(np.float32))
