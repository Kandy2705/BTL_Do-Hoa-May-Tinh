"""Depth-map conversion and export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from btl2.utils.image import save_grayscale


def linearize_depth(depth_buffer: np.ndarray, near: float, far: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert non-linear OpenGL depth into metric-like linear depth and 8-bit preview."""
    z = depth_buffer * 2.0 - 1.0
    linear = (2.0 * near * far) / np.clip(far + near - z * (far - near), 1e-6, None)

    finite = linear[np.isfinite(linear)]
    scene_far = float(np.percentile(finite, 95.0)) if finite.size else float(far)
    scene_far = min(scene_far, 40.0)
    visual_far = min(float(far), max(float(near) + 1.0, scene_far))
    normalized = (linear - near) / max(visual_far - near, 1e-6)

    # Keep depth previews soft and readable for near road scenes instead of
    # mapping close objects to nearly black against a very large far plane.
    grayscale = np.clip(92.0 + normalized * 140.0, 0.0, 245.0).astype(np.uint8)
    return linear.astype(np.float32), grayscale


def save_depth_outputs(
    png_path: str | Path,
    npy_path: str | Path,
    depth_gray: np.ndarray,
    depth_linear: np.ndarray,
    save_npy: bool,
) -> None:
    """Write the visualized PNG and optional raw .npy depth array."""
    save_grayscale(png_path, depth_gray)
    if save_npy:
        np.save(npy_path, depth_linear.astype(np.float32))
