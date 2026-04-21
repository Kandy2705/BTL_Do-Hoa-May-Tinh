"""Tiện ích màu cho class và instance segmentation của BTL 2."""

from __future__ import annotations

from typing import Iterable

import numpy as np


CLASS_COLOR_MAP = {
    # Màu class dùng cho preview/debug. Segmentation của object động dùng màu
    # instance riêng, nhưng màu class vẫn hữu ích cho road và fallback material.
    "person": (70, 200, 120),
    "car": (220, 70, 70),
    "bus": (255, 140, 70),
    "truck": (160, 110, 80),
    "motorbike": (70, 150, 240),
    "traffic_light": (90, 150, 255),
    "traffic_sign": (255, 210, 60),
    "road": (90, 90, 90),
    "lane_marking": (230, 230, 230),
    "background": (135, 206, 235),
}


def class_color(name: str) -> tuple[int, int, int]:
    """Trả về màu RGB ổn định cho một semantic class."""
    return CLASS_COLOR_MAP.get(name, (180, 180, 180))


def instance_color(instance_id: int) -> tuple[int, int, int]:
    """Mã hóa instance id số nguyên thành bộ màu RGB duy nhất trong 24 bit."""
    value = int(instance_id) & 0xFFFFFF
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def color_to_float(color: Iterable[int]) -> np.ndarray:
    """Đổi RGB 0-255 sang float32 [0, 1] để truyền vào shader."""
    return np.asarray(list(color), dtype=np.float32) / 255.0
