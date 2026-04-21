"""Helper sinh ánh sáng cho scene procedural của BTL 2."""

from __future__ import annotations

import math

import numpy as np

from btl2.scene.scene import DirectionalLight
from btl2.scene.randomizer import Randomizer
from btl2.utils.math3d import normalize


def sample_directional_light(config: dict, randomizer: Randomizer) -> DirectionalLight:
    """Lấy mẫu một directional light từ các khoảng cấu hình."""
    # Pitch/yaw thay đổi nhẹ giữa frame để ảnh RGB đa dạng hơn về bóng/độ sáng.
    pitch = math.radians(randomizer.uniform(*config["directional_pitch_range"]))
    yaw = math.radians(randomizer.uniform(*config["directional_yaw_range"]))
    direction = np.array(
        [
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
            math.cos(pitch) * math.cos(yaw),
        ],
        dtype=np.float32,
    )
    intensity = randomizer.uniform(*config["directional_intensity_range"])
    # Giữ màu trắng để dataset không bị bias màu ánh sáng; độ đa dạng đến từ hướng
    # và intensity, không phải tint màu.
    color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return DirectionalLight(
        direction=normalize(direction),
        color=color,
        intensity=intensity,
        ambient_strength=float(config["ambient_strength"]),
    )
