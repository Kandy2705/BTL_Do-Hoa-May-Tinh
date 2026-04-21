"""Sinh camera kiểu dashcam cho scene procedural của BTL 2."""

from __future__ import annotations

import math

import numpy as np

from btl2.scene.randomizer import Randomizer
from btl2.scene.scene import CameraState
from btl2.utils.math3d import normalize


def build_dashcam_camera(config: dict, image_width: int, image_height: int, randomizer: Randomizer) -> CameraState:
    """Tạo camera nhìn về phía trước, có jitter nhẹ như camera gắn trên xe."""
    # Chiều cao/pitch/yaw được nhiễu nhẹ để dataset không chỉ có một góc nhìn.
    # Vì dùng `Randomizer(seed)`, các nhiễu này vẫn tái lập được khi chạy lại.
    height = config["base_height"] + randomizer.uniform(-config["height_jitter"], config["height_jitter"])
    pitch_deg = config["base_pitch_degrees"] + randomizer.uniform(
        -config["pitch_jitter_degrees"],
        config["pitch_jitter_degrees"],
    )
    yaw_deg = randomizer.uniform(-config["yaw_jitter_degrees"], config["yaw_jitter_degrees"])
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    # Hệ trục scene: camera đứng tại gốc, nhìn chủ yếu theo +Z; pitch điều khiển
    # nhìn lên/xuống, yaw điều khiển lệch trái/phải.
    forward = np.array(
        [
            math.sin(yaw),
            math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ],
        dtype=np.float32,
    )
    forward = normalize(forward)
    position = np.array([0.0, height, 0.0], dtype=np.float32)
    # Target đặt xa phía trước đủ lớn để ma trận look_at ổn định.
    target = position + forward * 25.0
    return CameraState(
        position=position,
        target=target,
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        fov_y_degrees=float(config["fov_y_degrees"]),
        near=float(config["near"]),
        far=float(config["far"]),
        image_width=int(image_width),
        image_height=int(image_height),
    )
