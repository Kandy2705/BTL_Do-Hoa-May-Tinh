"""Material tối giản cho màu RGB và màu segmentation của object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    """Màu riêng của object được shader RGB/segmentation sử dụng."""

    base_color: np.ndarray
    segmentation_color: np.ndarray
