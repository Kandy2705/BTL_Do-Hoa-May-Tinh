"""Small material abstraction for RGB and segmentation colors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    """Per-object colors used by the shaders."""

    base_color: np.ndarray
    segmentation_color: np.ndarray
