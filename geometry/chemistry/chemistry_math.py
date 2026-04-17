"""Small transform helpers for the atom/molecule visualizer."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def rotate_y_point(point: Sequence[float], angle_rad: float) -> list[float]:
    """Rotate one 3D point around the Y axis."""
    x, y, z = [float(v) for v in point[:3]]
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [x * c + z * s, y, -x * s + z * c]


def orbit_position(radius: float, theta: float, rotation_deg: Sequence[float]) -> list[float]:
    """Return an electron position on a rotated circular orbit."""
    point = [float(radius) * math.cos(theta), 0.0, float(radius) * math.sin(theta)]
    x, y, z = point
    rx, ry, rz = [math.radians(float(v)) for v in rotation_deg[:3]]

    cy, sy = math.cos(ry), math.sin(ry)
    x, z = x * cy + z * sy, -x * sy + z * cy
    cx, sx = math.cos(rx), math.sin(rx)
    y, z = y * cx - z * sx, y * sx + z * cx
    cz, sz = math.cos(rz), math.sin(rz)
    x, y = x * cz - y * sz, x * sz + y * cz
    return [x, y, z]


def bond_transform_xy(p1: Sequence[float], p2: Sequence[float]) -> tuple[list[float], list[float], float]:
    """Compute midpoint, Euler rotation and length for a cylinder bond.

    The app's cylinder primitive is aligned with the local Y axis. For the
    minimal H2O/CO2 demos, atoms are built in the XY plane, so a Z rotation is
    enough to align the cylinder with the atom-to-atom vector.
    """
    start = np.asarray(p1[:3], dtype=np.float32)
    end = np.asarray(p2[:3], dtype=np.float32)
    mid = (start + end) * 0.5
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return mid.tolist(), [0.0, 0.0, 0.0], 1.0
    direction = vec / length
    angle_z = -math.degrees(math.atan2(float(direction[0]), float(direction[1])))
    return mid.tolist(), [0.0, 0.0, angle_z], length
