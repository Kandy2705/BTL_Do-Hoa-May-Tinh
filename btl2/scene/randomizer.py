"""Deterministic random utilities used by scene generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Randomizer:
    """Thin wrapper around NumPy Generator with helper sampling methods."""

    seed: int

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def uniform(self, low: float, high: float) -> float:
        """Sample a float uniformly in [low, high)."""
        return float(self.rng.uniform(low, high))

    def randint(self, low: int, high: int) -> int:
        """Sample an integer in [low, high] inclusive."""
        return int(self.rng.integers(low, high + 1))

    def choice(self, values: list):
        """Sample one element from a list."""
        return values[int(self.rng.integers(0, len(values)))]

    def vector_uniform(self, low: float, high: float, size: int) -> np.ndarray:
        """Sample a float vector with i.i.d. uniform entries."""
        return self.rng.uniform(low, high, size=size).astype(np.float32)
