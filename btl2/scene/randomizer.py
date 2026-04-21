"""Tiện ích random có seed cố định cho quá trình sinh scene BTL 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Randomizer:
    """Wrapper mỏng quanh NumPy Generator để mọi random đều đi qua cùng seed."""

    seed: int

    def __post_init__(self) -> None:
        # Dùng default_rng để kết quả ổn định và tách biệt khỏi random global.
        self.rng = np.random.default_rng(self.seed)

    def uniform(self, low: float, high: float) -> float:
        """Lấy một số thực phân bố đều trong [low, high)."""
        return float(self.rng.uniform(low, high))

    def randint(self, low: int, high: int) -> int:
        """Lấy số nguyên trong [low, high], bao gồm cả high."""
        return int(self.rng.integers(low, high + 1))

    def choice(self, values: list):
        """Chọn ngẫu nhiên một phần tử trong list."""
        return values[int(self.rng.integers(0, len(values)))]

    def vector_uniform(self, low: float, high: float, size: int) -> np.ndarray:
        """Lấy vector số thực, mỗi phần tử độc lập theo uniform [low, high)."""
        return self.rng.uniform(low, high, size=size).astype(np.float32)
