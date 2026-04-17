#!/usr/bin/env python3
"""CLI wrapper for the BTL2 dataset consistency checker."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btl2.annotations.dataset_consistency import main


if __name__ == "__main__":
    raise SystemExit(main())
