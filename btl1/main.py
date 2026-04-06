"""Clean entrypoint for BTL 1.

BTL 1 still runs on the legacy root modules (`main.py`, `controller.py`,
`viewer.py`, `model.py`, `geometry/`, `components/`, `libs/`).
This wrapper exists only to make the shared repository structure clearer.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main as legacy_main


def main() -> None:
    """Run the original BTL 1 application."""
    legacy_main()


if __name__ == "__main__":
    main()
