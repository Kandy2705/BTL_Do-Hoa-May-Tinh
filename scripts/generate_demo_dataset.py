"""Generate a small demo dataset using the YAML configuration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btl2.app import SyntheticRoadApp
from btl2.utils.io import load_yaml
from btl2.utils.logging_utils import configure_logging


def main() -> int:
    """Load the demo config and generate the dataset."""
    logger = configure_logging()
    config = load_yaml(ROOT / "configs" / "btl2" / "demo_small.yaml")
    app = SyntheticRoadApp(config)
    try:
        summaries = app.generate_dataset()
    finally:
        app.close()
    logger.info("Generated %d demo frames in %s", len(summaries), config["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
