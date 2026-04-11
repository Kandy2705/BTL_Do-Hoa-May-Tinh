"""Validate generated YOLO, COCO, mask, and metadata outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btl2.annotations.validators import run_full_validation


def main() -> int:
    """Run all validators and print a concise report."""
    dataset_root = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ROOT / "outputs" / "btl2" / "demo_dataset"
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    issues = run_full_validation(dataset_root)
    if issues:
        print("Validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Validation passed for {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
