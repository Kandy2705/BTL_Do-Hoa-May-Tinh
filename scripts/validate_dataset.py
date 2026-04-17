#!/usr/bin/env python3
"""Validate generated BTL2 dataset outputs with the strict consistency checker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btl2.annotations.dataset_consistency import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a BTL2 dataset")
    parser.add_argument("dataset_root", nargs="?", default=str(ROOT / "outputs" / "btl2" / "showcase_dataset"))
    parser.add_argument("--fix", action="store_true", help="Remove junk files and orphan generated artifacts")
    parser.add_argument("--no-require-depth", action="store_true", help="Allow datasets that are YOLO-training-only")
    parser.add_argument("--require-depth-npy", action="store_true", help="Require numeric depth .npy files next to depth PNG previews")
    parser.add_argument("--no-require-coco", action="store_true", help="Allow datasets without COCO JSON")
    parser.add_argument("--no-require-mask-pixels", action="store_true", help="Allow external metadata mask-color mismatches")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()

    report = validate_dataset(
        dataset_root,
        fix=args.fix,
        require_depth=not args.no_require_depth,
        require_depth_npy=args.require_depth_npy,
        require_coco=not args.no_require_coco,
        require_mask_pixels=not args.no_require_mask_pixels,
    )
    report_path = dataset_root / "quality_report.json"
    report_path.write_text(__import__("json").dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Validation status: {report['summary']['status']}")
    print(f"Issues: {report['summary']['total_issues']}")
    print(f"Warnings: {report['summary']['total_warnings']}")
    print(f"Report: {report_path}")
    return 0 if report["summary"]["total_issues"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
