"""Dataset validators used by the CLI and helper scripts."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def validate_yolo_labels(labels_dir: str | Path) -> list[str]:
    """Check YOLO labels have five values and normalized ranges."""
    issues: list[str] = []
    for path in sorted(Path(labels_dir).glob("*.txt")):
        for line_idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                issues.append(f"{path}: line {line_idx} should contain 5 values.")
                continue
            try:
                _, xc, yc, w, h = map(float, parts)
            except ValueError:
                issues.append(f"{path}: line {line_idx} contains non-float values.")
                continue
            for value in (xc, yc, w, h):
                if not 0.0 <= value <= 1.0:
                    issues.append(f"{path}: line {line_idx} has out-of-range YOLO values.")
                    break
    return issues


def validate_coco_json(coco_path: str | Path) -> list[str]:
    """Check required COCO keys and a few bbox constraints."""
    issues: list[str] = []
    payload = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in payload:
            issues.append(f"{coco_path}: missing key '{key}'.")
    for annotation in payload.get("annotations", []):
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            issues.append(f"{coco_path}: annotation {annotation.get('id')} has invalid bbox.")
        elif bbox[2] <= 0 or bbox[3] <= 0:
            issues.append(f"{coco_path}: annotation {annotation.get('id')} has non-positive bbox size.")
    return issues


def validate_dataset_tree(output_root: str | Path) -> list[str]:
    """Verify images, masks, YOLO labels, and metadata exist for every frame."""
    root = Path(output_root)
    issues: list[str] = []
    for split in ("train", "val"):
        image_dir = root / "images" / split
        mask_dir = root / "masks" / split
        labels_dir = root / "labels_yolo" / split
        metadata_dir = root / "metadata" / split
        for image_path in sorted(image_dir.glob("*.png")):
            stem = image_path.stem
            if not (mask_dir / f"{stem}_mask.png").exists():
                issues.append(f"{image_path}: missing mask image.")
            if not (labels_dir / f"{stem}.txt").exists():
                issues.append(f"{image_path}: missing YOLO label.")
            if not (metadata_dir / f"{stem}.json").exists():
                issues.append(f"{image_path}: missing metadata JSON.")
    return issues


def validate_instance_masks(mask_dir: str | Path) -> list[str]:
    """Check that each mask contains at least one non-background instance color."""
    issues: list[str] = []
    for mask_path in sorted(Path(mask_dir).glob("*.png")):
        image = Image.open(mask_path).convert("RGB")
        seen = {color for color in image.getdata() if color != (0, 0, 0)}
        if len(seen) == 0:
            issues.append(f"{mask_path}: no foreground instance ids found.")
    return issues


def run_full_validation(output_root: str | Path) -> list[str]:
    """Run every validator over the expected dataset tree."""
    root = Path(output_root)
    issues = []
    issues.extend(validate_dataset_tree(root))
    for split in ("train", "val"):
        issues.extend(validate_yolo_labels(root / "labels_yolo" / split))
        issues.extend(validate_instance_masks(root / "masks" / split))
    issues.extend(validate_coco_json(root / "annotations_coco" / "train.json"))
    issues.extend(validate_coco_json(root / "annotations_coco" / "val.json"))
    return issues
