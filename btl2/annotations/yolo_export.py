"""YOLO label export helpers."""

from __future__ import annotations

from pathlib import Path


def export_yolo_labels(bboxes: list[dict], image_width: int, image_height: int) -> list[str]:
    """Convert bbox annotations into normalized YOLO strings."""
    lines: list[str] = []
    for bbox in bboxes:
        x, y, w, h = bbox["bbox_xywh"]
        x_center = (x + w * 0.5) / image_width
        y_center = (y + h * 0.5) / image_height
        width = w / image_width
        height = h / image_height
        lines.append(
            f"{bbox['class_id']} "
            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return lines


def write_dataset_yaml(output_root: str | Path, categories: list[dict]) -> None:
    """Write a YOLO dataset.yaml that points to train/val images and class names."""
    lines = [
        f"path: {Path(output_root).as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(categories)}",
        "names:",
    ]
    for category in categories:
        lines.append(f"  {category['id']}: {category['name']}")
    (Path(output_root) / "dataset.yaml").write_text("\n".join(lines), encoding="utf-8")
