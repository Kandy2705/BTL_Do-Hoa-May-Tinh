"""Helper xuất nhãn YOLO cho dataset BTL 2."""

from __future__ import annotations

from pathlib import Path


def export_yolo_labels(bboxes: list[dict], image_width: int, image_height: int) -> list[str]:
    """Chuyển bbox pixel sang chuỗi YOLO đã normalize theo kích thước ảnh."""
    lines: list[str] = []
    for bbox in bboxes:
        x, y, w, h = bbox["bbox_xywh"]
        # YOLO yêu cầu tọa độ tâm và kích thước box trong khoảng [0, 1],
        # không dùng trực tiếp x_min/y_min theo pixel như COCO.
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
    """Ghi `dataset.yaml` để Ultralytics/YOLO biết path train/val và tên class."""
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
