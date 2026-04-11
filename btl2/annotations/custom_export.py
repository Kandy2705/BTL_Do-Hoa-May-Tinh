"""Custom CSV export for teams that want a flat, spreadsheet-friendly schema."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CustomCsvExporter:
    """Collect per-bbox rows and write one CSV per split."""

    rows_by_split: dict[str, list[dict]] = field(default_factory=lambda: {"train": [], "val": []})

    def add_frame(self, frame_id: str, split: str, bboxes: list[dict], paths: dict[str, str]) -> None:
        """Append one CSV row per bounding box."""
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox["bbox_xyxy"]
            x, y, w, h = bbox["bbox_xywh"]
            self.rows_by_split[split].append(
                {
                    "frame_id": frame_id,
                    "split": split,
                    "image_file": Path(paths["rgb"]).name,
                    "depth_file": Path(paths["depth_png"]).name,
                    "mask_file": Path(paths["mask"]).name,
                    "metadata_file": Path(paths["metadata"]).name,
                    "yolo_file": Path(paths["yolo"]).name,
                    "instance_id": bbox["instance_id"],
                    "class_id": bbox["class_id"],
                    "class_name": bbox["class_name"],
                    "x_min": round(x_min, 3),
                    "y_min": round(y_min, 3),
                    "x_max": round(x_max, 3),
                    "y_max": round(y_max, 3),
                    "bbox_x": round(x, 3),
                    "bbox_y": round(y, 3),
                    "bbox_w": round(w, 3),
                    "bbox_h": round(h, 3),
                    "visibility_ratio": round(float(bbox.get("visibility_ratio", 0.0)), 6),
                    "occlusion_ratio": round(float(bbox.get("occlusion_ratio", 0.0)), 6),
                }
            )

    def write(self, output_dir: str | Path) -> None:
        """Write train/val CSV files with a stable column order."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "frame_id",
            "split",
            "image_file",
            "depth_file",
            "mask_file",
            "metadata_file",
            "yolo_file",
            "instance_id",
            "class_id",
            "class_name",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "visibility_ratio",
            "occlusion_ratio",
        ]
        for split in ("train", "val"):
            csv_path = output_path / f"{split}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows_by_split[split])
