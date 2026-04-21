"""
XUẤT DỮ LIỆU CUSTOM CSV CHO ĐỘI NGƯỜI MUỐN SCHEMA PHẲNG.

Hàm này tạo file CSV với định dạng đơn giản, dễ đọc trong spreadsheet:
- Tách train/val thành 2 file riêng biệt
- Mỗi bounding box là 1 row
- Chứa đầy đủ metadata về files và annotations
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CustomCsvExporter:
    """
    EXPORTER CSV TÙY CHỈNH CHO DETECTION DATASET.

    Mục đích:
    - Tạo file CSV dễ đọc cho data analysis
    - Tách train/validation thành 2 files
    - Mỗi bounding box là 1 row với đầy đủ metadata

    Schema:
    - File paths: image, depth, mask, metadata, yolo
    - Bounding boxes: xyxy và xywh formats
    - Quality metrics: visibility, occlusion ratios
    """

    rows_by_split: dict[str, list[dict]] = field(default_factory=lambda: {"train": [], "val": []})
    # Lưu trữ rows theo split: {"train": [...], "val": [...]}

    def add_frame(self, frame_id: str, split: str, bboxes: list[dict], paths: dict[str, str]) -> None:
        """
        THÊM MỘT FRAME VÀO CSV DATASET.

        Args:
            frame_id: ID của frame (VD: "0001", "0002")
            split: "train" hoặc "val"
            bboxes: List các bounding boxes từ detection
            paths: Dict chứa paths đến các files output
        """
        for bbox in bboxes:
            # Lấy tọa độ bounding box từ 2 formats
            x_min, y_min, x_max, y_max = bbox["bbox_xyxy"]  # Top-left, bottom-right
            x, y, w, h = bbox["bbox_xywh"]           # Top-left, width, height

            # TẠO ROW MỚI CHO BOUNDING BOX NÀY
            self.rows_by_split[split].append(
                {
                    # === METADATA CỦA FRAME ===
                    "frame_id": frame_id,                              # ID của frame
                    "split": split,                                     # Train/val split
                    "image_file": Path(paths["rgb"]).name,             # Tên file ảnh RGB
                    "depth_file": Path(paths["depth_png"]).name,       # Tên file depth
                    "mask_file": Path(paths["mask"]).name,             # Tên file mask
                    "metadata_file": Path(paths["metadata"]).name,       # Tên file metadata
                    "yolo_file": Path(paths["yolo"]).name,           # Tên file YOLO format

                    # === BOUNDING BOX INFO ===
                    "instance_id": bbox["instance_id"],                  # ID duy nhất của object
                    "class_id": bbox["class_id"],                      # ID semantic class
                    "class_name": bbox["class_name"],                    # Tên class (VD: "car")

                    # === TOẠ ĐỘ XYXY FORMAT ===
                    "x_min": round(x_min, 3),                         # Left edge
                    "y_min": round(y_min, 3),                         # Top edge
                    "x_max": round(x_max, 3),                         # Right edge
                    "y_max": round(y_max, 3),                         # Bottom edge

                    # === TOẠ ĐỘ XYWH FORMAT ===
                    "bbox_x": round(x, 3),                              # Left edge (same as x_min)
                    "bbox_y": round(y, 3),                              # Top edge (same as y_min)
                    "bbox_w": round(w, 3),                              # Width of bounding box
                    "bbox_h": round(h, 3),                              # Height of bounding box

                    # === QUALITY METRICS ===
                    "visibility_ratio": round(float(bbox.get("visibility_ratio", 0.0)), 6),  # Tỷ lệ visible [0,1]
                    "occlusion_ratio": round(float(bbox.get("occlusion_ratio", 0.0)), 6), # Tỷ lệ bị che khuất [0,1]
                }
            )

    def write(self, output_dir: str | Path) -> None:
        """
        GHI FILE CSV VỚI COLUMN ORDER ỔN ĐỊNH.

        Args:
            output_dir: Thư mục output để lưu CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)  # Tạo thư mục nếu chưa có

        # ĐỊNH NGHĨA CÁC CỘT TRONG CSV - PHẢI ĐÚNG THỨ TỰ
        fieldnames = [
            "frame_id",          # ID của frame
            "split",             # Train/val split
            "image_file",        # Tên file ảnh
            "depth_file",        # Tên file depth
            "mask_file",         # Tên file mask
            "metadata_file",     # Tên file metadata
            "yolo_file",        # Tên file YOLO
            "instance_id",       # ID của object
            "class_id",          # ID semantic class
            "class_name",        # Tên class
            "x_min",             # Left edge
            "y_min",             # Top edge
            "x_max",             # Right edge
            "y_max",             # Bottom edge
            "bbox_x",            # Left edge (XYWH)
            "bbox_y",            # Top edge (XYWH)
            "bbox_w",            # Width
            "bbox_h",            # Height
            "visibility_ratio",   # Tỷ lệ visible
            "occlusion_ratio",   # Tỷ lệ bị che khuất
        ]

        # GHI 2 FILES: train.csv và val.csv
        for split in ("train", "val"):
            csv_path = output_path / f"{split}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()  # Ghi header row
                writer.writerows(self.rows_by_split[split])  # Ghi tất cả rows
