"""Các validator nhẹ để kiểm tra dataset BTL 2 sau khi xuất."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _iter_image_files(image_dir: Path) -> list[Path]:
    """Lấy danh sách ảnh RGB với các đuôi phổ biến."""
    image_paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(sorted(image_dir.glob(pattern)))
    return sorted(image_paths)


def _resolve_mask_path(mask_dir: Path, stem: str) -> Path | None:
    """Tìm mask theo nhiều quy ước tên file để tương thích dataset cũ/mới."""
    for candidate in (mask_dir / f"{stem}_mask.png", mask_dir / f"{stem}_layer.png", mask_dir / f"{stem}.png"):
        if candidate.exists():
            return candidate
    return None


def validate_yolo_labels(labels_dir: str | Path) -> list[str]:
    """Kiểm tra mỗi dòng YOLO có 5 giá trị và tọa độ đã normalize trong [0, 1]."""
    issues: list[str] = []
    for path in sorted(Path(labels_dir).glob("*.txt")):
        for line_idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            # Format YOLO detection: class_id x_center y_center width height.
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
    """Kiểm tra COCO JSON có key bắt buộc và bbox có kích thước dương."""
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
    """Đảm bảo mỗi ảnh đều có mask, YOLO label và metadata tương ứng."""
    root = Path(output_root)
    issues: list[str] = []
    for split in ("train", "val"):
        image_dir = root / "images" / split
        mask_dir = root / "masks" / split
        labels_dir = root / "labels_yolo" / split
        metadata_dir = root / "metadata" / split
        for image_path in _iter_image_files(image_dir):
            stem = image_path.stem
            # Quy ước cùng stem giúp phát hiện ngay frame bị thiếu một loại artifact.
            if _resolve_mask_path(mask_dir, stem) is None:
                issues.append(f"{image_path}: missing mask image.")
            if not (labels_dir / f"{stem}.txt").exists():
                issues.append(f"{image_path}: missing YOLO label.")
            if not (metadata_dir / f"{stem}.json").exists():
                issues.append(f"{image_path}: missing metadata JSON.")
    return issues


def validate_instance_masks(mask_dir: str | Path) -> list[str]:
    """Kiểm tra mask có ít nhất một màu foreground ngoài nền đen."""
    issues: list[str] = []
    for mask_path in sorted(Path(mask_dir).glob("*.png")):
        image = Image.open(mask_path).convert("RGB")
        seen = {color for color in image.getdata() if color != (0, 0, 0)}
        if len(seen) == 0:
            issues.append(f"{mask_path}: no foreground instance ids found.")
    return issues


def run_full_validation(output_root: str | Path) -> list[str]:
    """Chạy toàn bộ validator cơ bản trên cây thư mục dataset dự kiến."""
    root = Path(output_root)
    issues = []
    issues.extend(validate_dataset_tree(root))
    for split in ("train", "val"):
        issues.extend(validate_yolo_labels(root / "labels_yolo" / split))
        mask_dir = root / "masks" / split
        if mask_dir.exists():
            issues.extend(validate_instance_masks(mask_dir))

    coco_dir = root / "annotations_coco"
    train_coco = coco_dir / "train.json"
    val_coco = coco_dir / "val.json"
    if train_coco.exists():
        issues.extend(validate_coco_json(train_coco))
    if val_coco.exists():
        issues.extend(validate_coco_json(val_coco))
    return issues
