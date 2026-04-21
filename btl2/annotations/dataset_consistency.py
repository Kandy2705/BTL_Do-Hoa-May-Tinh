"""Kiểm tra và tùy chọn dọn dẹp output dataset BTL 2.

Đây là lớp kiểm tra phòng thủ cho pipeline nộp bài. Nó xác nhận RGB, mask,
depth, nhãn YOLO, metadata và COCO annotation khớp nhau theo frame stem. Khi bật
`--fix`, checker sẽ xóa file rác hệ điều hành và artifact mồ côi không có ảnh RGB.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

VALID_CLASS_IDS = set(range(7))
EXPECTED_SPLITS = ("train", "val")
JUNK_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
JUNK_DIR_NAMES = {"__MACOSX", ".ipynb_checkpoints", "__pycache__"}


class DatasetChecker:
    """Validator trạng thái đầy đủ của một cây thư mục dataset BTL 2."""

    def __init__(
        self,
        root: Path,
        fix: bool = False,
        require_depth: bool = True,
        require_depth_npy: bool = False,
        require_coco: bool = True,
        require_mask_pixels: bool = True,
    ) -> None:
        self.root = root
        self.fix = fix
        self.require_depth = require_depth
        self.require_depth_npy = require_depth_npy
        self.require_coco = require_coco
        self.require_mask_pixels = require_mask_pixels
        self.report: dict[str, Any] = {
            "dataset_root": str(root),
            "fixed": [],
            "splits": {},
            "coco": {},
            "optional_missing": {},
            "summary": {},
        }
        self.issues: list[str] = []
        self.warnings: list[str] = []

    @staticmethod
    def _frame_stem(path: Path) -> str:
        """Chuẩn hóa tên frame bằng cách bỏ hậu tố artifact như `_depth`, `_mask`."""
        stem = path.stem
        for suffix in ("_depth", "_mask", "_layer"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    @staticmethod
    def _read_image(path: Path, mode: str = "RGB") -> Image.Image | None:
        try:
            return Image.open(path).convert(mode)
        except Exception:
            return None

    @staticmethod
    def _size(path: Path) -> tuple[int, int] | None:
        try:
            with Image.open(path) as img:
                return img.size
        except Exception:
            return None

    def _issue(self, message: str) -> None:
        self.issues.append(message)

    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def _remove_path(self, path: Path, reason: str) -> None:
        if not self.fix:
            return
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            self.report["fixed"].append({"path": str(path), "reason": reason})
        except Exception as exc:
            self._issue(f"failed to remove {path}: {exc}")

    def cleanup_junk(self) -> None:
        """Tìm file/thư mục rác phổ biến và xóa nếu `fix=True`."""
        if not self.root.exists():
            self._issue(f"dataset root does not exist: {self.root}")
            return
        for path in sorted(self.root.rglob("*")):
            if path.name in JUNK_NAMES:
                self._warn(f"junk file: {path}")
                self._remove_path(path, "junk file")
            elif path.is_dir() and path.name in JUNK_DIR_NAMES:
                self._warn(f"junk directory: {path}")
                self._remove_path(path, "junk directory")

    def _collect(self, split: str) -> dict[str, dict[str, Path]]:
        """Thu thập artifact của một split, gom theo stem frame."""
        folders = {
            "images": self.root / "images" / split,
            "masks": self.root / "masks" / split,
            "labels_yolo": self.root / "labels_yolo" / split,
            "metadata": self.root / "metadata" / split,
            "depth_png": self.root / "depth" / split,
            "depth_npy": self.root / "depth" / split,
        }
        patterns = {
            "images": ("*.png", "*.jpg", "*.jpeg"),
            "masks": ("*_mask.png", "*_layer.png", "*.png"),
            "labels_yolo": ("*.txt",),
            "metadata": ("*.json",),
            "depth_png": ("*_depth.png",),
            "depth_npy": ("*_depth.npy",),
        }
        collected: dict[str, dict[str, Path]] = {}
        for key, folder in folders.items():
            if not folder.exists():
                collected[key] = {}
                continue
            items: dict[str, Path] = {}
            for pattern in patterns[key]:
                for path in sorted(folder.glob(pattern)):
                    if path.name.startswith("."):
                        continue
                    items[self._frame_stem(path)] = path
            collected[key] = items
        return collected

    def _remove_orphans(self, split: str, collected: dict[str, dict[str, Path]]) -> None:
        """Cảnh báo/xóa artifact không có ảnh RGB tương ứng."""
        image_stems = set(collected["images"].keys())
        for group in ("masks", "labels_yolo", "metadata", "depth_png", "depth_npy"):
            for stem, path in sorted(collected[group].items()):
                if stem not in image_stems:
                    self._warn(f"orphan {group}: {path}")
                    self._remove_path(path, f"orphan {group} without RGB image in {split}")

    def _check_yolo(self, path: Path, width: int, height: int) -> list[dict[str, Any]]:
        """Parse và kiểm tra file YOLO, đồng thời trả bbox đã decode sang pixel."""
        parsed: list[dict[str, Any]] = []
        if not path.exists():
            self._issue(f"missing YOLO label: {path}")
            return parsed
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                self._issue(f"{path}:{line_no} expected 5 YOLO columns, got {len(parts)}")
                continue
            try:
                cls_float = float(parts[0])
                cls = int(cls_float)
                xc, yc, bw, bh = [float(v) for v in parts[1:]]
            except ValueError:
                self._issue(f"{path}:{line_no} non-numeric YOLO value")
                continue
            if not math.isclose(cls_float, cls):
                self._issue(f"{path}:{line_no} class id must be integer-like")
            if cls not in VALID_CLASS_IDS:
                self._issue(f"{path}:{line_no} class id {cls} outside expected 0..6")
            if any(v < 0.0 or v > 1.0 for v in (xc, yc, bw, bh)):
                self._issue(f"{path}:{line_no} normalized bbox outside [0,1]")
            if bw <= 0.0 or bh <= 0.0:
                self._issue(f"{path}:{line_no} bbox width/height must be positive")
            x1 = (xc - bw * 0.5) * width
            y1 = (yc - bh * 0.5) * height
            x2 = (xc + bw * 0.5) * width
            y2 = (yc + bh * 0.5) * height
            if x2 <= x1 or y2 <= y1:
                self._issue(f"{path}:{line_no} decoded bbox has invalid order")
            parsed.append({"class_id": cls, "bbox_xyxy": [x1, y1, x2, y2]})
        return parsed

    def _check_metadata(self, path: Path, width: int, height: int) -> list[dict[str, Any]]:
        """Kiểm tra metadata JSON và trả danh sách bbox trong metadata."""
        if not path.exists():
            self._issue(f"missing metadata: {path}")
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._issue(f"metadata parse failed: {path}: {exc}")
            return []
        boxes = payload.get("bounding_boxes", [])
        if not boxes and isinstance(payload.get("objects"), list):
            # Unity Perception style metadata used by the user's imported dataset.
            boxes = []
            for obj in payload.get("objects", []):
                bbox_yolo = obj.get("bbox_yolo")
                if not bbox_yolo or len(bbox_yolo) != 4:
                    continue
                xc, yc, bw, bh = [float(v) for v in bbox_yolo]
                boxes.append({
                    "instance_id": obj.get("instance_id"),
                    "class_name": obj.get("class_name"),
                    "class_id": obj.get("class_id"),
                    "bbox_xyxy": [
                        (xc - bw * 0.5) * width,
                        (yc - bh * 0.5) * height,
                        (xc + bw * 0.5) * width,
                        (yc + bh * 0.5) * height,
                    ],
                    "mask_color_rgb": obj.get("mask_color_rgb"),
                })
        if not isinstance(boxes, list):
            self._issue(f"metadata bounding_boxes must be a list: {path}")
            return []
        for idx, box in enumerate(boxes):
            bbox = box.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                self._issue(f"{path}: box {idx} missing bbox_xyxy")
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                self._issue(f"{path}: box {idx} has invalid bbox order")
            if x1 < -1 or y1 < -1 or x2 > width + 1 or y2 > height + 1:
                self._issue(f"{path}: box {idx} exceeds image bounds")
            if box.get("class_id") not in VALID_CLASS_IDS:
                self._issue(f"{path}: box {idx} invalid class_id={box.get('class_id')}")
        objects = payload.get("objects", [])
        if isinstance(objects, list):
            annotated_instances = {int(b.get("instance_id")) for b in boxes if b.get("instance_id") is not None}
            for obj in objects:
                if obj.get("instance_id") is None or int(obj.get("instance_id", 0)) <= 0:
                    continue
                is_visible = bool(obj.get("visible", True))
                if is_visible and int(obj.get("instance_id")) not in annotated_instances:
                    # This can happen for tiny/fully occluded objects, so warn rather than fail.
                    self._warn(f"visible object has no annotation: {path} instance={obj.get('instance_id')} class={obj.get('class_name')}")
        return boxes

    @staticmethod
    def _mask_pixels(mask_array: np.ndarray, color: tuple[int, int, int]) -> int:
        return int(np.all(mask_array == np.asarray(color, dtype=np.uint8), axis=2).sum())

    def _check_mask(self, path: Path, width: int, height: int, metadata_path: Path, boxes: list[dict[str, Any]]) -> None:
        """Kiểm tra mask tồn tại, đúng size và có pixel cho annotation cần thiết."""
        if not path.exists():
            self._issue(f"missing mask: {path}")
            return
        mask_img = self._read_image(path, "RGB")
        if mask_img is None:
            self._issue(f"cannot read mask: {path}")
            return
        if mask_img.size != (width, height):
            self._issue(f"mask size mismatch: {path} rgb={(width, height)} mask={mask_img.size}")
            return
        mask = np.asarray(mask_img, dtype=np.uint8)
        foreground = int(np.any(mask != 0, axis=2).sum())
        if boxes and foreground == 0:
            self._issue(f"mask has no foreground but annotations exist: {path}")

        # If metadata provides segmentation mapping, every annotated instance should have pixels.
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        except Exception:
            metadata = {}
        mapping = metadata.get("segmentation_mapping", {}) if isinstance(metadata, dict) else {}
        for box in boxes:
            inst = box.get("instance_id")
            if inst is None:
                continue
            matched_colors = []
            if box.get("mask_color_rgb") and len(box.get("mask_color_rgb")) == 3:
                matched_colors.append(tuple(int(v) for v in box["mask_color_rgb"]))
            for color_key, info in mapping.items():
                if int(info.get("instance_id", -999)) == int(inst):
                    try:
                        matched_colors.append(tuple(int(v) for v in color_key.split("_")))
                    except Exception:
                        pass
            if not matched_colors:
                self._warn(f"missing segmentation mapping for annotated instance {inst}: {metadata_path}")
                continue
            if all(self._mask_pixels(mask, color) == 0 for color in matched_colors):
                message = f"annotated instance has zero mask pixels: {path} instance={inst}"
                if self.require_mask_pixels:
                    self._issue(message)
                else:
                    self.report["optional_missing"].setdefault("mask_pixels", 0)
                    self.report["optional_missing"]["mask_pixels"] += 1

    def _check_depth(self, png_path: Path, npy_path: Path | None, width: int, height: int) -> None:
        if not png_path.exists():
            if self.require_depth:
                self._issue(f"missing depth PNG: {png_path}")
            return
        depth_img = self._read_image(png_path, "L")
        if depth_img is None:
            self._issue(f"cannot read depth PNG: {png_path}")
            return
        if depth_img.size != (width, height):
            self._issue(f"depth PNG size mismatch: {png_path} rgb={(width, height)} depth={depth_img.size}")
        arr = np.asarray(depth_img, dtype=np.float32)
        if float(arr.max() - arr.min()) < 2.0:
            self._warn(f"depth PNG has very low dynamic range: {png_path} min={arr.min():.1f} max={arr.max():.1f}")
        if self.require_depth_npy and (npy_path is None or not npy_path.exists()):
            expected = npy_path or png_path.with_name(png_path.name.replace("_depth.png", "_depth.npy"))
            self._issue(f"missing depth NPY: {expected}")
            return
        if npy_path is not None and npy_path.exists():
            try:
                raw = np.load(npy_path)
                if raw.shape[:2] != (height, width):
                    self._issue(f"depth NPY shape mismatch: {npy_path} rgb={(height, width)} npy={raw.shape}")
                finite = raw[np.isfinite(raw)]
                if finite.size == 0:
                    self._issue(f"depth NPY has no finite values: {npy_path}")
                elif float(finite.max() - finite.min()) < 1e-4:
                    self._warn(f"depth NPY almost constant: {npy_path}")
            except Exception as exc:
                self._issue(f"cannot read depth NPY: {npy_path}: {exc}")

    def _check_coco_file(self, path: Path, split: str) -> None:
        split_report = {"path": str(path), "images": 0, "annotations": 0, "issues": []}
        self.report["coco"][split] = split_report
        if not path.exists():
            message = f"missing COCO file: {path}"
            if self.require_coco:
                self._issue(message)
            else:
                self.report["optional_missing"].setdefault("coco_files", 0)
                self.report["optional_missing"]["coco_files"] += 1
            split_report["issues"].append("missing" if self.require_coco else "optional_missing")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            msg = f"COCO parse failed: {path}: {exc}"
            self._issue(msg)
            split_report["issues"].append(msg)
            return
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        categories = payload.get("categories", [])
        split_report["images"] = len(images)
        split_report["annotations"] = len(annotations)
        if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
            self._issue(f"COCO images/annotations/categories must be lists: {path}")
            return
        category_ids = {c.get("id") for c in categories if isinstance(c, dict)}
        image_by_id = {img.get("id"): img for img in images if isinstance(img, dict)}
        for ann in annotations:
            ann_id = ann.get("id")
            if ann.get("image_id") not in image_by_id:
                self._issue(f"COCO annotation references missing image: {path} ann={ann_id}")
            if ann.get("category_id") not in category_ids:
                self._issue(f"COCO annotation invalid category_id: {path} ann={ann_id}")
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                self._issue(f"COCO annotation invalid bbox: {path} ann={ann_id}")
            segmentation = ann.get("segmentation")
            if not segmentation:
                self._issue(f"COCO annotation empty segmentation: {path} ann={ann_id}")
            elif isinstance(segmentation, list):
                if not any(isinstance(poly, list) and len(poly) >= 6 for poly in segmentation):
                    self._issue(f"COCO polygon segmentation invalid: {path} ann={ann_id}")
            elif isinstance(segmentation, dict):
                if "counts" not in segmentation or "size" not in segmentation:
                    self._issue(f"COCO RLE segmentation invalid: {path} ann={ann_id}")
            else:
                self._issue(f"COCO segmentation invalid type: {path} ann={ann_id}")

    def validate_split(self, split: str) -> None:
        collected = self._collect(split)
        self._remove_orphans(split, collected)
        image_stems = set(collected["images"].keys())
        split_report: dict[str, Any] = {
            "counts": {key: len(value) for key, value in collected.items()},
            "missing": {},
            "warnings": [],
        }
        self.report["splits"][split] = split_report

        required_groups = ["masks", "labels_yolo", "metadata", "depth_png"]
        if self.require_depth_npy:
            required_groups.append("depth_npy")
        for group in required_groups:
            missing = sorted(image_stems - set(collected[group].keys()))
            split_report["missing"][group] = missing
            for stem in missing:
                if group == "depth_png" and not self.require_depth:
                    self.report["optional_missing"].setdefault("depth_png", 0)
                    self.report["optional_missing"]["depth_png"] += 1
                elif group == "depth_npy" and not self.require_depth_npy:
                    self.report["optional_missing"].setdefault("depth_npy", 0)
                    self.report["optional_missing"]["depth_npy"] += 1
                else:
                    self._issue(f"missing {group} for {split}/{stem}")

        for stem in sorted(image_stems):
            rgb_path = collected["images"][stem]
            rgb_size = self._size(rgb_path)
            if rgb_size is None:
                self._issue(f"cannot read RGB: {rgb_path}")
                continue
            width, height = rgb_size
            yolo_path = collected["labels_yolo"].get(stem, self.root / "labels_yolo" / split / f"{stem}.txt")
            meta_path = collected["metadata"].get(stem, self.root / "metadata" / split / f"{stem}.json")
            mask_path = collected["masks"].get(stem, self.root / "masks" / split / f"{stem}_mask.png")
            depth_png = collected["depth_png"].get(stem, self.root / "depth" / split / f"{stem}_depth.png")
            depth_npy = collected["depth_npy"].get(stem)

            yolo_boxes = self._check_yolo(yolo_path, width, height)
            meta_boxes = self._check_metadata(meta_path, width, height)
            self._check_mask(mask_path, width, height, meta_path, meta_boxes)
            self._check_depth(depth_png, depth_npy, width, height)

            if meta_boxes and len(yolo_boxes) != len(meta_boxes):
                self._issue(f"YOLO/metadata annotation count mismatch for {split}/{stem}: yolo={len(yolo_boxes)} metadata={len(meta_boxes)}")
            elif meta_boxes:
                self._check_yolo_metadata_alignment(split, stem, yolo_boxes, meta_boxes)

    def _check_yolo_metadata_alignment(
        self,
        split: str,
        stem: str,
        yolo_boxes: list[dict[str, Any]],
        meta_boxes: list[dict[str, Any]],
    ) -> None:
        """Verify YOLO labels are the same boxes/classes exported in metadata."""
        used: set[int] = set()
        for meta_idx, meta in enumerate(meta_boxes):
            meta_cls = int(meta.get("class_id", -1))
            meta_bbox = meta.get("bbox_xyxy")
            if not meta_bbox or len(meta_bbox) != 4:
                continue
            best_idx = -1
            best_iou = -1.0
            for yolo_idx, yolo in enumerate(yolo_boxes):
                if yolo_idx in used or int(yolo.get("class_id", -1)) != meta_cls:
                    continue
                iou = self._bbox_iou(meta_bbox, yolo["bbox_xyxy"])
                if iou > best_iou:
                    best_idx = yolo_idx
                    best_iou = iou
            if best_idx < 0:
                self._issue(f"YOLO/metadata class mismatch for {split}/{stem}: metadata box {meta_idx} class={meta_cls}")
                continue
            used.add(best_idx)
            if best_iou < 0.995:
                self._issue(f"YOLO/metadata bbox mismatch for {split}/{stem}: metadata box {meta_idx} iou={best_iou:.4f}")

    @staticmethod
    def _bbox_iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 1e-9 else 0.0

    def validate(self) -> dict[str, Any]:
        """Chạy toàn bộ kiểm tra và trả report JSON-friendly."""
        self.cleanup_junk()
        for split in EXPECTED_SPLITS:
            self.validate_split(split)
        for split in EXPECTED_SPLITS:
            self._check_coco_file(self.root / "annotations_coco" / f"{split}.json", split)
        self.report["issues"] = self.issues
        self.report["warnings"] = self.warnings
        self.report["summary"] = {
            "total_issues": len(self.issues),
            "total_warnings": len(self.warnings),
            "status": "ok" if not self.issues else "needs_attention",
        }
        return self.report


def validate_dataset(
    root: Path,
    fix: bool = False,
    require_depth: bool = True,
    require_depth_npy: bool = False,
    require_coco: bool = True,
    require_mask_pixels: bool = True,
) -> dict[str, Any]:
    """API gọn để các module khác chạy checker mà không cần tạo class trực tiếp."""
    return DatasetChecker(
        root,
        fix=fix,
        require_depth=require_depth,
        require_depth_npy=require_depth_npy,
        require_coco=require_coco,
        require_mask_pixels=require_mask_pixels,
    ).validate()


def main() -> int:
    """CLI nhỏ để kiểm tra dataset đã sinh ngoài pipeline chính."""
    parser = argparse.ArgumentParser(description="Check and optionally clean BTL2 dataset consistency")
    parser.add_argument("dataset_root", type=Path, help="Path to a generated BTL2 dataset")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON report path")
    parser.add_argument("--fix", action="store_true", help="Remove junk files and orphan artifacts")
    parser.add_argument("--no-require-depth", action="store_true", help="Warn instead of failing when depth is missing")
    parser.add_argument("--require-depth-npy", action="store_true", help="Fail if depth numeric .npy files are missing")
    parser.add_argument("--no-require-coco", action="store_true", help="Warn instead of failing when COCO JSON is missing")
    parser.add_argument("--no-require-mask-pixels", action="store_true", help="Warn instead of failing when metadata mask colors do not appear in masks")
    args = parser.parse_args()

    report = validate_dataset(
        args.dataset_root,
        fix=args.fix,
        require_depth=not args.no_require_depth,
        require_depth_npy=args.require_depth_npy,
        require_coco=not args.no_require_coco,
        require_mask_pixels=not args.no_require_mask_pixels,
    )
    output = args.output or args.dataset_root / "quality_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset: {args.dataset_root}")
    print(f"Status: {report['summary']['status']}")
    print(f"Issues: {report['summary']['total_issues']}")
    print(f"Warnings: {report['summary']['total_warnings']}")
    if report.get("fixed"):
        print(f"Fixed items: {len(report['fixed'])}")
    print(f"Report: {output}")
    return 0 if report["summary"]["total_issues"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
