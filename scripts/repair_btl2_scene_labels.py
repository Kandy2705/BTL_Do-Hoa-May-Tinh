"""Repair BTL2 current-scene labels using per-instance masks.

This is mainly useful for datasets exported from the BTL1 scene bridge, where
class names are inferred from object names and large helper meshes such as the
road can accidentally become training labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btl2.annotations.coco_export import CocoExporter
from btl2.annotations.custom_export import CustomCsvExporter
from btl2.annotations.yolo_export import export_yolo_labels, write_dataset_yaml
from btl2.utils.constants import CLASS_NAMES, CLASS_TO_ID


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair BTL2 labels from masks and metadata.")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="outputs/btl2/showcase_dataset",
        help="Dataset root containing images/, masks/, metadata/, and labels_yolo/.",
    )
    return parser.parse_args()


def _looks_like_road(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("road", "street", "lane", "ground", "floor", "terrain"))


def _infer_class_name(name: str, fallback: str | None = None) -> str | None:
    lowered = name.lower()
    if _looks_like_road(lowered):
        return None
    if any(token in lowered for token in ("ped", "human", "person", "walker")):
        return "person"
    if any(token in lowered for token in ("motorbike", "motorcycle", "moto", "bike", "scooter")):
        return "motorbike"
    if "bus" in lowered:
        return "bus"
    if any(token in lowered for token in ("truck", "lorry")):
        return "truck"
    if "sign" in lowered:
        return "traffic_sign"
    if any(token in lowered for token in ("light", "signal")):
        return "traffic_light"
    if fallback in CLASS_TO_ID:
        return fallback
    return "car"


def _mask_path(dataset_root: Path, split: str, stem: str) -> Path:
    candidates = (
        dataset_root / "masks" / split / f"{stem}_mask.png",
        dataset_root / "masks" / split / f"{stem}_layer.png",
        dataset_root / "masks" / split / f"{stem}.png",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Mask not found for {split}/{stem}")


def _color_from_key(key: str) -> tuple[int, int, int]:
    r, g, b = key.split("_")
    return int(r), int(g), int(b)


def _bbox_from_mask(mask_rgb: np.ndarray, color: tuple[int, int, int]) -> tuple[list[float], list[float], float] | None:
    visible = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=2)
    if not np.any(visible):
        return None

    ys, xs = np.nonzero(visible)
    x_min = float(xs.min())
    y_min = float(ys.min())
    x_max = float(xs.max() + 1)
    y_max = float(ys.max() + 1)
    width = x_max - x_min
    height = y_max - y_min
    image_height, image_width = mask_rgb.shape[:2]
    visibility_ratio = float((width * height) / max(float(image_width * image_height), 1.0))
    return [x_min, y_min, x_max, y_max], [x_min, y_min, width, height], visibility_ratio


def _render_boxes_preview(image_path: Path, bboxes: list[dict], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x, y, w, h = bbox["bbox_xywh"]
        draw.rectangle((x, y, x + w, y + h), outline=(255, 255, 0), width=2)
        draw.text((x + 4, y + 4), bbox["class_name"], fill=(255, 255, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _repair_frame(
    dataset_root: Path,
    split: str,
    image_path: Path,
) -> tuple[str, int, int, list[dict], dict[str, str]]:
    stem = image_path.stem
    metadata_path = dataset_root / "metadata" / split / f"{stem}.json"
    mask_path = _mask_path(dataset_root, split, stem)
    yolo_path = dataset_root / "labels_yolo" / split / f"{stem}.txt"
    preview_path = dataset_root / "previews" / f"{stem}_boxes.png"
    depth_png_path = dataset_root / "depth" / split / f"{stem}_depth.png"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    objects = metadata.get("objects", [])
    objects_by_instance = {int(obj.get("instance_id", -1)): obj for obj in objects}
    old_bboxes = {int(box["instance_id"]): box for box in metadata.get("bounding_boxes", [])}

    mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
    height, width = mask_rgb.shape[:2]

    repaired_boxes: list[dict] = []
    repaired_mapping: dict[str, dict] = {}
    visible_instances: set[int] = set()

    for color_key, info in metadata.get("segmentation_mapping", {}).items():
        instance_id = int(info.get("instance_id", -1))
        obj = objects_by_instance.get(instance_id, {})
        object_name = str(obj.get("name") or f"instance_{instance_id}")
        fallback = obj.get("class_name") or info.get("class_name")

        class_name = _infer_class_name(object_name, fallback)
        if class_name is None:
            repaired_mapping[color_key] = {
                "instance_id": instance_id,
                "class_name": "road",
                "class_id": 255,
            }
            if obj:
                obj["class_name"] = "road"
                obj["class_id"] = 255
                obj["bbox_pixels"] = None
            continue

        class_id = CLASS_TO_ID[class_name]
        bbox_info = _bbox_from_mask(mask_rgb, _color_from_key(color_key))
        repaired_mapping[color_key] = {
            "instance_id": instance_id,
            "class_name": class_name,
            "class_id": class_id,
        }

        if obj:
            obj["class_name"] = class_name
            obj["class_id"] = class_id

        if bbox_info is None:
            if obj:
                obj["bbox_pixels"] = None
            continue

        bbox_xyxy, bbox_xywh, visibility_ratio = bbox_info
        visible_instances.add(instance_id)
        repaired_boxes.append(
            {
                "instance_id": instance_id,
                "class_name": class_name,
                "class_id": class_id,
                "bbox_xyxy": bbox_xyxy,
                "bbox_xywh": bbox_xywh,
                "visibility_ratio": visibility_ratio,
                "occlusion_ratio": float(old_bboxes.get(instance_id, {}).get("occlusion_ratio", 0.0)),
            }
        )
        if obj:
            obj["bbox_pixels"] = bbox_xywh

    for obj in objects:
        instance_id = int(obj.get("instance_id", -1))
        if instance_id in visible_instances:
            continue
        object_name = str(obj.get("name") or f"instance_{instance_id}")
        fallback = obj.get("class_name")
        class_name = _infer_class_name(object_name, fallback)
        if class_name is None:
            obj["class_name"] = "road"
            obj["class_id"] = 255
        elif class_name in CLASS_TO_ID:
            obj["class_name"] = class_name
            obj["class_id"] = CLASS_TO_ID[class_name]
        obj["bbox_pixels"] = None

    repaired_boxes.sort(key=lambda item: item["instance_id"])
    metadata["objects"] = objects
    metadata["bounding_boxes"] = repaired_boxes
    metadata["segmentation_mapping"] = repaired_mapping
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    yolo_lines = export_yolo_labels(repaired_boxes, width, height)
    yolo_path.parent.mkdir(parents=True, exist_ok=True)
    yolo_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    _render_boxes_preview(image_path, repaired_boxes, preview_path)

    paths = {
        "rgb": str(image_path),
        "depth_png": str(depth_png_path),
        "mask": str(mask_path),
        "metadata": str(metadata_path),
        "yolo": str(yolo_path),
    }
    return stem, width, height, repaired_boxes, paths


def main() -> int:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    coco = CocoExporter()
    custom = CustomCsvExporter()

    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        images = sorted(image_dir.glob("*.png"))
        for image_path in images:
            frame_id, width, height, bboxes, paths = _repair_frame(dataset_root, split, image_path)
            coco.add_frame(frame_id, split, width, height, bboxes, paths)
            custom.add_frame(frame_id, split, bboxes, paths)

    coco.write(dataset_root / "annotations_coco")
    custom.write(dataset_root / "annotations_custom")
    write_dataset_yaml(dataset_root, [{"id": idx, "name": name} for idx, name in enumerate(CLASS_NAMES)])

    print(f"Repaired dataset: {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
