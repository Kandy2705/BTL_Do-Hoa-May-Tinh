#!/usr/bin/env python3
"""Render ground-truth bounding-box overlays from BTL2 metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    "person": (52, 211, 153),
    "car": (239, 68, 68),
    "bus": (251, 146, 60),
    "truck": (168, 85, 247),
    "motorbike": (14, 165, 233),
    "traffic_light": (244, 63, 94),
    "traffic_sign": (250, 204, 21),
    "road": (120, 120, 120),
}


def _metadata_boxes(data: dict, width: int, height: int) -> list[dict]:
    """Return boxes from native BTL2 metadata or Unity-style object metadata."""
    boxes = data.get("bounding_boxes", [])
    if boxes:
        return boxes

    converted = []
    for obj in data.get("objects", []):
        bbox_yolo = obj.get("bbox_yolo")
        if not bbox_yolo or len(bbox_yolo) != 4:
            continue
        xc, yc, bw, bh = [float(v) for v in bbox_yolo]
        converted.append(
            {
                "class_name": obj.get("class_name", "unknown"),
                "bbox_xyxy": [
                    (xc - bw * 0.5) * width,
                    (yc - bh * 0.5) * height,
                    (xc + bw * 0.5) * width,
                    (yc + bh * 0.5) * height,
                ],
            }
        )
    return converted


def draw_boxes(rgb_path: Path, metadata_path: Path, output_path: Path) -> None:
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    font = ImageFont.load_default()

    for box in _metadata_boxes(data, image.width, image.height):
        cls = str(box.get("class_name", "unknown"))
        bbox = box.get("bbox_xyxy") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        color = PALETTE.get(cls, (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"GT {cls}"
        text_box = draw.textbbox((x1, y1), label, font=font)
        tx1, ty1, tx2, ty2 = text_box
        draw.rectangle([tx1, max(0, ty1 - 2), tx2 + 4, ty2 + 2], fill=color)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BTL2 GT box overlays")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Max overlays per split, 0 means all")
    args = parser.parse_args()

    total = 0
    for split in ("train", "val"):
        meta_dir = args.dataset_root / "metadata" / split
        image_dir = args.dataset_root / "images" / split
        out_dir = args.dataset_root / "previews" / "gt_overlays" / split
        if not meta_dir.exists():
            continue
        files = sorted(meta_dir.glob("*.json"))
        if args.limit > 0:
            files = files[: args.limit]
        for meta_path in files:
            stem = meta_path.stem
            rgb_path = image_dir / f"{stem}.png"
            if not rgb_path.exists():
                continue
            draw_boxes(rgb_path, meta_path, out_dir / f"{stem}_gt_overlay.png")
            total += 1

    print(f"Exported {total} GT overlay image(s) to {args.dataset_root / 'previews' / 'gt_overlays'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
