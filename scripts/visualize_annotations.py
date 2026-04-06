"""Draw sample bounding boxes and side-by-side previews for quick inspection."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def draw_boxes(image_path: Path, metadata_path: Path, output_path: Path) -> None:
    """Overlay bounding boxes and class names onto one RGB image."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for bbox in metadata.get("bounding_boxes", []):
        x, y, w, h = bbox["bbox_xywh"]
        draw.rectangle((x, y, x + w, y + h), outline=(255, 255, 0), width=2)
        draw.text((x + 4, y + 4), bbox["class_name"], fill=(255, 255, 0))
    image.save(output_path)


def build_contact_sheet(dataset_root: Path, output_path: Path, max_items: int = 6) -> None:
    """Create a simple collage of RGB, depth, and mask samples."""
    rgb_paths = sorted((dataset_root / "images" / "train").glob("*.png"))[:max_items]
    rows = []
    for rgb_path in rgb_paths:
        frame_id = rgb_path.stem
        depth_path = dataset_root / "depth" / "train" / f"{frame_id}_depth.png"
        mask_path = dataset_root / "masks" / "train" / f"{frame_id}_mask.png"
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        row = Image.new("RGB", (rgb.width * 3, rgb.height))
        row.paste(rgb, (0, 0))
        row.paste(depth, (rgb.width, 0))
        row.paste(mask, (rgb.width * 2, 0))
        rows.append(row)

    if not rows:
        raise FileNotFoundError("No generated frames found to build a contact sheet.")

    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)))
    for row_idx, row in enumerate(rows):
        canvas.paste(row, (0, row_idx * row.height))
    canvas.save(output_path)


def main() -> int:
    """Create overlay previews from the demo dataset output."""
    dataset_root = ROOT / "outputs" / "btl2" / "demo_dataset"
    previews_dir = dataset_root / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths = sorted((dataset_root / "images" / "train").glob("*.png"))
    for rgb_path in rgb_paths[: min(5, len(rgb_paths))]:
        frame_id = rgb_path.stem
        metadata_path = dataset_root / "metadata" / "train" / f"{frame_id}.json"
        output_path = previews_dir / f"{frame_id}_boxes.png"
        draw_boxes(rgb_path, metadata_path, output_path)

    build_contact_sheet(dataset_root, previews_dir / "contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
