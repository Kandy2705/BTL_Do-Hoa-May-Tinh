"""COCO-format JSON export with polygon segmentation from masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from btl2.utils.constants import CLASS_NAMES
from btl2.utils.io import write_json
from btl2.utils.colors import class_color, instance_color


@dataclass
class CocoExporter:
    """Collect image and bbox annotations across the whole dataset."""

    segmentation_mode: str = "polygon"
    next_image_id: int = 1
    next_annotation_id: int = 1
    images_by_split: dict[str, list[dict]] = field(default_factory=lambda: {"train": [], "val": []})
    annotations_by_split: dict[str, list[dict]] = field(default_factory=lambda: {"train": [], "val": []})
    categories: dict[int, dict] = field(
        default_factory=lambda: {
            idx: {"id": idx, "name": name, "supercategory": "road_scene"} for idx, name in enumerate(CLASS_NAMES)
        }
    )

    @staticmethod
    def _annotation_color(class_id: int, instance_id: int) -> tuple[int, int, int]:
        """Resolve the encoded RGB used by the segmentation pass for one object."""
        class_name = CLASS_NAMES[int(class_id)]
        if class_name == "road":
            return class_color("road")
        return instance_color(instance_id)

    @staticmethod
    def _extract_polygons(mask_rgb: np.ndarray, color: tuple[int, int, int]) -> tuple[list[list[float]], list[float], float]:
        """Convert one instance/class color region into COCO polygons, bbox, and area."""
        binary = np.all(mask_rgb == np.asarray(color, dtype=np.uint8), axis=2).astype(np.uint8)
        pixel_area = float(binary.sum())
        if pixel_area <= 0.0:
            return [], [], 0.0

        ys, xs = np.nonzero(binary)
        x_min = float(xs.min())
        y_min = float(ys.min())
        x_max = float(xs.max())
        y_max = float(ys.max())
        bbox_xywh = [x_min, y_min, float(x_max - x_min + 1), float(y_max - y_min + 1)]

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            epsilon = max(0.75, 0.002 * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if approx.shape[0] < 3:
                approx = contour
            points = approx.reshape(-1, 2).astype(np.float32)
            polygon = points.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append([round(float(v), 2) for v in polygon])

        if not polygons:
            polygon = [
                round(x_min, 2), round(y_min, 2),
                round(x_max + 1.0, 2), round(y_min, 2),
                round(x_max + 1.0, 2), round(y_max + 1.0, 2),
                round(x_min, 2), round(y_max + 1.0, 2),
            ]
            polygons = [polygon]

        return polygons, [round(v, 3) for v in bbox_xywh], round(pixel_area, 3)

    @staticmethod
    def _encode_rle(binary_mask: np.ndarray) -> dict[str, list[int] | list[int]]:
        """Encode a binary mask into COCO-style uncompressed RLE."""
        pixels = binary_mask.astype(np.uint8).T.flatten()
        counts: list[int] = []
        count = 0
        prev = 0
        for pixel in pixels:
            value = int(pixel)
            if value == prev:
                count += 1
            else:
                counts.append(count)
                count = 1
                prev = value
        counts.append(count)
        return {"counts": counts, "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])]}

    def add_frame(self, frame_id: str, split: str, width: int, height: int, bboxes: list[dict], paths: dict[str, str]) -> None:
        """Append one rendered frame and its object annotations."""
        image_id = self.next_image_id
        self.next_image_id += 1
        mask_rgb = np.asarray(Image.open(paths["mask"]).convert("RGB"), dtype=np.uint8)
        self.images_by_split[split].append(
            {
                "id": image_id,
                "file_name": Path(paths["rgb"]).name,
                "width": width,
                "height": height,
                "mask_file": Path(paths["mask"]).name,
                "metadata_file": Path(paths["metadata"]).name,
            }
        )

        for bbox in bboxes:
            color = self._annotation_color(bbox["class_id"], bbox["instance_id"])
            binary_mask = np.all(mask_rgb == np.asarray(color, dtype=np.uint8), axis=2).astype(np.uint8)
            polygons, coco_bbox, area = self._extract_polygons(mask_rgb, color)
            if not coco_bbox:
                coco_bbox = [round(v, 3) for v in bbox["bbox_xywh"]]
                area = round(coco_bbox[2] * coco_bbox[3], 3)
            segmentation: list[list[float]] | dict[str, list[int] | list[int]]
            if self.segmentation_mode == "rle":
                segmentation = self._encode_rle(binary_mask)
            else:
                segmentation = polygons
            self.annotations_by_split[split].append(
                {
                    "id": self.next_annotation_id,
                    "image_id": image_id,
                    "category_id": bbox["class_id"],
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                    "attributes": {
                        "instance_id": bbox["instance_id"],
                        "occlusion_ratio": bbox.get("occlusion_ratio", 0.0),
                        "visibility_ratio": bbox.get("visibility_ratio", 0.0),
                    },
                }
            )
            self.next_annotation_id += 1

    def write(self, output_dir: str | Path) -> None:
        """Write train and val COCO JSON files."""
        output_path = Path(output_dir)
        for split in ("train", "val"):
            payload = {
                "images": self.images_by_split[split],
                "annotations": self.annotations_by_split[split],
                "categories": list(self.categories.values()),
                "info": {
                    "description": "Synthetic road-scene dataset",
                    "note": f"COCO segmentation is exported as {self.segmentation_mode} from the exported instance/class mask PNG.",
                },
            }
            write_json(output_path / f"{split}.json", payload)
