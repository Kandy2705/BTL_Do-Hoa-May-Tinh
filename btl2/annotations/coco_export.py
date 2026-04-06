"""COCO-format JSON export with mask-path linkage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from btl2.utils.constants import CLASS_NAMES
from btl2.utils.io import write_json


@dataclass
class CocoExporter:
    """Collect image and bbox annotations across the whole dataset."""

    next_image_id: int = 1
    next_annotation_id: int = 1
    images_by_split: dict[str, list[dict]] = field(default_factory=lambda: {"train": [], "val": []})
    annotations_by_split: dict[str, list[dict]] = field(default_factory=lambda: {"train": [], "val": []})
    categories: dict[int, dict] = field(
        default_factory=lambda: {
            idx: {"id": idx, "name": name, "supercategory": "road_scene"} for idx, name in enumerate(CLASS_NAMES)
        }
    )

    def add_frame(self, frame_id: str, split: str, width: int, height: int, bboxes: list[dict], paths: dict[str, str]) -> None:
        """Append one rendered frame and its object annotations."""
        image_id = self.next_image_id
        self.next_image_id += 1
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
            self.annotations_by_split[split].append(
                {
                    "id": self.next_annotation_id,
                    "image_id": image_id,
                    "category_id": bbox["class_id"],
                    "bbox": [round(v, 3) for v in bbox["bbox_xywh"]],
                    "area": round(bbox["bbox_xywh"][2] * bbox["bbox_xywh"][3], 3),
                    "iscrowd": 0,
                    "segmentation": [],
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
                    "note": "Segmentation polygons are omitted; use mask PNG linkage in images[].mask_file.",
                },
            }
            write_json(output_path / f"{split}.json", payload)
