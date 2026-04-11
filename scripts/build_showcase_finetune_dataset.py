"""Build a showcase-style YOLO dataset by recoloring and augmenting an existing showcase export."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btl2.annotations.validators import _iter_image_files, _resolve_mask_path
from btl2.utils.colors import class_color


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a recolored showcase fine-tune dataset.")
    parser.add_argument("--source", default="outputs/btl2/showcase_dataset", help="Source showcase dataset root.")
    parser.add_argument("--output", default="outputs/btl2/showcase_finetune_dataset", help="Output dataset root.")
    parser.add_argument("--train-variants", type=int, default=12, help="Number of train variants per source frame.")
    parser.add_argument("--val-variants", type=int, default=1, help="Number of val variants per source frame.")
    parser.add_argument("--refresh-source-rgb", action="store_true", help="Overwrite source RGB images with recolored versions.")
    return parser.parse_args()


def _resolve_root(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _ensure_tree(root: Path) -> None:
    for rel in (
        "images/train",
        "images/val",
        "labels_yolo/train",
        "labels_yolo/val",
        "depth/train",
        "depth/val",
        "masks/train",
        "masks/val",
        "metadata/train",
        "metadata/val",
        "previews",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _stylize_with_mask(rgb: np.ndarray, mask_rgb: np.ndarray, metadata: dict) -> np.ndarray:
    styled = rgb.copy()
    segmentation_mapping = metadata.get("segmentation_mapping", {})

    for color_key, payload in segmentation_mapping.items():
        class_name = payload.get("class_name")
        if not class_name:
            continue
        try:
            color = np.array([int(v) for v in color_key.split("_")], dtype=np.uint8)
        except ValueError:
            continue
        mask = np.all(mask_rgb == color, axis=-1)
        if not np.any(mask):
            continue

        base = np.array(class_color(class_name), dtype=np.float32) / 255.0
        shading = np.mean(rgb[mask], axis=1, keepdims=True)
        tinted = np.clip(base[None, :] * (0.32 + 0.95 * shading), 0.0, 1.0)
        styled[mask] = np.clip(0.14 * rgb[mask] + 0.86 * tinted, 0.0, 1.0)

    return styled


def _apply_variant(rgb_uint8: np.ndarray, seed: int, variant_idx: int) -> Image.Image:
    rng = random.Random(seed + variant_idx * 991)
    image = Image.fromarray(rgb_uint8, mode="RGB")
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.18))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.95, 1.18))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.88, 1.18))
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.7)))
    return image


def _write_metadata_clone(source_metadata: dict, target_path: Path, stem: str, image_rel: str, mask_rel: str, yolo_rel: str) -> None:
    payload = json.loads(json.dumps(source_metadata))
    payload["frame_id"] = stem
    paths = payload.setdefault("paths", {})
    paths["rgb"] = image_rel
    paths["mask"] = mask_rel
    paths["yolo"] = yolo_rel
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_label(source_label: Path, target_label: Path) -> None:
    target_label.write_text(source_label.read_text(encoding="utf-8"), encoding="utf-8")


def _copy_mask(source_mask: Path, target_mask: Path) -> None:
    if target_mask.exists() or target_mask.is_symlink():
        target_mask.unlink()
    shutil.copy2(source_mask, target_mask)


def _copy_optional_file(source_path: Path, target_path: Path) -> bool:
    if not source_path.exists():
        return False
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return True


def _write_dataset_yaml(output_root: Path, names: dict) -> None:
    lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for idx, name in names.items():
        lines.append(f"  {idx}: {name}")
    (output_root / "dataset.yaml").write_text("\n".join(lines), encoding="utf-8")


def _variant_stem(stem: str, variant_idx: int) -> str:
    return f"{stem}__v{variant_idx:02d}"


def _stable_seed(text: str) -> int:
    value = 0
    for idx, char in enumerate(text):
        value = (value + (idx + 1) * ord(char)) & 0xFFFFFFFF
    return value


def build_dataset(source_root: Path, output_root: Path, train_variants: int, val_variants: int, refresh_source_rgb: bool) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    _ensure_tree(output_root)

    source_yaml = yaml.safe_load((source_root / "dataset.yaml").read_text(encoding="utf-8"))
    names = source_yaml.get("names", {})
    _write_dataset_yaml(output_root, names)

    for split in ("train", "val"):
        variant_count = train_variants if split == "train" else val_variants
        image_paths = _iter_image_files(source_root / "images" / split)
        for image_path in image_paths:
            stem = image_path.stem
            mask_path = _resolve_mask_path(source_root / "masks" / split, stem)
            metadata_path = source_root / "metadata" / split / f"{stem}.json"
            label_path = source_root / "labels_yolo" / split / f"{stem}.txt"
            depth_png_path = source_root / "depth" / split / f"{stem}_depth.png"
            depth_npy_path = source_root / "depth" / split / f"{stem}_depth.npy"
            if mask_path is None or not metadata_path.exists() or not label_path.exists():
                continue

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            styled_rgb = _stylize_with_mask(_load_rgb(image_path), _load_mask(mask_path), metadata)
            styled_uint8 = np.clip(styled_rgb * 255.0, 0.0, 255.0).astype(np.uint8)

            if refresh_source_rgb:
                Image.fromarray(styled_uint8, mode="RGB").save(image_path)

            for variant_idx in range(variant_count):
                new_stem = _variant_stem(stem, variant_idx)
                target_image = output_root / "images" / split / f"{new_stem}.png"
                target_mask = output_root / "masks" / split / f"{new_stem}_mask.png"
                target_depth_png = output_root / "depth" / split / f"{new_stem}_depth.png"
                target_depth_npy = output_root / "depth" / split / f"{new_stem}_depth.npy"
                target_label = output_root / "labels_yolo" / split / f"{new_stem}.txt"
                target_metadata = output_root / "metadata" / split / f"{new_stem}.json"

                variant_image = _apply_variant(styled_uint8, seed=_stable_seed(stem), variant_idx=variant_idx)
                variant_image.save(target_image)
                _copy_mask(mask_path, target_mask)
                _copy_optional_file(depth_png_path, target_depth_png)
                _copy_optional_file(depth_npy_path, target_depth_npy)
                _copy_label(label_path, target_label)
                _write_metadata_clone(
                    metadata,
                    target_metadata,
                    new_stem,
                    f"images/{split}/{new_stem}.png",
                    f"masks/{split}/{new_stem}_mask.png",
                    f"labels_yolo/{split}/{new_stem}.txt",
                )


def main() -> int:
    args = _parse_args()
    source_root = _resolve_root(args.source)
    output_root = _resolve_root(args.output)
    build_dataset(source_root, output_root, args.train_variants, args.val_variants, args.refresh_source_rgb)
    print(f"Built showcase fine-tune dataset: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
