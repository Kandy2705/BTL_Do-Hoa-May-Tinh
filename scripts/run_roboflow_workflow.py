#!/usr/bin/env python3
"""Run a Roboflow Workflow on one image or a folder and export CSV/JSON.

Example:
    ROBOFLOW_API_KEY=... \
    ROBOFLOW_WORKSPACE=ngs-workspace-kgefe \
    ROBOFLOW_WORKFLOW_ID=road-scene-detection-pipeline-1776617921973 \
    python scripts/run_roboflow_workflow.py --images outputs/btl2/showcase_dataset/images/train
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/btl_dhmt_matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btl2.inference.roboflow_workflow import (  # noqa: E402
    append_csv_rows,
    draw_predictions,
    extract_predictions,
    merge_predictions,
    run_workflow,
    save_first_annotated_image,
    save_json,
    translate_predictions,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def mask_secret(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "MISSING"
    if len(value) <= 10:
        return value[:3] + "..."
    return f"{value[:6]}...{value[-4:]}"


def iter_images(path: Path):
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
        return
    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def roboflow_output_size(payload):
    if isinstance(payload, dict):
        image = payload.get("image")
        if isinstance(image, dict) and "width" in image and "height" in image:
            return int(image["width"]), int(image["height"])
        for value in payload.values():
            found = roboflow_output_size(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = roboflow_output_size(item)
            if found is not None:
                return found
    return None


def edge_crop_boxes(width: int, height: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    crop_width = min(width, max(360, int(width * 0.32)))
    return [
        ("left_edge", (0, 0, crop_width, height)),
        ("right_edge", (width - crop_width, 0, width, height)),
    ]


def save_crop(image_path: Path, crop_box: tuple[int, int, int, int], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image.crop(crop_box).save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Image file or folder to process.")
    parser.add_argument("--output-dir", default="outputs/inference/roboflow", help="Where JSON/CSV outputs are written.")
    parser.add_argument("--preview-dir", default="outputs/inference/roboflow/previews", help="Where boxed preview images are written.")
    parser.add_argument("--csv", default="roboflow_detections.csv", help="CSV filename inside output-dir.")
    parser.add_argument("--api-url", default=os.environ.get("ROBOFLOW_API_URL", "https://detect.roboflow.com"))
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--workspace", default=os.environ.get("ROBOFLOW_WORKSPACE", ""))
    parser.add_argument("--workflow-id", default=os.environ.get("ROBOFLOW_WORKFLOW_ID", ""))
    parser.add_argument("--workflow-version-id", default=os.environ.get("ROBOFLOW_WORKFLOW_VERSION_ID", ""))
    parser.add_argument("--min-conf", type=float, default=0.0, help="Filter predictions below this confidence.")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Allow Roboflow API cache. By default the script asks for fresh results.",
    )
    parser.add_argument(
        "--edge-crops",
        action="store_true",
        help="Also run left/right edge crops, useful for small traffic_light/traffic_sign objects.",
    )
    parser.add_argument("--no-preview", action="store_true", help="Skip drawing preview images.")
    parser.add_argument("--quiet-config", action="store_true", help="Do not print masked Roboflow config before running.")
    return parser.parse_args()


def validate_config(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("Missing API key. Set ROBOFLOW_API_KEY or pass --api-key.")
    if args.api_key.strip() in {"rf_xxx", "YOUR_API_KEY", "MÃ_API_THẬT_CỦA_BẠN_Ở_ĐÂY"}:
        raise ValueError("ROBOFLOW_API_KEY is still a placeholder. Replace it with your real key.")
    if not args.workspace:
        raise ValueError("Missing workspace. Set ROBOFLOW_WORKSPACE or pass --workspace.")
    if not args.workflow_id:
        raise ValueError("Missing workflow id. Set ROBOFLOW_WORKFLOW_ID or pass --workflow-id.")


def print_masked_config(args: argparse.Namespace) -> None:
    print("Roboflow config:")
    print(f"  api_url:     {args.api_url}")
    print(f"  api_key:     {mask_secret(args.api_key)}")
    print(f"  workspace:   {args.workspace or 'MISSING'}")
    print(f"  workflow_id: {args.workflow_id or 'MISSING'}")
    print(f"  version_id:  {args.workflow_version_id or 'latest deployed'}")
    print(f"  min_conf:    {args.min_conf}")
    print(f"  use_cache:   {args.use_cache}")


def main() -> int:
    args = parse_args()
    image_root = Path(args.images).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    preview_dir = Path(args.preview_dir).expanduser()
    csv_path = output_dir / args.csv

    if not image_root.exists():
        raise FileNotFoundError(f"Image path not found: {image_root}")
    validate_config(args)
    if not args.quiet_config:
        print_masked_config(args)

    if csv_path.exists():
        csv_path.unlink()

    total_images = 0
    total_detections = 0
    for image_path in iter_images(image_root):
        total_images += 1
        size = image_size(image_path)
        if size is None:
            print(f"[{total_images}] Roboflow: {image_path}")
        else:
            print(f"[{total_images}] Roboflow: {image_path} input_size={size[0]}x{size[1]}")
        try:
            payload = run_workflow(
                api_url=args.api_url,
                api_key=args.api_key,
                workspace_name=args.workspace,
                workflow_id=args.workflow_id,
                image_path=image_path,
                use_cache=args.use_cache,
                workflow_version_id=args.workflow_version_id or None,
            )
        except Exception as exc:
            print("")
            print("Roboflow request failed.")
            print(f"  reason: {exc}")
            print(f"  api_key: {mask_secret(args.api_key)}")
            print(f"  workspace: {args.workspace}")
            print(f"  workflow_id: {args.workflow_id}")
            print("")
            print("Checklist:")
            print("  1. API key phai la key that, khong phai rf_xxx/placeholder.")
            print("  2. Key phai thuoc dung workspace dang chua Workflow.")
            print("  3. Workflow ID phai copy dung tu Roboflow.")
            print("  4. Neu Workflow dung custom Python/dynamic block, hay bo block do hoac deploy kieu cho phep custom code.")
            print("  5. Neu vua doi key, mo terminal moi hoac export lai bien moi truong.")
            return 2
        json_path = save_json(payload, output_dir / f"{image_path.stem}_roboflow.json")
        rows = extract_predictions(payload, image_name=image_path.name, min_confidence=args.min_conf)
        used_augmented_crops = False
        if args.edge_crops and size is not None:
            crop_dir = output_dir / "_edge_crops"
            for crop_name, crop_box in edge_crop_boxes(size[0], size[1]):
                left, top, _, _ = crop_box
                crop_path = save_crop(image_path, crop_box, crop_dir / f"{image_path.stem}_{crop_name}{image_path.suffix}")
                try:
                    crop_payload = run_workflow(
                        api_url=args.api_url,
                        api_key=args.api_key,
                        workspace_name=args.workspace,
                        workflow_id=args.workflow_id,
                        image_path=crop_path,
                        use_cache=args.use_cache,
                        workflow_version_id=args.workflow_version_id or None,
                    )
                except Exception as exc:
                    print(f"    {crop_name}=failed reason={exc}")
                    continue
                save_json(crop_payload, output_dir / f"{image_path.stem}_roboflow_{crop_name}.json")
                crop_rows = extract_predictions(
                    crop_payload,
                    image_name=image_path.name,
                    min_confidence=args.min_conf,
                )
                translated = translate_predictions(crop_rows, dx=left, dy=top, image_name=image_path.name)
                rows.extend(translated)
                used_augmented_crops = used_augmented_crops or bool(translated)
                print(f"    {crop_name}_detections={len(translated)}")
            rows = merge_predictions(rows)
        append_csv_rows(csv_path, rows)
        if not args.no_preview:
            preview_path = preview_dir / f"{image_path.stem}_roboflow_pred.jpg"
            if used_augmented_crops:
                draw_predictions(image_path, rows, preview_path)
            elif save_first_annotated_image(payload, preview_path) is None:
                draw_predictions(image_path, rows, preview_path)
        total_detections += len(rows)
        output_size = roboflow_output_size(payload)
        size_note = ""
        if output_size is not None:
            size_note = f" output_size={output_size[0]}x{output_size[1]}"
        print(f"    detections={len(rows)}{size_note} json={json_path}")

    print(f"Done: images={total_images}, detections={total_detections}, csv={csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Failed: {exc}")
        raise SystemExit(2)
