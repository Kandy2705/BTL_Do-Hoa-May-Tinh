"""Train a YOLO detector on a dataset inside this repo."""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector using a dataset.yaml or dataset root.",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=["outputs/btl2/unity_dataset"],
        help="One or more dataset roots or direct paths to dataset.yaml files.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model checkpoint to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=512, help="Square training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", default=None, help="Training device, for example cpu, 0, or mps.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--project",
        default="outputs/training/yolo",
        help="Folder used by Ultralytics to store runs.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Experiment name. Defaults to <dataset_folder>_<model_name>.",
    )
    parser.add_argument("--cache", action="store_true", help="Cache images in memory for faster training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing the same output folder.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings and exit.")
    return parser.parse_args()


def _resolve_input_path(raw_path: str) -> tuple[Path, Path]:
    input_path = Path(raw_path).expanduser()
    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()

    if input_path.is_dir():
        dataset_root = input_path
        dataset_yaml = dataset_root / "dataset.yaml"
    else:
        dataset_yaml = input_path
        dataset_root = dataset_yaml.parent

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")
    return dataset_root, dataset_yaml


def _resolve_input_paths(raw_paths: list[str]) -> list[tuple[Path, Path]]:
    return [_resolve_input_path(raw_path) for raw_path in raw_paths]


def _source_labels_root(dataset_root: Path) -> Path:
    labels_root = dataset_root / "labels"
    if labels_root.exists():
        return labels_root
    labels_yolo_root = dataset_root / "labels_yolo"
    if labels_yolo_root.exists():
        return labels_yolo_root
    raise FileNotFoundError(f"No labels directory found under {dataset_root} (expected labels or labels_yolo).")


def _image_files(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(sorted(image_dir.glob(pattern)))
    return sorted(files)


def _choose_device(device: str | None) -> str | None:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def _symlink_pair(image_path: Path, label_path: Path, images_out: Path, labels_out: Path) -> None:
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    image_target = images_out / image_path.name
    label_target = labels_out / label_path.name
    if not image_target.exists():
        image_target.symlink_to(image_path.resolve())
    if not label_target.exists():
        label_target.symlink_to(label_path.resolve())


def _symlink_prefixed_pair(
    image_path: Path,
    label_path: Path,
    images_out: Path,
    labels_out: Path,
    prefix: str,
) -> None:
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    image_target = images_out / f"{prefix}__{image_path.name}"
    label_target = labels_out / f"{prefix}__{label_path.name}"
    if not image_target.exists():
        image_target.symlink_to(image_path.resolve())
    if not label_target.exists():
        label_target.symlink_to(label_path.resolve())


def _prepare_runtime_dataset(dataset_root: Path) -> Path:
    source_images_root = dataset_root / "images"
    source_labels_root = _source_labels_root(dataset_root)
    source_train_images = _image_files(source_images_root / "train")
    source_val_images = _image_files(source_images_root / "val")

    if not source_train_images:
        raise FileNotFoundError(f"No training images found under {source_images_root / 'train'}")

    runtime_root = Path(tempfile.mkdtemp(prefix=f"{dataset_root.name}_runtime_"))
    runtime_images_train = runtime_root / "images" / "train"
    runtime_images_val = runtime_root / "images" / "val"
    runtime_labels_train = runtime_root / "labels" / "train"
    runtime_labels_val = runtime_root / "labels" / "val"

    if source_val_images:
        train_images = source_train_images
        val_images = source_val_images
    else:
        ordered = sorted(source_train_images)
        rng = random.Random(42)
        rng.shuffle(ordered)
        val_count = max(1, round(len(ordered) * 0.1))
        val_stems = {path.stem for path in ordered[:val_count]}
        train_images = [path for path in source_train_images if path.stem not in val_stems]
        val_images = [path for path in source_train_images if path.stem in val_stems]

    for image_path in train_images:
        label_path = source_labels_root / "train" / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
        _symlink_pair(image_path, label_path, runtime_images_train, runtime_labels_train)

    if source_val_images:
        for image_path in val_images:
            label_path = source_labels_root / "val" / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
            _symlink_pair(image_path, label_path, runtime_images_val, runtime_labels_val)
    else:
        for image_path in val_images:
            label_path = source_labels_root / "train" / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
            _symlink_pair(image_path, label_path, runtime_images_val, runtime_labels_val)

    return runtime_root


def _prepare_merged_runtime_dataset(dataset_roots: list[Path]) -> Path:
    if not dataset_roots:
        raise ValueError("At least one dataset root is required.")
    if len(dataset_roots) == 1:
        return _prepare_runtime_dataset(dataset_roots[0])

    runtime_root = Path(tempfile.mkdtemp(prefix="merged_yolo_runtime_"))
    runtime_images_train = runtime_root / "images" / "train"
    runtime_images_val = runtime_root / "images" / "val"
    runtime_labels_train = runtime_root / "labels" / "train"
    runtime_labels_val = runtime_root / "labels" / "val"

    for dataset_root in dataset_roots:
        source_images_root = dataset_root / "images"
        source_labels_root = _source_labels_root(dataset_root)
        source_train_images = _image_files(source_images_root / "train")
        source_val_images = _image_files(source_images_root / "val")

        if not source_train_images:
            raise FileNotFoundError(f"No training images found under {source_images_root / 'train'}")

        if source_val_images:
            train_images = source_train_images
            val_images = source_val_images
        else:
            ordered = sorted(source_train_images)
            rng = random.Random(42)
            rng.shuffle(ordered)
            val_count = max(1, round(len(ordered) * 0.1))
            val_stems = {path.stem for path in ordered[:val_count]}
            train_images = [path for path in source_train_images if path.stem not in val_stems]
            val_images = [path for path in source_train_images if path.stem in val_stems]

        prefix = dataset_root.name.replace(" ", "_")
        for image_path in train_images:
            label_path = source_labels_root / "train" / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
            _symlink_prefixed_pair(image_path, label_path, runtime_images_train, runtime_labels_train, prefix)

        if source_val_images:
            for image_path in val_images:
                label_path = source_labels_root / "val" / f"{image_path.stem}.txt"
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
                _symlink_prefixed_pair(image_path, label_path, runtime_images_val, runtime_labels_val, prefix)
        else:
            for image_path in val_images:
                label_path = source_labels_root / "train" / f"{image_path.stem}.txt"
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
                _symlink_prefixed_pair(image_path, label_path, runtime_images_val, runtime_labels_val, prefix)

    return runtime_root


def _build_runtime_dataset_yaml(runtime_root: Path, dataset_yaml: Path) -> Path:
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    payload["path"] = runtime_root.resolve().as_posix()
    payload["train"] = "images/train"
    payload["val"] = "images/val"

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"{runtime_root.name}_",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    return Path(handle.name)


def _default_run_name(dataset_root: Path, model_name: str) -> str:
    model_stem = Path(model_name).stem.replace(".", "_")
    return f"{dataset_root.name}_{model_stem}"


def _default_merged_run_name(dataset_roots: list[Path], model_name: str) -> str:
    model_stem = Path(model_name).stem.replace(".", "_")
    dataset_stem = "_".join(root.name for root in dataset_roots[:3])
    if len(dataset_roots) > 3:
        dataset_stem += f"_plus{len(dataset_roots) - 3}"
    return f"{dataset_stem}_{model_stem}"


def main() -> int:
    args = _parse_args()
    dataset_specs = _resolve_input_paths(args.data)
    dataset_roots = [dataset_root for dataset_root, _ in dataset_specs]
    dataset_yaml = dataset_specs[0][1]
    runtime_root = _prepare_merged_runtime_dataset(dataset_roots)
    runtime_dataset_yaml = _build_runtime_dataset_yaml(runtime_root, dataset_yaml)
    device = _choose_device(args.device)

    project_dir = Path(args.project).expanduser()
    if not project_dir.is_absolute():
        project_dir = (ROOT / project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.name or (
        _default_run_name(dataset_roots[0], args.model)
        if len(dataset_roots) == 1
        else _default_merged_run_name(dataset_roots, args.model)
    )
    cli_bits = [
        "yolo detect train",
        f"data=\"{runtime_dataset_yaml}\"",
        f"model={args.model}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"project=\"{project_dir}\"",
        f"name={run_name}",
    ]
    if device:
        cli_bits.append(f"device={device}")
    if args.cache:
        cli_bits.append("cache=True")
    if args.exist_ok:
        cli_bits.append("exist_ok=True")
    print("Resolved dataset roots:")
    for dataset_root in dataset_roots:
        print(" -", dataset_root)
    print("Primary dataset yaml:", dataset_yaml)
    print("Runtime dataset root:", runtime_root)
    print("Runtime dataset yaml:", runtime_dataset_yaml)
    print("Resolved device:", device)
    print("Equivalent command:")
    print(" ".join(cli_bits))

    if args.dry_run:
        return 0

    model = YOLO(args.model)
    results = model.train(
        data=str(runtime_dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        patience=args.patience,
        project=str(project_dir),
        name=run_name,
        cache=args.cache,
        exist_ok=args.exist_ok,
        plots=True,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"Training finished. Outputs saved to: {save_dir}")
    else:
        print("Training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
