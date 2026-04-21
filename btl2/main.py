"""Điểm vào CLI cho BTL 2: sinh dataset hoặc render thử một frame.

File này cố ý mỏng: nó chỉ đọc tham số dòng lệnh, nạp file YAML, rồi giao
toàn bộ nghiệp vụ cho `SyntheticRoadApp` trong `btl2/app.py`. Nhờ vậy phần
pipeline chính có thể được gọi lại từ giao diện BTL 1 mà không phụ thuộc CLI.
"""

from __future__ import annotations

import argparse
import sys

from btl2.app import SyntheticRoadApp
from btl2.utils.io import load_yaml
from btl2.utils.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Tạo bộ đọc tham số với hai lệnh chính: `generate` và `preview`."""
    parser = argparse.ArgumentParser(description="Synthetic road-scene dataset generator")
    # `--config` quyết định toàn bộ thông số render: kích thước ảnh, số frame,
    # seed, vùng spawn object, ánh sáng, tỉ lệ train/val và định dạng annotation.
    parser.add_argument("--config", default="configs/btl2/default.yaml", help="Path to YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Lệnh generate chạy đủ pipeline: dựng scene -> render 3 pass -> xuất ảnh,
    # depth, mask, nhãn YOLO/COCO/custom CSV và kiểm tra tính nhất quán dataset.
    generate = subparsers.add_parser("generate", help="Generate a dataset split")
    generate.add_argument("--frames", type=int, default=None, help="Override number of frames")
    generate.add_argument("--output-dir", default=None, help="Override output directory")
    generate.add_argument("--seed", type=int, default=None, help="Override base seed")

    # Lệnh preview chỉ render một frame trong bộ nhớ. Nó hữu ích để kiểm tra
    # nhanh cấu hình camera/ánh sáng/object trước khi sinh cả dataset.
    preview = subparsers.add_parser("preview", help="Render a single frame and save preview assets")
    preview.add_argument("--seed", type=int, default=None, help="Override preview seed")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Đọc tham số CLI, áp override nếu có, rồi gọi service BTL 2."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging()
    config = load_yaml(args.config)
    # Các override từ CLI chỉ thay đổi bản config trong bộ nhớ, không ghi ngược
    # lại YAML. Cách này giúp thử nghiệm nhiều lần mà không làm bẩn file cấu hình.
    if getattr(args, "output_dir", None):
        config["output_dir"] = args.output_dir
    if getattr(args, "seed", None) is not None:
        config["seed"] = args.seed

    app = SyntheticRoadApp(config)
    try:
        if args.command == "generate":
            # `summaries` là danh sách đường dẫn output theo từng frame, thường
            # dùng cho log/UI; dữ liệu thật đã được ghi trong `export_frame`.
            summaries = app.generate_dataset(args.frames)
            logger.info("Generated %d frames under %s", len(summaries), config["output_dir"])
        elif args.command == "preview":
            # Preview trả về các mảng numpy thay vì ghi ra bộ dataset hoàn chỉnh.
            frame = app.preview_scene(args.seed)
            logger.info(
                "Preview rendered: rgb=%s, depth=%s, mask=%s",
                frame.rgb.shape,
                frame.depth_gray.shape,
                frame.mask_rgb.shape,
            )
        else:
            parser.error(f"Unknown command: {args.command}")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
