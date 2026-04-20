"""CLI entry point for dataset generation and preview."""

from __future__ import annotations

import argparse
import sys

from btl2.app import SyntheticRoadApp
from btl2.utils.io import load_yaml
from btl2.utils.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser with generate and preview commands."""
    parser = argparse.ArgumentParser(description="Synthetic road-scene dataset generator")
    parser.add_argument("--config", default="configs/btl2/default.yaml", help="Path to YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a dataset split")
    generate.add_argument("--frames", type=int, default=None, help="Override number of frames")
    generate.add_argument("--output-dir", default=None, help="Override output directory")
    generate.add_argument("--seed", type=int, default=None, help="Override base seed")

    preview = subparsers.add_parser("preview", help="Render a single frame and save preview assets")
    preview.add_argument("--seed", type=int, default=None, help="Override preview seed")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch CLI commands into the application service."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging()
    config = load_yaml(args.config)
    if getattr(args, "output_dir", None):
        config["output_dir"] = args.output_dir
    if getattr(args, "seed", None) is not None:
        config["seed"] = args.seed

    app = SyntheticRoadApp(config)
    try:
        if args.command == "generate":
            summaries = app.generate_dataset(args.frames)
            logger.info("Generated %d frames under %s", len(summaries), config["output_dir"])
        elif args.command == "preview":
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
