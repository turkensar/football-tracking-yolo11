"""Command-line interface for running the futbol-cv analysis pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from futbolcv.config import load_config

_ULTRALYTICS_ASSETS_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ``python -m futbolcv``."""
    parser = argparse.ArgumentParser(
        prog="python -m futbolcv",
        description=(
            "Detect and track football players and ball, classify teams by jersey color, "
            "and export an annotated video alongside tracks.csv and summary.json."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the input football match video.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs"),
        help="Directory where annotated video, tracks.csv, and summary.json will be written.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of sampled frames to process.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override detector device (e.g., auto, cpu, 0, cuda:0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, run the pipeline, and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    weights_label = "yolo11s.pt"
    try:
        if args.max_frames is not None and args.max_frames < 0:
            raise ValueError(f"--max-frames must be >= 0, got {args.max_frames}")

        input_path: Path = args.input
        if not input_path.is_file():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cfg = load_config(args.config)
        weights_label = cfg.detector.weights
        if args.device is not None:
            cleaned_device = args.device.strip()
            if not cleaned_device:
                raise ValueError("--device must be a non-empty string.")
            cfg.detector.device = cleaned_device

        # Lazy import keeps `--help` and path/config error handling fast (avoids loading torch).
        from futbolcv.pipeline import run

        output_video = run(
            input_path,
            args.output,
            cfg,
            max_frames=args.max_frames,
        )
        print(f"Done. Annotated video: {output_video}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except ConnectionError:
        weights_name = Path(weights_label).name
        url = f"{_ULTRALYTICS_ASSETS_BASE_URL}/{weights_name}"
        print(
            f"Error: failed to download model weights ({weights_label}). "
            f"Download manually and place under models/: {url}",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, ValueError, RuntimeError, OSError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
