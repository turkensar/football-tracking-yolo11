"""Generate a compact (<= 10 MB) demo GIF from an annotated match video."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
from PIL import Image


def make_demo_gif(
    input_path: Path,
    output_path: Path,
    *,
    start_sec: float = 0.0,
    duration_sec: float = 7.2,
    width: int = 960,
    fps: float = 10.0,
    colors: int = 128,
    max_mb: float = 10.0,
) -> Path:
    """Extract a time window from a video and save it as an optimized animated GIF.

    Args:
        input_path: Path to the source video file.
        output_path: Destination path for the generated ``.gif``.
        start_sec: Start timestamp in seconds (>= 0).
        duration_sec: Clip length in seconds (> 0).
        width: Output width in pixels (> 0); height preserves aspect ratio.
        fps: Target GIF frame rate (> 0).
        colors: Palette color count per frame (2..256).
        max_mb: Maximum allowed output size in megabytes (> 0).

    Returns:
        The written ``output_path``.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If arguments are invalid, no frames are extracted, or GIF exceeds ``max_mb``.
        RuntimeError: If OpenCV fails to open the input video.
    """
    if start_sec < 0:
        raise ValueError(f"start_sec must be >= 0, got {start_sec}")
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if not (2 <= colors <= 256):
        raise ValueError(f"colors must be between 2 and 256, got {colors}")
    if max_mb <= 0:
        raise ValueError(f"max_mb must be > 0, got {max_mb}")

    src = Path(input_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input video not found: {src}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src}")

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0.0:
            src_fps = 25.0

        start_frame = int(round(start_sec * src_fps))
        end_frame = int(round((start_sec + duration_sec) * src_fps))
        step = max(1.0, src_fps / fps)

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        pil_frames: list[Image.Image] = []
        frame_idx = start_frame
        next_sample = float(start_frame)

        while frame_idx < end_frame:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx + 1e-6 >= next_sample:
                h, w = frame_bgr.shape[:2]
                target_h = max(1, int(round(h * (width / w))))
                resized = cv2.resize(frame_bgr, (width, target_h), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb).quantize(
                    colors=colors,
                    method=Image.Quantize.FASTOCTREE,
                    dither=Image.Dither.NONE,
                )
                pil_frames.append(img)
                next_sample += step
            frame_idx += 1
    finally:
        cap.release()

    if not pil_frames:
        end_sec = start_sec + duration_sec
        raise ValueError(f"No frames extracted from {src} for window [{start_sec}s, {end_sec}s].")

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    effective_fps = src_fps / step
    frame_duration_ms = max(20, int(round(1000.0 / effective_fps)))

    pil_frames[0].save(
        dst,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=True,
        duration=frame_duration_ms,
        loop=0,
    )

    size_mb = dst.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"Generated GIF is {size_mb:.2f} MiB, exceeding max_mb={max_mb:.2f} MiB. "
            "Reduce --width, --fps, --duration, or --colors."
        )
    return dst


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``make_demo_gif.py``."""
    parser = argparse.ArgumentParser(
        description="Convert a slice of an annotated video into a <= 10 MB demo GIF.",
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("docs/assets/demo.gif"),
        help="Output GIF path (default: docs/assets/demo.gif).",
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Start time in seconds (default: 0.0)."
    )
    parser.add_argument(
        "--duration", type=float, default=7.2, help="Clip duration in seconds (default: 7.2)."
    )
    parser.add_argument(
        "--width", type=int, default=960, help="Output width in pixels (default: 960)."
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Target GIF FPS (default: 10.0).")
    parser.add_argument(
        "--colors",
        type=int,
        default=128,
        help="Palette colors per frame, 2..256 (default: 128).",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=10.0,
        help="Maximum allowed GIF size in MiB (default: 10.0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for GIF generation."""
    args = build_parser().parse_args(argv)
    try:
        out = make_demo_gif(
            args.input,
            args.output,
            start_sec=args.start,
            duration_sec=args.duration,
            width=args.width,
            fps=args.fps,
            colors=args.colors,
            max_mb=args.max_mb,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Saved {out} ({size_mb:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
