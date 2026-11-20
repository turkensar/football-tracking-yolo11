"""Video reading and writing utilities built on OpenCV."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


class VideoReader:
    """Iterable video reader exposing video metadata and optional stride/limit.

    Each iteration opens a fresh ``cv2.VideoCapture`` and releases it in a
    ``finally`` block, so the same ``VideoReader`` instance can safely be
    iterated multiple times (for example, in a two-pass pipeline) without
    leaking OS file handles.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_frames: int | None = None,
        stride: int = 1,
    ) -> None:
        if max_frames is not None and (
            isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 0
        ):
            raise ValueError(
                f"'max_frames' must be a non-negative integer or None, got {max_frames!r}"
            )
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError(f"'stride' must be an integer >= 1, got {stride!r}")

        self.path = Path(path)
        self.max_frames = max_frames
        self.stride = stride

        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Unable to open video file: {self.path}")
            self._fps = float(cap.get(cv2.CAP_PROP_FPS))
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        if self._width <= 0 or self._height <= 0:
            raise ValueError(
                f"Video has invalid dimensions ({self._width}x{self._height}): {self.path}"
            )

    @property
    def fps(self) -> float:
        """Frames per second reported by the video container."""
        return self._fps

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def frame_count(self) -> int:
        """Total frame count reported by the video container."""
        return self._frame_count

    @property
    def size(self) -> tuple[int, int]:
        """Frame dimensions as ``(width, height)``."""
        return (self._width, self._height)

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_idx, bgr_frame)`` pairs honoring ``stride`` and ``max_frames``."""
        if self.max_frames == 0:
            return

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Unable to open video file: {self.path}")

        raw_idx = 0
        yielded = 0
        try:
            while True:
                if self.max_frames is not None and yielded >= self.max_frames:
                    break
                if self.stride == 1 or raw_idx % self.stride == 0:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    yield raw_idx, frame
                    yielded += 1
                else:
                    if not cap.grab():
                        break
                raw_idx += 1
        finally:
            cap.release()


class VideoWriter:
    """Context-managed video writer that auto-creates output directories."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        *,
        codec: str = "mp4v",
    ) -> None:
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps):
            raise ValueError(f"'fps' must be a finite positive number, got {fps!r}")
        if float(fps) <= 0.0:
            raise ValueError(f"'fps' must be positive, got {fps!r}")
        if (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in size)
        ):
            raise ValueError(
                f"'size' must be a (width, height) pair of positive ints, got {size!r}"
            )
        if not isinstance(codec, str) or len(codec) != 4:
            raise ValueError(f"'codec' must be a 4-character FourCC string, got {codec!r}")

        self.path = Path(path)
        self.fps = float(fps)
        self.width = int(size[0])
        self.height = int(size[1])
        self.codec = codec
        self.frames_written = 0
        self._closed = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            self._writer.release()
            self._closed = True
            raise RuntimeError(f"Failed to open VideoWriter for: {self.path}")

    @property
    def size(self) -> tuple[int, int]:
        """Output frame dimensions as ``(width, height)``."""
        return (self.width, self.height)

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR ``uint8`` frame matching ``(height, width, 3)``."""
        if self._closed:
            raise RuntimeError("Cannot write to a closed VideoWriter.")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel BGR numpy array, got {type(frame).__name__} "
                f"with shape {getattr(frame, 'shape', None)!r}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected frame dtype uint8, got {frame.dtype}")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame shape {frame.shape[:2]} does not match writer size "
                f"({self.height}, {self.width})"
            )
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        """Release the underlying OpenCV writer (idempotent)."""
        if not self._closed:
            self._writer.release()
            self._closed = True

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["VideoReader", "VideoWriter"]
