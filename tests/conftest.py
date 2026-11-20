"""Shared pytest fixtures for synthetic frames and videos."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import pytest

SYNTHETIC_VIDEO_WIDTH = 320
SYNTHETIC_VIDEO_HEIGHT = 240
SYNTHETIC_VIDEO_FPS = 25.0
SYNTHETIC_VIDEO_FRAMES = 10


class SyntheticVideoMeta(TypedDict):
    """Expected metadata for the synthetic video fixture."""

    width: int
    height: int
    fps: float
    frame_count: int


def create_synthetic_video(
    path: Path,
    *,
    width: int = SYNTHETIC_VIDEO_WIDTH,
    height: int = SYNTHETIC_VIDEO_HEIGHT,
    fps: float = SYNTHETIC_VIDEO_FPS,
    num_frames: int = SYNTHETIC_VIDEO_FRAMES,
) -> Path:
    """Write a deterministic synthetic MP4 video with a moving white square."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"OpenCV failed to open VideoWriter at {path}"
    try:
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (40, 120, 40)  # Green pitch-like background (BGR)
            x0 = 10 + i * 20
            cv2.rectangle(frame, (x0, 80), (x0 + 30, 140), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Return a clean 240x320 BGR synthetic pitch frame."""
    frame = np.zeros((SYNTHETIC_VIDEO_HEIGHT, SYNTHETIC_VIDEO_WIDTH, 3), dtype=np.uint8)
    frame[:, :] = (40, 120, 40)
    return frame


@pytest.fixture
def synthetic_video_meta() -> SyntheticVideoMeta:
    """Return expected width, height, fps, and frame count of ``synthetic_video_path``."""
    return {
        "width": SYNTHETIC_VIDEO_WIDTH,
        "height": SYNTHETIC_VIDEO_HEIGHT,
        "fps": SYNTHETIC_VIDEO_FPS,
        "frame_count": SYNTHETIC_VIDEO_FRAMES,
    }


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Path:
    """Return path to a 10-frame synthetic MP4 video written with OpenCV."""
    return create_synthetic_video(tmp_path / "synthetic.mp4")
