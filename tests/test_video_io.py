"""Unit tests for VideoReader and VideoWriter."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from futbolcv.video_io import VideoReader, VideoWriter


def test_video_reader_metadata_and_frames(
    synthetic_video_path: Path, synthetic_video_meta: dict[str, Any]
) -> None:
    """VideoReader matches synthetic video frame count, size, fps, and content."""
    reader = VideoReader(synthetic_video_path)
    expected_w = synthetic_video_meta["width"]
    expected_h = synthetic_video_meta["height"]
    expected_fps = synthetic_video_meta["fps"]
    expected_frames = synthetic_video_meta["frame_count"]

    assert reader.width == expected_w
    assert reader.height == expected_h
    assert reader.size == (expected_w, expected_h)
    assert reader.fps == pytest.approx(expected_fps)
    assert reader.frame_count == expected_frames

    frames = list(reader)
    assert len(frames) == expected_frames
    assert [idx for idx, _ in frames] == list(range(expected_frames))

    for _, frame in frames:
        assert frame.shape == (expected_h, expected_w, 3)
        assert frame.dtype == np.uint8

    # First and last frames differ because the synthetic square moves across frames
    assert not np.array_equal(frames[0][1], frames[-1][1])


def test_video_reader_max_frames(synthetic_video_path: Path) -> None:
    """max_frames limits the number of yielded frames."""
    frames = list(VideoReader(synthetic_video_path, max_frames=4))
    assert [idx for idx, _ in frames] == [0, 1, 2, 3]

    empty = list(VideoReader(synthetic_video_path, max_frames=0))
    assert empty == []


def test_video_reader_stride_and_max_frames(synthetic_video_path: Path) -> None:
    """stride skips intermediate frames while preserving original frame indices."""
    strided = list(VideoReader(synthetic_video_path, stride=3))
    assert [idx for idx, _ in strided] == [0, 3, 6, 9]

    capped = list(VideoReader(synthetic_video_path, stride=2, max_frames=3))
    assert [idx for idx, _ in capped] == [0, 2, 4]


def test_video_reader_reiterable_and_context_manager(
    synthetic_video_path: Path, synthetic_video_meta: dict[str, Any]
) -> None:
    """VideoReader supports context manager usage and repeated iteration (two-pass)."""
    with VideoReader(synthetic_video_path) as reader:
        first_pass = [idx for idx, _ in reader]
        second_pass = [idx for idx, _ in reader]

    expected = list(range(synthetic_video_meta["frame_count"]))
    assert first_pass == expected
    assert second_pass == expected


def test_video_reader_missing_file_raises(tmp_path: Path) -> None:
    """Missing video path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "missing.mp4")


def test_video_reader_corrupt_file_raises(tmp_path: Path) -> None:
    """Unreadable/corrupt file raises ValueError."""
    bad_file = tmp_path / "corrupt.mp4"
    bad_file.write_bytes(b"not a valid video file")

    with pytest.raises(ValueError, match="Unable to open"):
        VideoReader(bad_file)


@pytest.mark.parametrize(
    ("kwargs",),
    [
        ({"stride": 0},),
        ({"stride": -2},),
        ({"stride": True},),
        ({"max_frames": -1},),
        ({"max_frames": False},),
    ],
)
def test_video_reader_invalid_parameters(
    synthetic_video_path: Path, kwargs: dict[str, object]
) -> None:
    """Invalid stride or max_frames values raise ValueError."""
    with pytest.raises(ValueError):
        VideoReader(synthetic_video_path, **kwargs)  # type: ignore[arg-type]


def test_video_writer_creates_parent_dirs_and_roundtrips(tmp_path: Path) -> None:
    """VideoWriter creates missing directories and writes a readable MP4 video."""
    out_path = tmp_path / "nested" / "dir" / "output.mp4"
    width, height, fps, num_frames = 160, 120, 20.0, 10

    with VideoWriter(out_path, fps=fps, size=(width, height)) as writer:
        assert writer.size == (width, height)
        for i in range(num_frames):
            frame = np.full((height, width, 3), fill_value=i * 20, dtype=np.uint8)
            writer.write(frame)
        assert writer.frames_written == num_frames

    assert out_path.is_file()
    reader = VideoReader(out_path)
    assert reader.width == width
    assert reader.height == height
    assert reader.fps == pytest.approx(fps)
    assert reader.frame_count == num_frames
    assert len(list(reader)) == num_frames


def test_video_writer_validation_and_closed_guard(tmp_path: Path) -> None:
    """VideoWriter rejects wrong shapes/dtypes and writes after closing."""
    out_path = tmp_path / "guarded.mp4"
    writer = VideoWriter(out_path, fps=25.0, size=(160, 120))

    with writer:
        with pytest.raises(ValueError, match="shape"):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="uint8"):
            writer.write(np.zeros((120, 160, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="3-channel"):
            writer.write(np.zeros((120, 160), dtype=np.uint8))

    with pytest.raises(RuntimeError, match="closed"):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("fps", "size", "codec"),
    [
        (0.0, (160, 120), "mp4v"),
        (-25.0, (160, 120), "mp4v"),
        (float("nan"), (160, 120), "mp4v"),
        (True, (160, 120), "mp4v"),
        (25.0, (0, 120), "mp4v"),
        (25.0, (160, -1), "mp4v"),
        (25.0, (160, 120), "mp4"),
    ],
)
def test_video_writer_invalid_constructor_args(
    tmp_path: Path, fps: float, size: tuple[int, int], codec: str
) -> None:
    """Invalid fps, size, or codec raises ValueError before opening OpenCV writer."""
    with pytest.raises(ValueError):
        VideoWriter(tmp_path / "invalid.mp4", fps=fps, size=size, codec=codec)
