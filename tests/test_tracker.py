"""Unit tests for the ByteTrack-backed PlayerTracker."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import supervision as sv

from futbolcv.config import TrackerConfig
from futbolcv.tracker import PlayerTracker


def _two_player_detections(frame_idx: int, *, step_px: float = 5.0) -> sv.Detections:
    """Create two synthetic player boxes shifting right by ``step_px`` per frame."""
    shift = frame_idx * step_px
    xyxy = np.array(
        [
            [50.0 + shift, 100.0, 90.0 + shift, 200.0],
            [200.0 + shift, 120.0, 240.0 + shift, 220.0],
        ],
        dtype=np.float32,
    )
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9, 0.88], dtype=np.float32),
        class_id=np.array([0, 0], dtype=int),
    )


def test_player_tracker_preserves_ids_across_20_frames() -> None:
    """Two boxes moving 5 px/frame right keep the exact same two tracker IDs for 20 frames."""
    tracker = PlayerTracker(TrackerConfig(), fps=25.0)

    initial_ids: list[int] | None = None
    for frame_idx in range(20):
        tracked = tracker.update(_two_player_detections(frame_idx, step_px=5.0))
        assert len(tracked) == 2
        assert tracked.tracker_id is not None

        # Sort by x1 to verify each physical player keeps its own persistent ID
        order = np.argsort(tracked.xyxy[:, 0])
        ordered_ids = [int(tracked.tracker_id[i]) for i in order]
        assert len(set(ordered_ids)) == 2

        if initial_ids is None:
            initial_ids = ordered_ids
        else:
            assert ordered_ids == initial_ids

    assert initial_ids is not None


def test_player_tracker_reset_restarts_ids() -> None:
    """Calling reset() clears state and restarts tracker IDs from the beginning."""
    tracker = PlayerTracker(TrackerConfig(), fps=25.0)

    first_run = tracker.update(_two_player_detections(0))
    assert first_run.tracker_id is not None
    first_ids = sorted(int(tid) for tid in first_run.tracker_id)

    for frame_idx in range(1, 5):
        tracker.update(_two_player_detections(frame_idx))

    tracker.reset()

    after_reset = tracker.update(_two_player_detections(0))
    assert after_reset.tracker_id is not None
    reset_ids = sorted(int(tid) for tid in after_reset.tracker_id)
    assert reset_ids == first_ids


def test_player_tracker_uses_video_fps_or_config_override() -> None:
    """frame_rate=None uses video fps; explicit config frame_rate overrides it."""
    auto_cfg = TrackerConfig(frame_rate=None, lost_track_buffer=30)
    tracker_25 = PlayerTracker(auto_cfg, fps=25.0)
    assert tracker_25.frame_rate == pytest.approx(25.0)
    assert tracker_25._tracker.max_time_lost == int(25.0 / 30.0 * 30)

    override_cfg = TrackerConfig(frame_rate=50.0, lost_track_buffer=30)
    tracker_override = PlayerTracker(override_cfg, fps=25.0)
    assert tracker_override.frame_rate == pytest.approx(50.0)
    assert tracker_override._tracker.max_time_lost == int(50.0 / 30.0 * 30)


def test_player_tracker_handles_empty_detections() -> None:
    """Passing sv.Detections.empty() succeeds and advances internal frame counter."""
    tracker = PlayerTracker(TrackerConfig(), fps=25.0)

    out = tracker.update(sv.Detections.empty())
    assert len(out) == 0
    assert out.tracker_id is not None
    assert len(out.tracker_id) == 0
    assert tracker._tracker.frame_id == 1


def test_player_tracker_suppresses_future_warning_only_locally() -> None:
    """ByteTrack deprecation warning is suppressed locally without mutating global filters."""
    with warnings.catch_warnings(record=True) as caught_init:
        warnings.simplefilter("always")
        PlayerTracker(TrackerConfig(), fps=25.0)
    assert not any(issubclass(w.category, FutureWarning) for w in caught_init)

    with warnings.catch_warnings(record=True) as caught_after:
        warnings.simplefilter("always")
        warnings.warn("probe", FutureWarning, stacklevel=1)
    assert len(caught_after) == 1
    assert issubclass(caught_after[0].category, FutureWarning)


@pytest.mark.parametrize("bad_fps", [0.0, -25.0, float("nan"), float("inf"), True])
def test_player_tracker_invalid_fps_raises(bad_fps: float) -> None:
    """Non-positive, non-finite, or boolean fps raises ValueError."""
    with pytest.raises(ValueError):
        PlayerTracker(TrackerConfig(), fps=bad_fps)


def test_player_tracker_preserves_class_ids_for_players_and_referees() -> None:
    """ByteTrack preserves distinct class_id values (player vs referee) across tracked frames."""
    tracker = PlayerTracker(TrackerConfig(), fps=25.0)

    for frame_idx in range(10):
        shift = frame_idx * 4.0
        dets = sv.Detections(
            xyxy=np.array(
                [
                    [40.0 + shift, 80.0, 80.0 + shift, 180.0],
                    [200.0 + shift, 90.0, 240.0 + shift, 190.0],
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.85], dtype=np.float32),
            class_id=np.array([2, 3], dtype=int),  # 2=player, 3=referee
        )
        tracked = tracker.update(dets)
        assert len(tracked) == 2
        assert tracked.class_id is not None
        assert tracked.tracker_id is not None

        order = np.argsort(tracked.xyxy[:, 0])
        assert [int(tracked.class_id[i]) for i in order] == [2, 3]
