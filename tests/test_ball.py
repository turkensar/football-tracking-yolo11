"""Unit tests for BallTracker candidate selection, jump gating, and interpolation."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from futbolcv.ball import BallTracker, interpolate
from futbolcv.config import BallConfig
from futbolcv.types import BallState, FrameResult


def _make_candidates(boxes: list[list[float]], confs: list[float]) -> sv.Detections:
    """Build synthetic ball candidate Detections."""
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.full(len(boxes), 32, dtype=int),
    )


def _state(box: list[float] | None, conf: float = 0.8) -> BallState:
    """Create a BallState from a 4-float box list or None."""
    if box is None:
        return BallState(xyxy=None, conf=0.0, interpolated=False)
    return BallState(xyxy=np.array(box, dtype=np.float32), conf=conf, interpolated=False)


def test_interpolate_fills_3_frame_gap_with_exact_coordinates() -> None:
    """(a) A 3-frame interior gap is linearly filled with conf=0.0 and interpolated=True."""
    states = [
        _state([0.0, 0.0, 10.0, 10.0], conf=0.9),
        _state(None),
        _state(None),
        _state(None),
        _state([40.0, 20.0, 50.0, 30.0], conf=0.7),
    ]

    filled = interpolate(states, max_gap=5)

    # Original list is not mutated
    assert states[1].xyxy is None
    assert states[2].xyxy is None

    # Endpoints stay untouched
    assert filled[0].interpolated is False
    assert filled[0].conf == pytest.approx(0.9)
    assert filled[4].interpolated is False
    assert filled[4].conf == pytest.approx(0.7)

    # 3 interior frames match exact 1/4, 2/4, 3/4 linear steps
    expected = [
        [10.0, 5.0, 20.0, 15.0],
        [20.0, 10.0, 30.0, 20.0],
        [30.0, 15.0, 40.0, 25.0],
    ]
    for idx, exp_box in zip((1, 2, 3), expected, strict=True):
        assert filled[idx].xyxy is not None
        assert np.allclose(filled[idx].xyxy, np.array(exp_box, dtype=np.float32))
        assert filled[idx].conf == pytest.approx(0.0)
        assert filled[idx].interpolated is True


def test_interpolate_respects_max_gap_and_leaves_max_gap_plus_one_empty() -> None:
    """(b) A gap of length max_gap is filled, while max_gap+1 stays None."""
    max_gap = 3
    states_exact = [_state([0, 0, 10, 10])] + [_state(None)] * max_gap + [_state([40, 0, 50, 10])]
    filled_exact = interpolate(states_exact, max_gap=max_gap)
    assert all(s.xyxy is not None for s in filled_exact)
    assert all(filled_exact[i].interpolated for i in range(1, max_gap + 1))

    states_too_long = (
        [_state([0, 0, 10, 10])] + [_state(None)] * (max_gap + 1) + [_state([50, 0, 60, 10])]
    )
    filled_long = interpolate(states_too_long, max_gap=max_gap)
    for i in range(1, max_gap + 2):
        assert filled_long[i].xyxy is None
        assert filled_long[i].conf == pytest.approx(0.0)
        assert filled_long[i].interpolated is False


def test_interpolate_does_not_fill_leading_or_trailing_gaps() -> None:
    """(c) Leading and trailing missing frames are never extrapolated."""
    states = [
        _state(None),
        _state(None),
        _state([10, 10, 20, 20], conf=0.8),
        _state(None),
        _state([30, 10, 40, 20], conf=0.85),
        _state(None),
        _state(None),
    ]

    filled = interpolate(states, max_gap=10)

    # Leading & trailing remain None
    for idx in (0, 1, 5, 6):
        assert filled[idx].xyxy is None
        assert filled[idx].interpolated is False

    # Interior 1-frame gap is filled at exact midpoint
    assert filled[3].xyxy is not None
    assert np.allclose(filled[3].xyxy, np.array([20, 10, 30, 20], dtype=np.float32))
    assert filled[3].interpolated is True
    assert filled[3].conf == pytest.approx(0.0)


def test_interpolate_edge_cases_and_validation() -> None:
    """Empty, all-None, single-detection, max_gap=0, and invalid max_gap are handled."""
    assert interpolate([], max_gap=5) == []

    all_none = [_state(None), _state(None)]
    out_none = interpolate(all_none, max_gap=5)
    assert all(s.xyxy is None for s in out_none)

    single = [_state(None), _state([5, 5, 15, 15]), _state(None)]
    out_single = interpolate(single, max_gap=5)
    assert out_single[0].xyxy is None
    assert out_single[1].xyxy is not None
    assert out_single[2].xyxy is None

    gap_one = [_state([0, 0, 10, 10]), _state(None), _state([20, 0, 30, 10])]
    out_zero = interpolate(gap_one, max_gap=0)
    assert out_zero[1].xyxy is None

    with pytest.raises(ValueError):
        interpolate(gap_one, max_gap=-1)
    with pytest.raises(ValueError):
        interpolate(gap_one, max_gap=True)


def test_select_picks_candidate_closest_to_previous() -> None:
    """(d) Given two candidates and a previous box, the closer candidate wins over higher conf."""
    tracker = BallTracker(BallConfig(max_jump_px=150.0))
    prev = np.array([100.0, 100.0, 110.0, 110.0], dtype=np.float32)

    candidates = _make_candidates(
        boxes=[
            [210.0, 100.0, 220.0, 110.0],  # 110 px away, higher conf
            [120.0, 100.0, 130.0, 110.0],  # 20 px away, lower conf -> should win
        ],
        confs=[0.95, 0.35],
    )

    chosen = tracker.select(candidates, prev)
    assert chosen.xyxy is not None
    assert np.allclose(chosen.xyxy, np.array([120.0, 100.0, 130.0, 110.0], dtype=np.float32))
    assert chosen.conf == pytest.approx(0.35)
    assert chosen.interpolated is False


def test_select_without_previous_picks_highest_confidence() -> None:
    """Without prev_xyxy, select picks the highest-confidence candidate."""
    tracker = BallTracker(BallConfig(max_jump_px=150.0))
    candidates = _make_candidates(
        boxes=[
            [10.0, 10.0, 20.0, 20.0],
            [300.0, 200.0, 310.0, 210.0],
        ],
        confs=[0.30, 0.82],
    )

    chosen = tracker.select(candidates, prev_xyxy=None)
    assert chosen.xyxy is not None
    assert np.allclose(chosen.xyxy, np.array([300.0, 200.0, 310.0, 210.0], dtype=np.float32))
    assert chosen.conf == pytest.approx(0.82)


def test_select_empty_candidates_returns_none_state() -> None:
    """Empty detections return an empty BallState."""
    tracker = BallTracker(BallConfig())
    state = tracker.select(sv.Detections.empty(), prev_xyxy=None)
    assert state.xyxy is None
    assert state.conf == pytest.approx(0.0)
    assert state.interpolated is False


def test_select_jump_threshold_scales_with_frames_since_last() -> None:
    """Max jump limit is per-frame: max_jump_px * frames_since_last."""
    tracker = BallTracker(BallConfig(max_jump_px=100.0))
    prev = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    # Center moves from (5, 5) to (255, 5) -> 250 px jump
    far_candidate = _make_candidates([[250.0, 0.0, 260.0, 10.0]], [0.9])

    # 1 frame -> limit 100 px -> rejected
    assert tracker.select(far_candidate, prev, frames_since_last=1).xyxy is None
    # 2 frames -> limit 200 px -> rejected
    assert tracker.select(far_candidate, prev, frames_since_last=2).xyxy is None
    # 3 frames -> limit 300 px -> accepted
    accepted = tracker.select(far_candidate, prev, frames_since_last=3)
    assert accepted.xyxy is not None
    assert np.allclose(accepted.xyxy, np.array([250.0, 0.0, 260.0, 10.0], dtype=np.float32))

    with pytest.raises(ValueError):
        tracker.select(far_candidate, prev, frames_since_last=0)


def test_ball_tracker_stateful_update_and_reset() -> None:
    """Stateful update rejects 1-frame outlier jump and reacquires ball after missing frames."""
    tracker = BallTracker(BallConfig(max_jump_px=100.0))

    # Frame 0: initial detection at x=0..10
    s0 = tracker.update(_make_candidates([[0.0, 0.0, 10.0, 10.0]], [0.8]), frame_idx=0)
    assert s0.xyxy is not None

    # Frame 1: false positive 250 px away -> rejected (limit 100*1)
    s1 = tracker.update(_make_candidates([[250.0, 0.0, 260.0, 10.0]], [0.9]), frame_idx=1)
    assert s1.xyxy is None

    # Frame 2: ball occluded
    s2 = tracker.update(sv.Detections.empty(), frame_idx=2)
    assert s2.xyxy is None

    # Frame 3: ball reappears 250 px from frame 0 -> accepted (limit 100*3 = 300)
    s3 = tracker.update(_make_candidates([[250.0, 0.0, 260.0, 10.0]], [0.75]), frame_idx=3)
    assert s3.xyxy is not None
    assert np.allclose(s3.xyxy, np.array([250.0, 0.0, 260.0, 10.0], dtype=np.float32))

    # Reset forgets previous position, so a far jump is immediately accepted on next frame
    tracker.reset()
    s4 = tracker.update(_make_candidates([[900.0, 500.0, 910.0, 510.0]], [0.7]), frame_idx=4)
    assert s4.xyxy is not None


def test_frame_result_contract() -> None:
    """FrameResult holds frame_idx, players Detections, and BallState."""
    ball = _state([10.0, 10.0, 20.0, 20.0], conf=0.6)
    res = FrameResult(frame_idx=7, players=sv.Detections.empty(), ball=ball)
    assert res.frame_idx == 7
    assert len(res.players) == 0
    assert res.ball is ball


def test_penalty_spot_static_lock_switches_to_moving_ball_within_3_frames() -> None:
    """Synthetic penalty-spot test: switches from static spot to moving ball in <= 3 frames."""
    tracker = BallTracker(
        BallConfig(
            max_jump_px=150.0,
            distance_weight=0.3,
            static_radius_px=5.0,
            static_frames=25,
            reacquire_conf=0.5,
        )
    )
    spot_box = [500.0, 500.0, 512.0, 512.0]

    # Frames 0..49: single candidate sitting on the penalty spot -> selected on every frame.
    for frame_idx in range(50):
        state = tracker.update(_make_candidates([spot_box], [0.70]), frame_idx=frame_idx)
        assert state.xyxy is not None
        assert np.allclose(state.xyxy, np.array(spot_box, dtype=np.float32))

    # Frames 50..79: spot stays at (500, 500) with conf=0.60, real ball moves right at 20 px/frame.
    switch_frame: int | None = None
    for frame_idx in range(50, 80):
        step = frame_idx - 49
        real_box = [500.0 + step * 20.0, 500.0, 512.0 + step * 20.0, 512.0]
        state = tracker.update(
            _make_candidates([spot_box, real_box], [0.60, 0.75]),
            frame_idx=frame_idx,
        )
        assert state.xyxy is not None
        on_real = bool(np.allclose(state.xyxy, np.array(real_box, dtype=np.float32)))
        if switch_frame is None and on_real:
            switch_frame = frame_idx
        if switch_frame is not None:
            assert on_real, f"Dropped back to spot at frame {frame_idx}"

    assert switch_frame is not None
    assert switch_frame <= 52  # Switched within <= 3 frames (50, 51, or 52)


def test_reacquire_breaks_static_lock_when_moving_ball_is_outside_jump_gate() -> None:
    """Out-of-gate moving candidate >= reacquire_conf breaks a static-spot lock; weak FP cannot."""
    tracker = BallTracker(
        BallConfig(
            max_jump_px=150.0,
            static_radius_px=5.0,
            static_frames=25,
            reacquire_conf=0.5,
        )
    )
    spot_box = [500.0, 500.0, 512.0, 512.0]

    # Establish lock on static spot over 30 frames
    for frame_idx in range(30):
        state = tracker.update(_make_candidates([spot_box], [0.68]), frame_idx=frame_idx)
        assert state.xyxy is not None

    # Frame 30: far moving candidate below reacquire_conf (0.35 < 0.50) must NOT steal lock
    weak_far_box = [1100.0, 500.0, 1112.0, 512.0]
    s30 = tracker.update(_make_candidates([spot_box, weak_far_box], [0.65, 0.35]), frame_idx=30)
    assert s30.xyxy is not None
    assert np.allclose(s30.xyxy, np.array(spot_box, dtype=np.float32))

    # Frame 31: far moving real ball >= reacquire_conf (0.74 >= 0.50) breaks lock immediately
    real_far_box = [1150.0, 510.0, 1162.0, 522.0]
    s31 = tracker.update(_make_candidates([spot_box, real_far_box], [0.65, 0.74]), frame_idx=31)
    assert s31.xyxy is not None
    assert np.allclose(s31.xyxy, np.array(real_far_box, dtype=np.float32))

    # Frame 32: continues tracking the moving real ball smoothly
    next_real_box = [1180.0, 512.0, 1192.0, 524.0]
    s32 = tracker.update(_make_candidates([spot_box, next_real_box], [0.65, 0.72]), frame_idx=32)
    assert s32.xyxy is not None
    assert np.allclose(s32.xyxy, np.array(next_real_box, dtype=np.float32))


def test_stationary_single_ball_stays_tracked_for_set_piece() -> None:
    """A single stationary ball (pre-penalty / corner) remains selected on every frame."""
    tracker = BallTracker(BallConfig(static_radius_px=5.0, static_frames=25))
    ball_box = [320.0, 410.0, 332.0, 422.0]

    for frame_idx in range(60):
        state = tracker.update(_make_candidates([ball_box], [0.78]), frame_idx=frame_idx)
        assert state.xyxy is not None
        assert np.allclose(state.xyxy, np.array(ball_box, dtype=np.float32))


def test_static_spot_forgotten_after_absence_and_reset() -> None:
    """Static spots expire after > static_frames of absence and clear immediately on reset()."""
    tracker = BallTracker(BallConfig(static_radius_px=5.0, static_frames=10, reacquire_conf=0.5))
    spot_box = [100.0, 100.0, 112.0, 112.0]

    for frame_idx in range(12):
        tracker.update(_make_candidates([spot_box], [0.70]), frame_idx=frame_idx)
    assert tracker._is_on_static_spot()

    # Absent for 11 frames (> static_frames=10) -> static memory pruned
    tracker.update(sv.Detections.empty(), frame_idx=23)
    assert tracker._static_spots == []

    # Re-establish and verify reset() clears all state
    for frame_idx in range(24, 36):
        tracker.update(_make_candidates([spot_box], [0.70]), frame_idx=frame_idx)
    assert tracker._is_on_static_spot()

    tracker.reset()
    assert tracker._static_spots == []
    assert not tracker._is_on_static_spot()


def test_abandoned_static_spot_returns_none_when_only_spot_remains() -> None:
    """After leaving a static spot for a moving ball, sole spot detections return None."""
    tracker = BallTracker(
        BallConfig(
            max_jump_px=150.0,
            distance_weight=0.3,
            static_radius_px=5.0,
            static_frames=25,
            reacquire_conf=0.5,
        )
    )
    spot_box = [500.0, 500.0, 512.0, 512.0]

    # Frames 0..49: only the stationary ball on the spot -> selected on every frame.
    for frame_idx in range(50):
        state = tracker.update(_make_candidates([spot_box], [0.75]), frame_idx=frame_idx)
        assert state.xyxy is not None
        assert np.allclose(state.xyxy, np.array(spot_box, dtype=np.float32))

    # Frames 50..59: real ball kicks away from the spot -> tracked cleanly; spot is abandoned.
    for frame_idx in range(50, 60):
        step = frame_idx - 49
        real_box = [500.0 + step * 25.0, 500.0, 512.0 + step * 25.0, 512.0]
        state = tracker.update(
            _make_candidates([spot_box, real_box], [0.65, 0.80]), frame_idx=frame_idx
        )
        assert state.xyxy is not None
        assert np.allclose(state.xyxy, np.array(real_box, dtype=np.float32))

    # Frames 60..99: real ball disappears (e.g. in net); only the abandoned spot remains -> None.
    for frame_idx in range(60, 100):
        state = tracker.update(_make_candidates([spot_box], [0.70]), frame_idx=frame_idx)
        assert state.xyxy is None, f"Expected None at frame {frame_idx}, got {state.xyxy}"

    # After reset(), the abandoned memory is cleared and a new stationary ball is accepted again.
    tracker.reset()
    after_reset = tracker.update(_make_candidates([spot_box], [0.75]), frame_idx=100)
    assert after_reset.xyxy is not None
