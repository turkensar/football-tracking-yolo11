"""Unit tests for the Annotator overlay renderer."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import supervision as sv

from futbolcv.annotate import Annotator, centers_to_bgr
from futbolcv.config import AnnotateConfig
from futbolcv.types import BallState, FrameResult


def _sample_result(*, ball_interpolated: bool = False) -> FrameResult:
    """Create a synthetic FrameResult with Team 0, Team 1, unknown (-1), and a ball."""
    players = sv.Detections(
        xyxy=np.array(
            [
                [50.0, 80.0, 90.0, 170.0],
                [140.0, 80.0, 180.0, 170.0],
                [230.0, 80.0, 270.0, 170.0],
            ],
            dtype=np.float32,
        ),
        confidence=np.array([0.9, 0.85, 0.88], dtype=np.float32),
        class_id=np.array([0, 0, 0], dtype=int),
        tracker_id=np.array([1, 2, 3], dtype=int),
        data={"team_id": np.array([0, 1, -1], dtype=int)},
    )
    ball = BallState(
        xyxy=np.array([150.0, 145.0, 162.0, 157.0], dtype=np.float32),
        conf=0.0 if ball_interpolated else 0.72,
        interpolated=ball_interpolated,
    )
    return FrameResult(frame_idx=42, players=players, ball=ball)


def test_annotate_preserves_shape_dtype_and_does_not_mutate_input(
    synthetic_frame: np.ndarray,
) -> None:
    """Annotate returns a modified copy with identical shape/dtype while input stays untouched."""
    original_snapshot = synthetic_frame.copy()
    annotator = Annotator(AnnotateConfig())

    out = annotator.annotate(synthetic_frame, _sample_result(), fps=25.0)

    assert out is not synthetic_frame
    assert out.shape == synthetic_frame.shape
    assert out.dtype == synthetic_frame.dtype
    assert np.array_equal(synthetic_frame, original_snapshot)
    assert not np.array_equal(out, original_snapshot)


def test_annotate_empty_detections_and_missing_ball_does_not_crash(
    synthetic_frame: np.ndarray,
) -> None:
    """Empty players and None ball render cleanly without raising."""
    original_snapshot = synthetic_frame.copy()
    empty_result = FrameResult(
        frame_idx=0,
        players=sv.Detections.empty(),
        ball=BallState(xyxy=None, conf=0.0, interpolated=False),
    )

    # With HUD enabled
    out_hud = Annotator(AnnotateConfig(show_hud=True)).annotate(synthetic_frame, empty_result)
    assert out_hud.shape == synthetic_frame.shape
    assert out_hud.dtype == synthetic_frame.dtype
    assert np.array_equal(synthetic_frame, original_snapshot)
    assert not np.array_equal(out_hud, original_snapshot)

    # With HUD disabled -> output equals input pixel-for-pixel
    out_plain = Annotator(AnnotateConfig(show_hud=False)).annotate(synthetic_frame, empty_result)
    assert np.array_equal(out_plain, original_snapshot)


def test_annotate_interpolated_ball_differs_from_detected_ball(
    synthetic_frame: np.ndarray,
) -> None:
    """Detected (filled) and interpolated (hollow) ball markers render distinct pixels."""
    annotator = Annotator(AnnotateConfig(show_hud=False))
    empty_players = sv.Detections.empty()

    ball_box = np.array([150.0, 120.0, 164.0, 134.0], dtype=np.float32)
    detected_res = FrameResult(
        frame_idx=1,
        players=empty_players,
        ball=BallState(xyxy=ball_box, conf=0.8, interpolated=False),
    )
    interp_res = FrameResult(
        frame_idx=1,
        players=empty_players,
        ball=BallState(xyxy=ball_box, conf=0.0, interpolated=True),
    )

    out_det = annotator.annotate(synthetic_frame, detected_res)
    out_interp = annotator.annotate(synthetic_frame, interp_res)

    assert not np.array_equal(out_det, out_interp)

    # Interior centroid of the triangle above the ball: filled when detected, hollow when interp
    bx = int(round((ball_box[0] + ball_box[2]) * 0.5))
    tip_y = max(16, int(round(ball_box[1])) - 4)
    center_y = tip_y - 9  # inside the 14px-tall downward triangle
    ball_color = np.array(AnnotateConfig().ball_color_bgr, dtype=np.uint8)

    assert np.array_equal(out_det[center_y, bx], ball_color)
    assert np.array_equal(out_interp[center_y, bx], synthetic_frame[center_y, bx])


def test_annotate_unknown_team_uses_gray_and_handles_missing_metadata(
    synthetic_frame: np.ndarray,
) -> None:
    """Players with team_id=-1 or missing team_id/tracker_id draw in unknown gray without error."""
    gray = (160, 160, 160)
    annotator = Annotator(AnnotateConfig(unknown_color_bgr=gray, show_hud=False))

    untagged = sv.Detections(
        xyxy=np.array([[120.0, 60.0, 160.0, 160.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )
    res = FrameResult(frame_idx=5, players=untagged, ball=BallState())
    out = annotator.annotate(synthetic_frame, res)

    # Ellipse pixels should contain the exact configured unknown gray color
    gray_matches = np.all(out == np.array(gray, dtype=np.uint8), axis=2)
    assert int(gray_matches.sum()) > 0


def test_centers_to_bgr_and_use_jersey_colors() -> None:
    """Cluster centers convert accurately to BGR and drive Annotator colors when enabled."""
    red_bgr = (20, 20, 220)
    blue_bgr = (220, 50, 20)
    centers_lab = np.vstack(
        [
            cv2.cvtColor(np.uint8([[red_bgr]]), cv2.COLOR_BGR2LAB)[0, 0],
            cv2.cvtColor(np.uint8([[blue_bgr]]), cv2.COLOR_BGR2LAB)[0, 0],
        ]
    ).astype(np.float64)

    converted = centers_to_bgr(centers_lab, color_space="lab")
    assert np.allclose(converted[0], red_bgr, atol=12)
    assert np.allclose(converted[1], blue_bgr, atol=12)
    assert converted[0][2] > converted[0][0]  # Red dominates R > B
    assert converted[1][0] > converted[1][2]  # Blue dominates B > R

    # use_jersey_colors=True adopts converted centers
    ann_jersey = Annotator(
        AnnotateConfig(use_jersey_colors=True),
        cluster_centers=centers_lab,
        color_space="lab",
    )
    assert ann_jersey.team_colors == converted

    # use_jersey_colors=False keeps static config colors
    cfg_static = AnnotateConfig(use_jersey_colors=False)
    ann_static = Annotator(cfg_static, cluster_centers=centers_lab, color_space="lab")
    assert ann_static.team_colors == {
        0: cfg_static.team_0_color_bgr,
        1: cfg_static.team_1_color_bgr,
    }

    with pytest.raises(ValueError, match="shape"):
        centers_to_bgr(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="color_space"):
        centers_to_bgr(centers_lab, color_space="rgb")
