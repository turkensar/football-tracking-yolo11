"""Unit tests for jersey-color extraction and TeamClassifier."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import supervision as sv

from futbolcv.config import TeamConfig
from futbolcv.team import TeamClassifier, extract_jersey_color

PITCH_BGR = (40, 120, 40)
RED_BGR = (20, 20, 220)
BLUE_BGR = (220, 50, 20)
YELLOW_BGR = (0, 230, 240)
SHORTS_BGR = (240, 240, 240)


def _draw_player(
    frame: np.ndarray,
    box: list[float],
    jersey_bgr: tuple[int, int, int],
    *,
    shorts_bgr: tuple[int, int, int] = SHORTS_BGR,
) -> None:
    """Paint a synthetic player with a jersey in the upper-middle region and white shorts below."""
    x1, y1, x2, y2 = (int(v) for v in box)
    mid_y = y1 + (y2 - y1) // 2
    width = x2 - x1
    margin = width // 5
    # Paint shorts across the entire lower half
    frame[mid_y:y2, x1:x2] = shorts_bgr
    # Paint jersey in the center of the upper half (leaving grass visible on left/right edges)
    frame[y1:mid_y, x1 + margin : x2 - margin] = jersey_bgr


def _make_detections(
    boxes: list[list[float]],
    tracker_ids: list[int] | None = None,
) -> sv.Detections:
    """Build synthetic player Detections with optional tracker IDs."""
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.full(len(boxes), 0.9, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
        tracker_id=None if tracker_ids is None else np.array(tracker_ids, dtype=int),
    )


def test_extract_jersey_color_focuses_on_upper_middle_region() -> None:
    """Color extraction samples only the upper-half middle crop, ignoring shorts and edges."""
    frame = np.full((240, 320, 3), PITCH_BGR, dtype=np.uint8)
    box = [100.0, 40.0, 140.0, 160.0]
    _draw_player(frame, box, jersey_bgr=RED_BGR, shorts_bgr=BLUE_BGR)

    feat = extract_jersey_color(frame, np.array(box, dtype=np.float32), color_space="lab")
    assert feat is not None

    expected_red_lab = cv2.cvtColor(np.uint8([[RED_BGR]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    assert np.allclose(feat, expected_red_lab, atol=1.0)

    # Box completely outside frame returns None
    assert extract_jersey_color(frame, np.array([400.0, 400.0, 450.0, 500.0])) is None

    with pytest.raises(ValueError, match="color_space"):
        extract_jersey_color(frame, np.array(box), color_space="rgb")


def test_extract_jersey_color_excludes_green_grass_around_thin_torso() -> None:
    """Grass masking strips green background pixels around a thin black jersey."""
    frame = np.full((240, 320, 3), PITCH_BGR, dtype=np.uint8)
    # Thin black stripe (4px wide) inside a 24px wide box surrounded by green pitch
    box = np.array([100.0, 40.0, 124.0, 120.0], dtype=np.float32)
    frame[52:85, 110:114] = (15, 15, 15)

    with_mask = extract_jersey_color(frame, box, color_space="lab", exclude_grass=True)
    without_mask = extract_jersey_color(frame, box, color_space="lab", exclude_grass=False)
    assert with_mask is not None and without_mask is not None
    # With grass masking, L channel stays dark (~15); without masking, green pulls L much higher
    assert with_mask[0] < 30.0
    assert without_mask[0] > 70.0


def test_team_classifier_separates_red_blue_and_flags_yellow_outlier() -> None:
    """Red and blue jerseys form two distinct teams; yellow referee becomes -1."""
    frame = np.full((240, 480, 3), PITCH_BGR, dtype=np.uint8)
    red_boxes = [
        [20.0, 40.0, 60.0, 140.0],
        [80.0, 40.0, 120.0, 140.0],
        [140.0, 40.0, 180.0, 140.0],
    ]
    blue_boxes = [
        [200.0, 40.0, 240.0, 140.0],
        [260.0, 40.0, 300.0, 140.0],
        [320.0, 40.0, 360.0, 140.0],
    ]
    yellow_box = [390.0, 40.0, 430.0, 140.0]

    for b in red_boxes:
        _draw_player(frame, b, RED_BGR)
    for b in blue_boxes:
        _draw_player(frame, b, BLUE_BGR)
    _draw_player(frame, yellow_box, YELLOW_BGR)

    # Fit on red + blue players (also robust even if a single referee is included)
    train_dets = _make_detections(red_boxes + blue_boxes, tracker_ids=[1, 2, 3, 4, 5, 6])
    clf = TeamClassifier(TeamConfig(n_init_frames=5, outlier_distance_threshold=45.0))
    clf.fit([(frame, train_dets)])
    assert clf.is_fitted
    assert clf.cluster_centers is not None
    assert clf.cluster_centers.shape == (2, 3)

    all_dets = _make_detections(
        red_boxes + blue_boxes + [yellow_box],
        tracker_ids=[1, 2, 3, 4, 5, 6, 99],
    )
    team_ids = clf.predict(frame, all_dets)

    assert isinstance(team_ids, np.ndarray)
    assert np.issubdtype(team_ids.dtype, np.integer)
    assert "team_id" in all_dets.data
    assert np.array_equal(all_dets.data["team_id"], team_ids)

    red_labels = set(int(x) for x in team_ids[:3])
    blue_labels = set(int(x) for x in team_ids[3:6])
    yellow_label = int(team_ids[6])

    assert len(red_labels) == 1
    assert len(blue_labels) == 1
    assert red_labels | blue_labels == {0, 1}
    assert yellow_label == -1


def test_majority_vote_corrects_single_faulty_frame_and_resets_cleanly() -> None:
    """Sliding-window majority vote suppresses a single misclassified frame."""
    fit_frame = np.full((240, 320, 3), PITCH_BGR, dtype=np.uint8)
    box_a = [30.0, 40.0, 70.0, 140.0]
    box_b = [130.0, 40.0, 170.0, 140.0]
    _draw_player(fit_frame, box_a, RED_BGR)
    _draw_player(fit_frame, box_b, BLUE_BGR)

    clf = TeamClassifier(TeamConfig(vote_window=5, outlier_distance_threshold=45.0))
    clf.fit([(fit_frame, _make_detections([box_a, box_b]))])

    # Determine which cluster ID corresponds to red vs blue
    probe = clf.predict(fit_frame, _make_detections([box_a, box_b]))
    red_id, blue_id = int(probe[0]), int(probe[1])
    assert {red_id, blue_id} == {0, 1}

    # Build a red frame and a single faulty blue frame at the same player location
    red_frame = np.full((240, 320, 3), PITCH_BGR, dtype=np.uint8)
    _draw_player(red_frame, box_a, RED_BGR)

    faulty_blue_frame = np.full((240, 320, 3), PITCH_BGR, dtype=np.uint8)
    _draw_player(faulty_blue_frame, box_a, BLUE_BGR)

    # 3 red frames -> 1 faulty blue frame -> 1 red frame
    sequence = [red_frame, red_frame, red_frame, faulty_blue_frame, red_frame]
    preds: list[int] = []
    for frm in sequence:
        dets = _make_detections([box_a], tracker_ids=[42])
        preds.append(int(clf.predict(frm, dets)[0]))

    assert preds == [red_id, red_id, red_id, red_id, red_id]

    # After reset_history() (e.g., on scene cut), same tracker_id immediately reflects new jersey
    clf.reset_history()
    after_reset = clf.predict(faulty_blue_frame, _make_detections([box_a], tracker_ids=[42]))
    assert int(after_reset[0]) == blue_id


def test_incremental_init_respects_n_init_frames_cap() -> None:
    """add_init_frame stops collecting once n_init_frames frames are reached."""
    frame = np.full((240, 320, 3), PITCH_BGR, dtype=np.uint8)
    boxes = [[30.0, 40.0, 70.0, 140.0], [130.0, 40.0, 170.0, 140.0]]
    _draw_player(frame, boxes[0], RED_BGR)
    _draw_player(frame, boxes[1], BLUE_BGR)
    dets = _make_detections(boxes)

    clf = TeamClassifier(TeamConfig(n_init_frames=3))
    added_counts = [clf.add_init_frame(frame, dets) for _ in range(5)]

    assert added_counts == [2, 2, 2, 0, 0]
    assert clf.init_frames_collected == 3

    clf.fit()
    assert clf.is_fitted
    labels = clf.predict(frame, dets)
    assert set(int(x) for x in labels) == {0, 1}


def test_unfitted_empty_and_untracked_edge_cases() -> None:
    """Classifier handles unfitted state, insufficient samples, empty dets, and missing IDs."""
    frame = np.full((240, 320, 3), PITCH_BGR, dtype=np.uint8)
    box = [30.0, 40.0, 70.0, 140.0]
    _draw_player(frame, box, RED_BGR)

    clf = TeamClassifier(TeamConfig())
    # Before fit -> all -1
    dets = _make_detections([box], tracker_ids=[1])
    pre_fit = clf.predict(frame, dets)
    assert np.array_equal(pre_fit, np.array([-1]))
    assert np.array_equal(dets.data["team_id"], np.array([-1]))

    # Insufficient samples (<2) -> stays unfitted gracefully
    clf.fit([(frame, dets)])
    assert not clf.is_fitted

    # Multiple boxes with identical colors (<2 distinct clusters) -> stays unfitted without warning
    same_color_dets = _make_detections([box, box], tracker_ids=[1, 2])
    clf.fit([(frame, same_color_dets)])
    assert not clf.is_fitted

    # Empty detections -> empty int array, no crash
    empty = sv.Detections.empty()
    empty_pred = clf.predict(frame, empty)
    assert len(empty_pred) == 0
    assert np.issubdtype(empty_pred.dtype, np.integer)
    assert "team_id" in empty.data
    assert len(empty.data["team_id"]) == 0


def test_hsv_color_space_mode() -> None:
    """HSV color space option also separates distinct jersey colors and rejects outliers."""
    frame = np.full((240, 420, 3), PITCH_BGR, dtype=np.uint8)
    red_box = [20.0, 40.0, 60.0, 140.0]
    blue_box = [120.0, 40.0, 160.0, 140.0]
    yellow_box = [220.0, 40.0, 260.0, 140.0]
    _draw_player(frame, red_box, RED_BGR)
    _draw_player(frame, blue_box, BLUE_BGR)
    _draw_player(frame, yellow_box, YELLOW_BGR)

    clf = TeamClassifier(TeamConfig(color_space="hsv", outlier_distance_threshold=20.0))
    clf.fit([(frame, _make_detections([red_box, blue_box]))])

    dets = _make_detections([red_box, blue_box, yellow_box])
    labels = clf.predict(frame, dets)
    assert {int(labels[0]), int(labels[1])} == {0, 1}
    assert int(labels[2]) == -1


def test_referee_class_excluded_from_kmeans_and_always_unknown() -> None:
    """Explicit referee class IDs never enter KMeans and always predict -1 regardless of color."""
    frame = np.full((240, 480, 3), PITCH_BGR, dtype=np.uint8)
    red_box = [20.0, 40.0, 60.0, 140.0]
    blue_box = [120.0, 40.0, 160.0, 140.0]
    # Referee wearing the EXACT same red jersey as Team 0 AND another bright yellow ref
    red_ref_box = [220.0, 40.0, 260.0, 140.0]
    yellow_ref_box = [320.0, 40.0, 360.0, 140.0]
    _draw_player(frame, red_box, RED_BGR)
    _draw_player(frame, blue_box, BLUE_BGR)
    _draw_player(frame, red_ref_box, RED_BGR)
    _draw_player(frame, yellow_ref_box, YELLOW_BGR)

    init_dets = sv.Detections(
        xyxy=np.array([red_box, blue_box, yellow_ref_box], dtype=np.float32),
        confidence=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        class_id=np.array([2, 2, 3], dtype=int),
    )

    clf = TeamClassifier(TeamConfig(), referee_class_ids=[3])
    added = clf.add_init_frame(frame, init_dets)
    # Only the 2 non-referee players enter KMeans init features
    assert added == 2

    clf_no_ref = TeamClassifier(TeamConfig())
    clf_no_ref.add_init_frame(frame, _make_detections([red_box, blue_box]))
    clf.fit()
    clf_no_ref.fit()
    assert clf.cluster_centers is not None
    assert clf_no_ref.cluster_centers is not None
    assert np.allclose(clf.cluster_centers, clf_no_ref.cluster_centers)

    # First prime tracker_id=7 as a red player, then simulate ID switch onto a referee (class_id=3)
    player_det = sv.Detections(
        xyxy=np.array([red_box], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2], dtype=int),
        tracker_id=np.array([7], dtype=int),
    )
    red_team_id = int(clf.predict(frame, player_det)[0])
    assert red_team_id in (0, 1)

    mixed_dets = sv.Detections(
        xyxy=np.array([red_box, blue_box, red_ref_box], dtype=np.float32),
        confidence=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        class_id=np.array([2, 2, 3], dtype=int),
        tracker_id=np.array([1, 2, 7], dtype=int),
    )
    labels = clf.predict(frame, mixed_dets)
    assert int(labels[0]) == red_team_id
    assert int(labels[1]) == 1 - red_team_id
    # Even with identical red color and prior Team-0 vote history on ID 7, referee is immediately -1
    assert int(labels[2]) == -1
    assert np.array_equal(mixed_dets.data["team_id"], labels)
