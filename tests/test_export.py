"""Unit tests for CSV track export and JSON summary generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from futbolcv.export import (
    CSV_COLUMNS,
    VideoInfo,
    build_summary,
    export_summary_json,
    export_tracks_csv,
)
from futbolcv.types import BallState, FrameResult


def _make_players(
    boxes: list[list[float]],
    tracker_ids: list[int],
    team_ids: list[int],
    confs: list[float],
) -> sv.Detections:
    """Build synthetic tracked player Detections with team_id data."""
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
        data={"team_id": np.array(team_ids, dtype=int)},
    )


def _sample_results() -> list[FrameResult]:
    """Return 4 synthetic frames covering players, unknown team, detected/interp/missing ball."""
    return [
        FrameResult(
            frame_idx=0,
            players=_make_players(
                boxes=[[10, 20, 50, 120], [100, 30, 140, 130]],
                tracker_ids=[1, 2],
                team_ids=[0, 1],
                confs=[0.9, 0.85],
            ),
            ball=BallState(
                xyxy=np.array([200.5, 150.25, 212.5, 162.25], dtype=np.float32),
                conf=0.82,
                interpolated=False,
            ),
        ),
        FrameResult(
            frame_idx=1,
            players=_make_players(
                boxes=[[15, 20, 55, 120], [300, 40, 340, 140]],
                tracker_ids=[1, 3],
                team_ids=[0, -1],
                confs=[0.91, 0.78],
            ),
            ball=BallState(
                xyxy=np.array([210.0, 155.0, 222.0, 167.0], dtype=np.float32),
                conf=0.0,
                interpolated=True,
            ),
        ),
        FrameResult(
            frame_idx=2,
            players=_make_players(
                boxes=[[105, 30, 145, 130]],
                tracker_ids=[2],
                team_ids=[1],
                confs=[0.88],
            ),
            ball=BallState(xyxy=None, conf=0.0, interpolated=False),
        ),
        FrameResult(
            frame_idx=3,
            players=sv.Detections.empty(),
            ball=BallState(xyxy=None, conf=0.0, interpolated=False),
        ),
    ]


def test_export_tracks_csv_schema_row_count_and_empty_ball_ids(tmp_path: Path) -> None:
    """CSV writes exact columns, row count, and empty track_id/team_id cells for ball rows."""
    csv_path = tmp_path / "nested" / "tracks.csv"
    returned = export_tracks_csv(_sample_results(), csv_path)

    assert returned == csv_path
    assert csv_path.is_file()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        rows = list(reader)

    # Frame 0: 2p+1b (3), Frame 1: 2p+1b (3), Frame 2: 1p (1), Frame 3: 0 -> total 7 rows
    assert len(rows) == 7

    player_rows = [r for r in rows if r["kind"] == "player"]
    ball_rows = [r for r in rows if r["kind"] == "ball"]
    assert len(player_rows) == 5
    assert len(ball_rows) == 2

    # First player row values
    assert player_rows[0]["frame"] == "0"
    assert player_rows[0]["track_id"] == "1"
    assert player_rows[0]["team_id"] == "0"
    assert player_rows[0]["x1"] == "10.00"
    assert player_rows[0]["conf"] == "0.9000"
    assert player_rows[0]["interpolated"] == "False"

    # Ball rows MUST leave track_id and team_id empty (not -1)
    det_ball, interp_ball = ball_rows
    assert det_ball["frame"] == "0"
    assert det_ball["track_id"] == ""
    assert det_ball["team_id"] == ""
    assert det_ball["x1"] == "200.50"
    assert det_ball["conf"] == "0.8200"
    assert det_ball["interpolated"] == "False"

    assert interp_ball["frame"] == "1"
    assert interp_ball["track_id"] == ""
    assert interp_ball["team_id"] == ""
    assert interp_ball["conf"] == "0.0000"
    assert interp_ball["interpolated"] == "True"


def test_export_summary_json_fields_and_values(tmp_path: Path) -> None:
    """summary.json contains accurate video info, player counts, ball ratios, and processing FPS."""
    video_info = VideoInfo(
        source="match.mp4",
        width=1280,
        height=720,
        source_fps=25.0,
        effective_fps=25.0,
        stride=1,
    )
    json_path = tmp_path / "nested" / "summary.json"
    returned = export_summary_json(_sample_results(), video_info, json_path, elapsed_sec=2.0)

    assert returned == json_path
    assert json_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["video"] == {
        "source": "match.mp4",
        "width": 1280,
        "height": 720,
        "source_fps": 25.0,
        "effective_fps": 25.0,
        "stride": 1,
    }
    assert data["total_frames"] == 4
    assert data["unique_player_ids"] == 3
    assert data["player_ids_by_team"] == {"0": 1, "1": 1, "-1": 1}
    assert data["ball"] == {
        "detected_frames": 1,
        "interpolated_frames": 1,
        "visible_frames": 2,
        "detected_ratio": 0.25,
        "interpolated_ratio": 0.25,
        "visible_ratio": 0.5,
    }
    assert data["processing"] == {"elapsed_sec": 2.0, "fps": 2.0}


def test_summary_counts_each_track_id_under_its_dominant_team() -> None:
    """A player whose team fluctuated once is counted exactly once under its dominant team."""
    results = [
        FrameResult(0, _make_players([[0, 0, 10, 20]], [7], [-1], [0.9]), BallState()),
        FrameResult(1, _make_players([[0, 0, 10, 20]], [7], [0], [0.9]), BallState()),
        FrameResult(2, _make_players([[0, 0, 10, 20]], [7], [0], [0.9]), BallState()),
    ]
    info = VideoInfo("clip.mp4", 640, 360, 25.0, 25.0)
    summary = build_summary(results, info, elapsed_sec=1.5)

    assert summary["unique_player_ids"] == 1
    assert summary["player_ids_by_team"] == {"0": 1, "1": 0, "-1": 0}
    assert sum(summary["player_ids_by_team"].values()) == summary["unique_player_ids"]


def test_export_empty_results_and_invalid_elapsed(tmp_path: Path) -> None:
    """Empty results produce header-only CSV and zeroed JSON; invalid elapsed_sec raises."""
    csv_path = export_tracks_csv([], tmp_path / "empty.csv")
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows == [list(CSV_COLUMNS)]

    info = VideoInfo("empty.mp4", 320, 240, 25.0, 25.0)
    summary = build_summary([], info, elapsed_sec=0.0)
    assert summary["total_frames"] == 0
    assert summary["unique_player_ids"] == 0
    assert summary["ball"]["visible_ratio"] == 0.0
    assert summary["processing"]["fps"] == 0.0

    for bad_elapsed in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="elapsed_sec"):
            build_summary([], info, elapsed_sec=bad_elapsed)  # type: ignore[arg-type]


def test_export_handles_players_missing_optional_fields(tmp_path: Path) -> None:
    """Players lacking tracker_id, team_id, or confidence export cleanly without crashing."""
    untagged = sv.Detections(xyxy=np.array([[5.0, 10.0, 25.0, 60.0]], dtype=np.float32))
    csv_path = export_tracks_csv([FrameResult(0, untagged, BallState())], tmp_path / "untagged.csv")

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    assert rows[0]["track_id"] == ""
    assert rows[0]["team_id"] == "-1"
    assert rows[0]["conf"] == ""
    assert rows[0]["kind"] == "player"
