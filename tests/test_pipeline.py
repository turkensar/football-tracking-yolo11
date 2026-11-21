"""Integration tests for the two-pass analysis pipeline with a mocked detector."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from futbolcv.config import AppConfig
from futbolcv.pipeline import DEFAULT_FALLBACK_FPS, run
from futbolcv.video_io import VideoReader

PITCH_BGR = (40, 120, 40)
RED_BGR = (20, 20, 220)
BLUE_BGR = (220, 50, 20)


def _write_match_clip(
    path: Path,
    *,
    num_frames: int = 10,
    fps: float = 25.0,
    width: int = 320,
    height: int = 240,
    scene_cut_at: int | None = None,
) -> Path:
    """Create a synthetic video with one red player, one blue player, and optional scene cut."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    try:
        for i in range(num_frames):
            bg = (220, 40, 200) if (scene_cut_at is not None and i >= scene_cut_at) else PITCH_BGR
            frame = np.full((height, width, 3), bg, dtype=np.uint8)
            shift = i * 4
            # Red player box: [30+shift, 50, 70+shift, 150]
            cv2.rectangle(frame, (30 + shift, 50), (70 + shift, 100), RED_BGR, -1)
            # Blue player box: [160+shift, 50, 200+shift, 150]
            cv2.rectangle(frame, (160 + shift, 50), (200 + shift, 100), BLUE_BGR, -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


class _MockDetector:
    """Deterministic fake Detector emitting 2 moving players and a ball with controlled gaps."""

    def __init__(self, *, missing_ball_calls: set[int] | None = None) -> None:
        self.missing_ball_calls = missing_ball_calls or set()
        self.call_count = 0

    def detect(self, frame: np.ndarray) -> tuple[sv.Detections, sv.Detections]:
        idx = self.call_count
        self.call_count += 1
        shift = idx * 4.0

        players = sv.Detections(
            xyxy=np.array(
                [
                    [30.0 + shift, 50.0, 70.0 + shift, 150.0],
                    [160.0 + shift, 50.0, 200.0 + shift, 150.0],
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.92, 0.89], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )

        if idx in self.missing_ball_calls:
            balls = sv.Detections.empty()
        else:
            bx = 100.0 + idx * 8.0
            balls = sv.Detections(
                xyxy=np.array([[bx, 120.0, bx + 12.0, 132.0]], dtype=np.float32),
                confidence=np.array([0.75], dtype=np.float32),
                class_id=np.array([32], dtype=int),
            )
        return players, balls


def test_pipeline_run_produces_mp4_csv_json_and_warms_up_teams(tmp_path: Path) -> None:
    """End-to-end mocked run creates mp4, csv, json with warmed-up frame-0 teams and ball interp."""
    video_path = _write_match_clip(tmp_path / "clip.mp4", num_frames=10, fps=25.0)
    out_dir = tmp_path / "outputs"
    cfg = AppConfig()
    cfg.team.n_init_frames = 5
    cfg.ball.max_gap_frames = 5

    # Missing ball on frames 4 and 5 -> should be linearly interpolated
    mock_det = _MockDetector(missing_ball_calls={4, 5})
    result_video = run(video_path, out_dir, cfg, detector=mock_det, show_progress=False)

    # Detector called once per unique frame (warmup detections reused in Pass 1)
    assert mock_det.call_count == 10

    # Annotated video exists and matches dimensions, fps, and frame count
    assert result_video == out_dir / cfg.output.video_filename
    assert result_video.is_file()
    out_reader = VideoReader(result_video)
    assert out_reader.size == (320, 240)
    assert out_reader.fps == pytest.approx(25.0)
    assert len(list(out_reader)) == 10

    # tracks.csv exists and includes frame 0 valid team assignments + interpolated ball rows
    csv_path = out_dir / cfg.output.tracks_filename
    assert csv_path.is_file()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    frame0_players = [r for r in rows if r["frame"] == "0" and r["kind"] == "player"]
    assert len(frame0_players) == 2
    # Warmup fitted before frame 0 -> teams are {0, 1}, NOT -1
    assert {int(r["team_id"]) for r in frame0_players} == {0, 1}

    ball_rows = [r for r in rows if r["kind"] == "ball"]
    assert len(ball_rows) == 10
    interp_frames = [int(r["frame"]) for r in ball_rows if r["interpolated"] == "True"]
    assert interp_frames == [4, 5]

    # summary.json exists, leaks only filename (not full path), and reports exact stats
    summary_path = out_dir / cfg.output.summary_filename
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["video"]["source"] == "clip.mp4"
    assert summary["video"]["source_fps"] == pytest.approx(25.0)
    assert summary["video"]["effective_fps"] == pytest.approx(25.0)
    assert summary["total_frames"] == 10
    assert summary["unique_player_ids"] == 2
    assert summary["player_ids_by_team"] == {"0": 1, "1": 1, "-1": 0}
    assert summary["ball"]["detected_frames"] == 8
    assert summary["ball"]["interpolated_frames"] == 2
    assert summary["ball"]["visible_ratio"] == pytest.approx(1.0)

    proc = summary["processing"]
    assert set(proc.keys()) == {
        "elapsed_sec",
        "model_load_sec",
        "pass1_sec",
        "pass1_fps",
        "pass2_sec",
    }
    assert proc["elapsed_sec"] >= 0.0
    assert proc["model_load_sec"] >= 0.0
    assert proc["pass1_sec"] > 0.0
    assert proc["pass1_fps"] > 0.0
    assert proc["pass2_sec"] >= 0.0


def test_pipeline_stride_scales_effective_fps_and_interpolation_gap(tmp_path: Path) -> None:
    """stride=2 halves effective_fps and scales max_gap_frames // stride."""
    video_path = _write_match_clip(tmp_path / "strided.mp4", num_frames=10, fps=30.0)
    out_dir = tmp_path / "out_stride"
    cfg = AppConfig()
    cfg.output.stride = 2
    # max_gap_frames=3 with stride=2 -> effective_max_gap = max(1, 3 // 2) = 1 sampled step
    cfg.ball.max_gap_frames = 3
    cfg.team.n_init_frames = 3

    # 5 sampled frames (source indices 0, 2, 4, 6, 8); missing calls 1 & 2 (2 sampled steps > 1)
    mock_det = _MockDetector(missing_ball_calls={1, 2})
    result_video = run(video_path, out_dir, cfg, detector=mock_det, show_progress=False)

    out_reader = VideoReader(result_video)
    assert out_reader.fps == pytest.approx(15.0)
    assert len(list(out_reader)) == 5

    summary = json.loads((out_dir / cfg.output.summary_filename).read_text(encoding="utf-8"))
    assert summary["video"]["effective_fps"] == pytest.approx(15.0)
    assert summary["video"]["stride"] == 2
    assert summary["total_frames"] == 5
    # 2-step gap exceeds effective_max_gap=1 -> NOT interpolated
    assert summary["ball"]["interpolated_frames"] == 0
    assert summary["ball"]["detected_frames"] == 3


def test_pipeline_scene_cut_resets_trackers_and_blocks_cross_cut_interpolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hard camera cut resets player/ball/team state and prevents cross-cut ball interpolation."""
    video_path = _write_match_clip(tmp_path / "cut.mp4", num_frames=10, fps=25.0, scene_cut_at=5)
    out_dir = tmp_path / "out_cut"
    cfg = AppConfig()
    cfg.team.n_init_frames = 4
    cfg.ball.max_gap_frames = 5
    cfg.tracker.scene_cut_threshold = 0.5

    reset_events: list[str] = []
    orig_player_reset = __import__(
        "futbolcv.tracker", fromlist=["PlayerTracker"]
    ).PlayerTracker.reset
    orig_ball_reset = __import__("futbolcv.ball", fromlist=["BallTracker"]).BallTracker.reset
    orig_team_reset = __import__(
        "futbolcv.team", fromlist=["TeamClassifier"]
    ).TeamClassifier.reset_history

    def spy_player_reset(self: object) -> None:
        reset_events.append("player")
        orig_player_reset(self)

    def spy_ball_reset(self: object) -> None:
        reset_events.append("ball")
        orig_ball_reset(self)

    def spy_team_reset(self: object) -> None:
        reset_events.append("team")
        orig_team_reset(self)

    monkeypatch.setattr("futbolcv.tracker.PlayerTracker.reset", spy_player_reset)
    monkeypatch.setattr("futbolcv.ball.BallTracker.reset", spy_ball_reset)
    monkeypatch.setattr("futbolcv.team.TeamClassifier.reset_history", spy_team_reset)

    # Frame 4 has ball, frame 5 (first frame of new scene) is missing ball, frame 6 has ball.
    # Because cut occurs at index 5, frame 5 is a leading gap of segment 2 -> MUST NOT interpolate.
    mock_det = _MockDetector(missing_ball_calls={5})
    run(video_path, out_dir, cfg, detector=mock_det, show_progress=False)

    assert "player" in reset_events
    assert "ball" in reset_events
    # Team reset_history runs once inside fit() and once at the scene cut (>= 2 times)
    assert reset_events.count("team") >= 2

    with (out_dir / cfg.output.tracks_filename).open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    ball_frames = {int(r["frame"]) for r in rows if r["kind"] == "ball"}
    assert 5 not in ball_frames


def test_pipeline_invalid_fps_warns_and_falls_back_to_25(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When container reports fps=0, pipeline emits UserWarning and uses 25.0 fps."""
    video_path = _write_match_clip(tmp_path / "zero_fps.mp4", num_frames=4, fps=25.0)
    out_dir = tmp_path / "out_fallback"
    cfg = AppConfig()
    cfg.team.n_init_frames = 2

    monkeypatch.setattr(VideoReader, "fps", property(lambda self: 0.0))

    with pytest.warns(UserWarning, match="falling back to 25.0"):
        result_video = run(
            video_path,
            out_dir,
            cfg,
            detector=_MockDetector(),
            show_progress=False,
        )

    monkeypatch.undo()
    assert VideoReader(result_video).fps == pytest.approx(DEFAULT_FALLBACK_FPS)
    summary = json.loads((out_dir / cfg.output.summary_filename).read_text(encoding="utf-8"))
    assert summary["video"]["source_fps"] == pytest.approx(DEFAULT_FALLBACK_FPS)
    assert summary["video"]["effective_fps"] == pytest.approx(DEFAULT_FALLBACK_FPS)


def test_pipeline_max_frames_caps_processing(tmp_path: Path) -> None:
    """max_frames limits both passes and exported outputs."""
    video_path = _write_match_clip(tmp_path / "capped.mp4", num_frames=10, fps=25.0)
    out_dir = tmp_path / "out_cap"
    cfg = AppConfig()
    cfg.team.n_init_frames = 4

    result_video = run(
        video_path,
        out_dir,
        cfg,
        max_frames=4,
        detector=_MockDetector(),
        show_progress=False,
    )
    assert len(list(VideoReader(result_video))) == 4
    summary = json.loads((out_dir / cfg.output.summary_filename).read_text(encoding="utf-8"))
    assert summary["total_frames"] == 4


def test_pipeline_football_config_keeps_referee_unknown(tmp_path: Path) -> None:
    """End-to-end pipeline with football class IDs tracks referees and labels them team -1."""
    video_path = tmp_path / "football_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, (320, 240))
    assert writer.isOpened()
    try:
        for i in range(6):
            frame = np.full((240, 320, 3), PITCH_BGR, dtype=np.uint8)
            shift = i * 4
            # Red player, blue player, and red-wearing referee (non-overlapping boxes)
            cv2.rectangle(frame, (20 + shift, 50), (60 + shift, 100), RED_BGR, -1)
            cv2.rectangle(frame, (120 + shift, 50), (160 + shift, 100), BLUE_BGR, -1)
            cv2.rectangle(frame, (220 + shift, 50), (260 + shift, 100), RED_BGR, -1)
            writer.write(frame)
    finally:
        writer.release()

    out_dir = tmp_path / "out_football"
    cfg = AppConfig()
    cfg.detector.player_class_ids = [1, 2]
    cfg.detector.referee_class_ids = [3]
    cfg.detector.ball_class_ids = [0]
    cfg.team.n_init_frames = 3

    class _FootballMockDetector:
        def __init__(self) -> None:
            self._call = 0

        def detect(self, frame: np.ndarray) -> tuple[sv.Detections, sv.Detections]:
            idx = self._call
            self._call += 1
            shift = idx * 4.0
            players = sv.Detections(
                xyxy=np.array(
                    [
                        [20.0 + shift, 50.0, 60.0 + shift, 150.0],
                        [120.0 + shift, 50.0, 160.0 + shift, 150.0],
                        [220.0 + shift, 50.0, 260.0 + shift, 150.0],
                    ],
                    dtype=np.float32,
                ),
                confidence=np.array([0.92, 0.90, 0.88], dtype=np.float32),
                class_id=np.array([2, 2, 3], dtype=int),
            )
            balls = sv.Detections.empty()
            return players, balls

    run(
        video_path,
        out_dir,
        cfg,
        detector=_FootballMockDetector(),  # type: ignore[arg-type]
        show_progress=False,
    )

    summary = json.loads((out_dir / cfg.output.summary_filename).read_text(encoding="utf-8"))
    assert summary["unique_player_ids"] == 3
    assert summary["player_ids_by_team"] == {"0": 1, "1": 1, "-1": 1}
