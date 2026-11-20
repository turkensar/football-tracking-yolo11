"""End-to-end two-pass analysis pipeline tying together all futbolcv modules."""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from futbolcv.annotate import Annotator
from futbolcv.ball import BallTracker, interpolate
from futbolcv.config import AppConfig
from futbolcv.detector import Detector
from futbolcv.export import VideoInfo, export_summary_json, export_tracks_csv
from futbolcv.team import TeamClassifier
from futbolcv.tracker import PlayerTracker
from futbolcv.types import FrameResult
from futbolcv.video_io import VideoReader, VideoWriter

DEFAULT_FALLBACK_FPS = 25.0


def _resolve_source_fps(raw_fps: float, source_name: str) -> float:
    """Validate container FPS and fall back to ``DEFAULT_FALLBACK_FPS`` with a warning."""
    if not math.isfinite(raw_fps) or raw_fps <= 0.0:
        warnings.warn(
            f"Invalid source FPS ({raw_fps!r}) for '{source_name}'; "
            f"falling back to {DEFAULT_FALLBACK_FPS} fps.",
            UserWarning,
            stacklevel=3,
        )
        return DEFAULT_FALLBACK_FPS
    return float(raw_fps)


def _compute_hs_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute an L1-normalized 2D Hue-Saturation histogram for scene-cut detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def _is_scene_cut(
    prev_hist: np.ndarray | None,
    curr_hist: np.ndarray,
    threshold: float | None,
) -> bool:
    """Return True when Bhattacharyya histogram distance indicates a hard camera cut."""
    if prev_hist is None or threshold is None:
        return False
    distance = float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))
    return distance >= threshold


def _interpolate_ball_segments(
    results: list[FrameResult],
    cut_indices: list[int],
    max_gap: int,
) -> None:
    """Interpolate ball gaps independently inside each continuous scene segment."""
    if not results:
        return

    boundaries = [0, *sorted(idx for idx in cut_indices if 0 < idx < len(results)), len(results)]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        segment_balls = [res.ball for res in results[start:end]]
        filled = interpolate(segment_balls, max_gap=max_gap)
        for offset, ball_state in enumerate(filled):
            results[start + offset].ball = ball_state


def _estimate_total_frames(
    frame_count: int,
    *,
    stride: int,
    max_frames: int | None,
) -> int | None:
    """Compute a best-effort frame count hint for ``tqdm`` (never relied on for logic)."""
    estimate = math.ceil(frame_count / stride) if frame_count > 0 else None
    if max_frames is not None:
        return max_frames if estimate is None else min(estimate, max_frames)
    return estimate


def run(
    input_path: str | Path,
    output_dir: str | Path,
    cfg: AppConfig,
    max_frames: int | None = None,
    *,
    detector: Detector | None = None,
    show_progress: bool = True,
) -> Path:
    """Run the full two-pass football analysis pipeline and return the annotated video path.

    Stages:
        0. Team warmup pre-pass on the first ``cfg.team.n_init_frames`` sampled frames
           (detections are cached in memory as lightweight boxes to avoid duplicate YOLO
           inference in Pass 1 while keeping full frames out of RAM).
        1. Pass 1: detection, ByteTrack player tracking, team classification, raw ball
           selection, and histogram-based scene-cut resets (player, ball, team history).
        2. Segmented linear ball interpolation (never crossing scene-cut boundaries).
        3. Pass 2: re-read video frames, annotate overlays, write MP4, and export
           ``tracks.csv`` and ``summary.json``.
    """
    t0 = time.perf_counter()

    src_path = Path(input_path)
    out_dir = Path(output_dir)
    stride = cfg.output.stride

    reader = VideoReader(src_path, max_frames=max_frames, stride=stride)
    source_fps = _resolve_source_fps(reader.fps, src_path.name)
    effective_fps = source_fps / stride

    t_model_start = time.perf_counter()
    det = detector if detector is not None else Detector(cfg.detector)
    model_load_sec = time.perf_counter() - t_model_start

    player_tracker = PlayerTracker(cfg.tracker, fps=effective_fps)
    ball_tracker = BallTracker(cfg.ball)
    team_classifier = TeamClassifier(cfg.team)

    # Pass 1 timer covers warmup inference + tracking/team pass + ball interpolation
    # (every frame runs detection once, excluding model weight loading).
    t_pass1_start = time.perf_counter()

    # Warmup pre-pass: collect jersey samples from initial frames and fit KMeans before frame 0.
    warmup_limit = cfg.team.n_init_frames
    if max_frames is not None:
        warmup_limit = min(warmup_limit, max_frames)

    warmup_cache: dict[int, tuple[sv.Detections, sv.Detections]] = {}
    if warmup_limit > 0:
        warmup_reader = VideoReader(src_path, max_frames=warmup_limit, stride=stride)
        warmup_total = _estimate_total_frames(
            warmup_reader.frame_count, stride=stride, max_frames=warmup_limit
        )
        for frame_idx, frame in tqdm(
            warmup_reader,
            total=warmup_total,
            desc="Warmup (teams)",
            disable=not show_progress,
        ):
            raw_players, raw_balls = det.detect(frame)
            team_classifier.add_init_frame(frame, raw_players)
            warmup_cache[frame_idx] = (raw_players, raw_balls)
        team_classifier.fit()

    # Pass 1: track players & ball, classify teams, detect scene cuts (no frames stored in RAM).
    results: list[FrameResult] = []
    cut_indices: list[int] = []
    prev_hist: np.ndarray | None = None
    est_total = _estimate_total_frames(reader.frame_count, stride=stride, max_frames=max_frames)

    for frame_idx, frame in tqdm(
        reader,
        total=est_total,
        desc="Pass 1/2 (track)",
        disable=not show_progress,
    ):
        curr_hist = (
            _compute_hs_histogram(frame) if cfg.tracker.scene_cut_threshold is not None else None
        )
        if curr_hist is not None and _is_scene_cut(
            prev_hist, curr_hist, cfg.tracker.scene_cut_threshold
        ):
            cut_indices.append(len(results))
            player_tracker.reset()
            ball_tracker.reset()
            team_classifier.reset_history()
        prev_hist = curr_hist

        cached = warmup_cache.pop(frame_idx, None)
        raw_players, raw_balls = cached if cached is not None else det.detect(frame)

        tracked_players = player_tracker.update(raw_players)
        team_classifier.predict(frame, tracked_players)
        ball_state = ball_tracker.update(raw_balls, frame_idx=frame_idx)

        results.append(FrameResult(frame_idx=frame_idx, players=tracked_players, ball=ball_state))

    # Interpolate short ball gaps per continuous scene segment (stride-adjusted gap length).
    effective_max_gap = (
        0 if cfg.ball.max_gap_frames == 0 else max(1, cfg.ball.max_gap_frames // stride)
    )
    _interpolate_ball_segments(results, cut_indices, max_gap=effective_max_gap)
    pass1_sec = time.perf_counter() - t_pass1_start

    # Pass 2: re-read video frames, draw annotations, and write output MP4.
    t_pass2_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / cfg.output.video_filename
    annotator = Annotator(
        cfg.annotate,
        cluster_centers=team_classifier.cluster_centers,
        color_space=cfg.team.color_space,
    )

    results_by_frame = {res.frame_idx: res for res in results}
    with VideoWriter(
        video_path,
        fps=effective_fps,
        size=reader.size,
        codec=cfg.output.codec,
    ) as writer:
        for frame_idx, frame in tqdm(
            reader,
            total=len(results),
            desc="Pass 2/2 (render)",
            disable=not show_progress,
        ):
            res = results_by_frame.get(frame_idx)
            if res is None:
                continue
            annotated = annotator.annotate(frame, res, fps=effective_fps)
            writer.write(annotated)
    pass2_sec = time.perf_counter() - t_pass2_start

    elapsed_sec = time.perf_counter() - t0
    video_info = VideoInfo(
        source=src_path.name,
        width=reader.width,
        height=reader.height,
        source_fps=source_fps,
        effective_fps=effective_fps,
        stride=stride,
    )
    export_tracks_csv(results, out_dir / cfg.output.tracks_filename)
    export_summary_json(
        results,
        video_info,
        out_dir / cfg.output.summary_filename,
        elapsed_sec=elapsed_sec,
        timings={
            "model_load_sec": model_load_sec,
            "pass1_sec": pass1_sec,
            "pass2_sec": pass2_sec,
        },
    )
    return video_path


__all__ = ["DEFAULT_FALLBACK_FPS", "run"]
