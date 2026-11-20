"""CSV and JSON exporters for per-frame tracks and run summary statistics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from futbolcv.types import FrameResult

_TIMING_KEYS: tuple[str, ...] = ("model_load_sec", "pass1_sec", "pass2_sec")


def _validate_non_negative_sec(name: str, value: Any) -> float:
    """Validate a non-negative finite duration in seconds."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or float(value) < 0.0
    ):
        raise ValueError(f"'{name}' must be a non-negative finite number, got {value!r}")
    return float(value)


CSV_COLUMNS: tuple[str, ...] = (
    "frame",
    "track_id",
    "team_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "conf",
    "kind",
    "interpolated",
)


@dataclass(frozen=True)
class VideoInfo:
    """Source and effective video metadata included in ``summary.json``."""

    source: str
    width: int
    height: int
    source_fps: float
    effective_fps: float
    stride: int = 1


def _dominant_team(labels: list[int]) -> int:
    """Return the most frequent team label for one track, preferring valid teams on ties."""
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    if len(tied) == 1:
        return tied[0]
    known = [label for label in tied if label != -1]
    return min(known) if known else -1


def export_tracks_csv(results: Sequence[FrameResult], path: str | Path) -> Path:
    """Write per-frame player and ball tracks to a UTF-8 CSV file.

    Ball rows leave ``track_id`` and ``team_id`` empty (not ``-1``) so downstream
    analytics tools such as pandas do not conflate ball entries with player IDs.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)

        for res in results:
            players = res.players
            count = len(players)
            tracker_ids = players.tracker_id
            team_ids = players.data.get("team_id")
            confidences = players.confidence

            for idx in range(count):
                x1, y1, x2, y2 = (float(v) for v in players.xyxy[idx])
                tid_cell: int | str = "" if tracker_ids is None else int(tracker_ids[idx])
                team_cell = -1 if team_ids is None else int(team_ids[idx])
                conf_cell = "" if confidences is None else f"{float(confidences[idx]):.4f}"
                writer.writerow(
                    [
                        int(res.frame_idx),
                        tid_cell,
                        team_cell,
                        f"{x1:.2f}",
                        f"{y1:.2f}",
                        f"{x2:.2f}",
                        f"{y2:.2f}",
                        conf_cell,
                        "player",
                        False,
                    ]
                )

            if res.ball.xyxy is not None:
                bx1, by1, bx2, by2 = (float(v) for v in res.ball.xyxy)
                writer.writerow(
                    [
                        int(res.frame_idx),
                        "",
                        "",
                        f"{bx1:.2f}",
                        f"{by1:.2f}",
                        f"{bx2:.2f}",
                        f"{by2:.2f}",
                        f"{float(res.ball.conf):.4f}",
                        "ball",
                        bool(res.ball.interpolated),
                    ]
                )

    return out_path


def build_summary(
    results: Sequence[FrameResult],
    video_info: VideoInfo,
    *,
    elapsed_sec: float,
    timings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute structured run statistics from ``FrameResult`` records."""
    elapsed = _validate_non_negative_sec("elapsed_sec", elapsed_sec)

    stage_secs: dict[str, float] = dict.fromkeys(_TIMING_KEYS, 0.0)
    if timings is not None:
        unknown_keys = set(timings.keys()) - set(_TIMING_KEYS)
        if unknown_keys:
            unknown_str = ", ".join(sorted(str(k) for k in unknown_keys))
            raise ValueError(
                f"Unknown timing key(s): {unknown_str}. Valid keys: {', '.join(_TIMING_KEYS)}"
            )
        for key, value in timings.items():
            stage_secs[key] = _validate_non_negative_sec(f"timings.{key}", value)

    total_frames = len(results)
    team_labels_per_track: dict[int, list[int]] = defaultdict(list)
    ball_detected_frames = 0
    ball_interpolated_frames = 0

    for res in results:
        players = res.players
        tracker_ids = players.tracker_id
        team_ids = players.data.get("team_id")
        if tracker_ids is not None:
            for idx in range(len(players)):
                tid = int(tracker_ids[idx])
                team_id = -1 if team_ids is None else int(team_ids[idx])
                team_labels_per_track[tid].append(team_id)

        if res.ball.xyxy is not None:
            if res.ball.interpolated:
                ball_interpolated_frames += 1
            else:
                ball_detected_frames += 1

    player_ids_by_team: dict[str, int] = {"0": 0, "1": 0, "-1": 0}
    for labels in team_labels_per_track.values():
        dominant = _dominant_team(labels)
        key = str(dominant) if dominant in (0, 1) else "-1"
        player_ids_by_team[key] += 1

    ball_visible_frames = ball_detected_frames + ball_interpolated_frames
    if total_frames > 0:
        detected_ratio = round(ball_detected_frames / total_frames, 4)
        interpolated_ratio = round(ball_interpolated_frames / total_frames, 4)
        visible_ratio = round(ball_visible_frames / total_frames, 4)
    else:
        detected_ratio = 0.0
        interpolated_ratio = 0.0
        visible_ratio = 0.0

    pass1_sec = stage_secs["pass1_sec"]
    pass1_fps = round(total_frames / pass1_sec, 2) if pass1_sec > 0.0 else 0.0

    return {
        "video": {
            "source": video_info.source,
            "width": int(video_info.width),
            "height": int(video_info.height),
            "source_fps": round(float(video_info.source_fps), 4),
            "effective_fps": round(float(video_info.effective_fps), 4),
            "stride": int(video_info.stride),
        },
        "total_frames": total_frames,
        "unique_player_ids": len(team_labels_per_track),
        "player_ids_by_team": player_ids_by_team,
        "ball": {
            "detected_frames": ball_detected_frames,
            "interpolated_frames": ball_interpolated_frames,
            "visible_frames": ball_visible_frames,
            "detected_ratio": detected_ratio,
            "interpolated_ratio": interpolated_ratio,
            "visible_ratio": visible_ratio,
        },
        "processing": {
            "elapsed_sec": round(elapsed, 4),
            "model_load_sec": round(stage_secs["model_load_sec"], 4),
            "pass1_sec": round(pass1_sec, 4),
            "pass1_fps": pass1_fps,
            "pass2_sec": round(stage_secs["pass2_sec"], 4),
        },
    }


def export_summary_json(
    results: Sequence[FrameResult],
    video_info: VideoInfo,
    path: str | Path,
    *,
    elapsed_sec: float,
    timings: Mapping[str, float] | None = None,
) -> Path:
    """Build and write run statistics to a UTF-8 ``summary.json`` file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(results, video_info, elapsed_sec=elapsed_sec, timings=timings)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out_path


__all__ = [
    "CSV_COLUMNS",
    "VideoInfo",
    "build_summary",
    "export_summary_json",
    "export_tracks_csv",
]
