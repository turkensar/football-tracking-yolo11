"""Shared data contracts passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv


@dataclass
class BallState:
    """Ball bounding box and provenance for a single video frame."""

    xyxy: np.ndarray | None = None
    conf: float = 0.0
    interpolated: bool = False


@dataclass
class FrameResult:
    """Per-frame tracking and team-classification outputs."""

    frame_idx: int
    players: sv.Detections
    ball: BallState


__all__ = ["BallState", "FrameResult"]
