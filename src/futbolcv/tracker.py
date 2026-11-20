"""Multi-object player tracking wrapper around supervision ByteTrack."""

# Pinned to supervision==0.30.x because sv.ByteTrack was removed in 0.31.0 (moved to trackers).

from __future__ import annotations

import math
import warnings

import numpy as np
import supervision as sv

from futbolcv.config import TrackerConfig


class PlayerTracker:
    """Persistent-ID player tracker backed by ``sv.ByteTrack``."""

    def __init__(self, cfg: TrackerConfig, fps: float) -> None:
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps):
            raise ValueError(f"'fps' must be a finite positive number, got {fps!r}")
        if float(fps) <= 0.0:
            raise ValueError(f"'fps' must be positive, got {fps!r}")

        self.cfg = cfg
        self.frame_rate = float(cfg.frame_rate) if cfg.frame_rate is not None else float(fps)
        self._tracker = self._create_tracker()

    def _create_tracker(self) -> sv.ByteTrack:
        """Instantiate ``sv.ByteTrack`` while locally suppressing its deprecation FutureWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return sv.ByteTrack(
                track_activation_threshold=self.cfg.track_activation_threshold,
                lost_track_buffer=self.cfg.lost_track_buffer,
                minimum_matching_threshold=self.cfg.minimum_matching_threshold,
                frame_rate=self.frame_rate,
                minimum_consecutive_frames=self.cfg.minimum_consecutive_frames,
            )

    def update(self, dets: sv.Detections) -> sv.Detections:
        """Update tracks with current-frame player detections and return tracked detections."""
        if len(dets) == 0 and dets.confidence is None:
            dets = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int),
            )
        return self._tracker.update_with_detections(dets)

    def reset(self) -> None:
        """Clear active/lost tracks and restart tracker IDs from scratch."""
        self._tracker.reset()


__all__ = ["PlayerTracker"]
