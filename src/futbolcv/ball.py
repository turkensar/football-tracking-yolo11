"""Ball candidate selection and linear gap interpolation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import supervision as sv

from futbolcv.config import BallConfig
from futbolcv.types import BallState


def _box_center(xyxy: np.ndarray) -> np.ndarray:
    """Return the ``(cx, cy)`` center of an ``[x1, y1, x2, y2]`` bounding box."""
    return np.array(
        [(float(xyxy[0]) + float(xyxy[2])) * 0.5, (float(xyxy[1]) + float(xyxy[3])) * 0.5],
        dtype=np.float64,
    )


def _clone_state(state: BallState) -> BallState:
    """Return a shallow copy of ``BallState`` with a copied ``xyxy`` array."""
    return BallState(
        xyxy=None if state.xyxy is None else np.asarray(state.xyxy, dtype=np.float32).copy(),
        conf=float(state.conf),
        interpolated=bool(state.interpolated),
    )


class BallTracker:
    """Selects the best ball detection per frame with per-frame jump gating."""

    def __init__(self, cfg: BallConfig) -> None:
        self.cfg = cfg
        self._last_xyxy: np.ndarray | None = None
        self._last_frame_idx: int | None = None
        self._step_counter: int = 0

    def reset(self) -> None:
        """Clear previous ball position history."""
        self._last_xyxy = None
        self._last_frame_idx = None
        self._step_counter = 0

    def select(
        self,
        candidates: sv.Detections,
        prev_xyxy: np.ndarray | None,
        *,
        frames_since_last: int = 1,
    ) -> BallState:
        """Pick one ball candidate using proximity to ``prev_xyxy`` or highest confidence.

        When ``prev_xyxy`` is provided, picks the closest candidate (breaking ties
        by higher confidence) and rejects jumps exceeding
        ``cfg.max_jump_px * frames_since_last``. When ``prev_xyxy`` is ``None``,
        picks the highest-confidence candidate.
        """
        if (
            isinstance(frames_since_last, bool)
            or not isinstance(frames_since_last, int)
            or frames_since_last < 1
        ):
            raise ValueError(
                f"'frames_since_last' must be an integer >= 1, got {frames_since_last!r}"
            )

        if len(candidates) == 0:
            return BallState(xyxy=None, conf=0.0, interpolated=False)

        confidences = (
            np.asarray(candidates.confidence, dtype=np.float64)
            if candidates.confidence is not None
            else np.zeros(len(candidates), dtype=np.float64)
        )

        if prev_xyxy is None:
            best_idx = int(np.argmax(confidences))
            return BallState(
                xyxy=np.asarray(candidates.xyxy[best_idx], dtype=np.float32).copy(),
                conf=float(confidences[best_idx]),
                interpolated=False,
            )

        prev_center = _box_center(prev_xyxy)
        centers = np.column_stack(
            (
                (candidates.xyxy[:, 0] + candidates.xyxy[:, 2]) * 0.5,
                (candidates.xyxy[:, 1] + candidates.xyxy[:, 3]) * 0.5,
            )
        ).astype(np.float64)
        distances = np.linalg.norm(centers - prev_center, axis=1)

        # Sort by smallest distance first, breaking exact ties by highest confidence.
        order = np.lexsort((-confidences, distances))
        best_idx = int(order[0])

        allowed_jump = self.cfg.max_jump_px * frames_since_last
        if float(distances[best_idx]) > allowed_jump:
            return BallState(xyxy=None, conf=0.0, interpolated=False)

        return BallState(
            xyxy=np.asarray(candidates.xyxy[best_idx], dtype=np.float32).copy(),
            conf=float(confidences[best_idx]),
            interpolated=False,
        )

    def update(self, candidates: sv.Detections, *, frame_idx: int | None = None) -> BallState:
        """Stateful wrapper around ``select`` tracking elapsed frames since last sight."""
        current_idx = self._step_counter if frame_idx is None else frame_idx
        self._step_counter += 1

        if self._last_xyxy is None or self._last_frame_idx is None:
            frames_since_last = 1
        else:
            frames_since_last = max(1, current_idx - self._last_frame_idx)

        state = self.select(candidates, self._last_xyxy, frames_since_last=frames_since_last)
        if state.xyxy is not None:
            self._last_xyxy = state.xyxy.copy()
            self._last_frame_idx = current_idx
        return state


def interpolate(states: Sequence[BallState], max_gap: int) -> list[BallState]:
    """Linearly interpolate missing ball boxes for interior gaps up to ``max_gap`` frames.

    Leading/trailing gaps and interior gaps strictly longer than ``max_gap`` remain
    ``BallState(xyxy=None, conf=0.0, interpolated=False)``. Interpolated states have
    ``conf=0.0`` and ``interpolated=True``. Input states are not mutated.
    """
    if isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap < 0:
        raise ValueError(f"'max_gap' must be a non-negative integer, got {max_gap!r}")

    result = [_clone_state(s) for s in states]
    known_indices = [i for i, s in enumerate(result) if s.xyxy is not None]
    if len(known_indices) < 2 or max_gap == 0:
        return result

    for start_idx, end_idx in zip(known_indices, known_indices[1:], strict=False):
        gap = end_idx - start_idx - 1
        if gap < 1 or gap > max_gap:
            continue

        start_box = result[start_idx].xyxy
        end_box = result[end_idx].xyxy
        assert start_box is not None and end_box is not None
        start_f64 = start_box.astype(np.float64)
        end_f64 = end_box.astype(np.float64)

        span = end_idx - start_idx
        for step in range(1, span):
            alpha = step / span
            interp_box = ((1.0 - alpha) * start_f64 + alpha * end_f64).astype(np.float32)
            result[start_idx + step] = BallState(xyxy=interp_box, conf=0.0, interpolated=True)

    return result


__all__ = ["BallTracker", "interpolate"]
