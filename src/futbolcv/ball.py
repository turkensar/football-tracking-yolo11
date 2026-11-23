"""Ball candidate selection, static false-positive suppression, and linear interpolation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass
class _StaticSpot:
    """Tracks a persistent low-motion candidate location (e.g. painted penalty spot)."""

    center: np.ndarray
    hits: int
    last_seen: int


class BallTracker:
    """Selects the best ball detection per frame with jump gating and static-spot recovery."""

    def __init__(self, cfg: BallConfig) -> None:
        self.cfg = cfg
        self._last_xyxy: np.ndarray | None = None
        self._last_frame_idx: int | None = None
        self._last_was_static: bool = False
        self._step_counter: int = 0
        self._static_spots: list[_StaticSpot] = []

    def reset(self) -> None:
        """Clear previous ball position history and tracked static spots."""
        self._last_xyxy = None
        self._last_frame_idx = None
        self._last_was_static = False
        self._step_counter = 0
        self._static_spots.clear()

    def _update_static_spots(self, centers: np.ndarray, current_idx: int) -> np.ndarray:
        """Update persistent spot memory and return a boolean mask of static candidates."""
        # Forget spots that have not been seen for more than static_frames frames.
        self._static_spots = [
            spot
            for spot in self._static_spots
            if (current_idx - spot.last_seen) <= self.cfg.static_frames
        ]

        count = len(centers)
        static_mask = np.zeros(count, dtype=bool)
        if count == 0:
            return static_mask

        # Prioritize established spots so they keep claiming their own repeated detections.
        self._static_spots.sort(key=lambda s: -s.hits)
        matched_cands: set[int] = set()
        radius = self.cfg.static_radius_px

        for spot in self._static_spots:
            best_idx: int | None = None
            best_dist = float("inf")
            for idx in range(count):
                if idx in matched_cands:
                    continue
                dist = float(np.linalg.norm(centers[idx] - spot.center))
                if dist <= radius and dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None:
                matched_cands.add(best_idx)
                spot.hits += 1
                spot.last_seen = current_idx
                spot.center = centers[best_idx].copy()
                if spot.hits >= self.cfg.static_frames:
                    static_mask[best_idx] = True

        for idx in range(count):
            if idx not in matched_cands:
                self._static_spots.append(
                    _StaticSpot(center=centers[idx].copy(), hits=1, last_seen=current_idx)
                )

        return static_mask

    def _is_on_static_spot(self) -> bool:
        """Return True if the current track sits on an established static spot."""
        if self._last_was_static:
            return True
        if self._last_xyxy is None:
            return False
        prev_center = _box_center(self._last_xyxy)
        for spot in self._static_spots:
            if (
                spot.hits >= self.cfg.static_frames
                and float(np.linalg.norm(prev_center - spot.center)) <= self.cfg.static_radius_px
            ):
                return True
        return False

    def _select_index(
        self,
        candidates: sv.Detections,
        prev_xyxy: np.ndarray | None,
        *,
        frames_since_last: int = 1,
        static_mask: np.ndarray | None = None,
        on_static_spot: bool = False,
        use_score: bool = False,
    ) -> int | None:
        """Return the winning candidate index or ``None`` if all candidates are rejected."""
        if (
            isinstance(frames_since_last, bool)
            or not isinstance(frames_since_last, int)
            or frames_since_last < 1
        ):
            raise ValueError(
                f"'frames_since_last' must be an integer >= 1, got {frames_since_last!r}"
            )

        count = len(candidates)
        if count == 0:
            return None

        confidences = (
            np.asarray(candidates.confidence, dtype=np.float64)
            if candidates.confidence is not None
            else np.zeros(count, dtype=np.float64)
        )

        # Preserve original proximity-first contract when called without T13d scoring/static info.
        if not use_score and static_mask is None:
            if prev_xyxy is None:
                return int(np.argmax(confidences))

            prev_center = _box_center(prev_xyxy)
            centers = np.column_stack(
                (
                    (candidates.xyxy[:, 0] + candidates.xyxy[:, 2]) * 0.5,
                    (candidates.xyxy[:, 1] + candidates.xyxy[:, 3]) * 0.5,
                )
            ).astype(np.float64)
            distances = np.linalg.norm(centers - prev_center, axis=1)
            order = np.lexsort((-confidences, distances))
            best_idx = int(order[0])
            allowed_jump = self.cfg.max_jump_px * frames_since_last
            if float(distances[best_idx]) > allowed_jump:
                return None
            return best_idx

        is_static = (
            np.asarray(static_mask, dtype=bool)
            if static_mask is not None
            else np.zeros(count, dtype=bool)
        )
        if len(is_static) != count:
            raise ValueError(
                f"'static_mask' length ({len(is_static)}) must match candidates ({count})"
            )

        allowed_jump = self.cfg.max_jump_px * frames_since_last
        if prev_xyxy is None:
            distances = np.zeros(count, dtype=np.float64)
            in_gate = np.ones(count, dtype=bool)
        else:
            prev_center = _box_center(prev_xyxy)
            centers = np.column_stack(
                (
                    (candidates.xyxy[:, 0] + candidates.xyxy[:, 2]) * 0.5,
                    (candidates.xyxy[:, 1] + candidates.xyxy[:, 3]) * 0.5,
                )
            ).astype(np.float64)
            distances = np.linalg.norm(centers - prev_center, axis=1)
            in_gate = distances <= allowed_jump

        scores = confidences - self.cfg.distance_weight * (distances / allowed_jump)

        def _best_by_score(indices: list[int]) -> int:
            return max(
                indices,
                key=lambda i: (float(scores[i]), -float(distances[i]), float(confidences[i])),
            )

        moving_indices = [i for i in range(count) if not is_static[i]]
        moving_in_gate = [i for i in moving_indices if in_gate[i]]

        # (a) Prefer moving (non-static) candidates inside the jump gate by combined score.
        if moving_in_gate:
            return _best_by_score(moving_in_gate)

        # (b) Reacquire a high-confidence moving candidate outside the gate if locked on a spot.
        if on_static_spot:
            reacquire_cands = [
                i for i in moving_indices if float(confidences[i]) >= self.cfg.reacquire_conf
            ]
            if reacquire_cands:
                return max(
                    reacquire_cands,
                    key=lambda i: (float(confidences[i]), -float(distances[i])),
                )

        # (c) Fall back to static candidates inside the jump gate (preserves stationary set pieces).
        static_in_gate = [i for i in range(count) if is_static[i] and in_gate[i]]
        if static_in_gate:
            return _best_by_score(static_in_gate)

        return None

    def select(
        self,
        candidates: sv.Detections,
        prev_xyxy: np.ndarray | None,
        *,
        frames_since_last: int = 1,
        static_mask: np.ndarray | None = None,
        on_static_spot: bool = False,
        use_score: bool = False,
    ) -> BallState:
        """Pick one ball candidate using proximity/score gating and optional static suppression.

        When called with default keyword arguments (``static_mask=None, use_score=False``),
        picks the closest candidate to ``prev_xyxy`` (breaking ties by confidence) within
        ``cfg.max_jump_px * frames_since_last``, or highest confidence when ``prev_xyxy`` is
        ``None``. When ``use_score=True`` (used by ``update()``), applies combined
        ``conf - distance_weight * (dist / allowed_jump)`` scoring and static-spot recovery.
        """
        best_idx = self._select_index(
            candidates,
            prev_xyxy,
            frames_since_last=frames_since_last,
            static_mask=static_mask,
            on_static_spot=on_static_spot,
            use_score=use_score,
        )
        if best_idx is None:
            return BallState(xyxy=None, conf=0.0, interpolated=False)

        conf = float(candidates.confidence[best_idx]) if candidates.confidence is not None else 0.0
        return BallState(
            xyxy=np.asarray(candidates.xyxy[best_idx], dtype=np.float32).copy(),
            conf=conf,
            interpolated=False,
        )

    def update(self, candidates: sv.Detections, *, frame_idx: int | None = None) -> BallState:
        """Stateful ball selection with static-spot memory and jump/reacquire gating."""
        current_idx = self._step_counter if frame_idx is None else frame_idx
        self._step_counter += 1

        if len(candidates) > 0:
            centers = np.column_stack(
                (
                    (candidates.xyxy[:, 0] + candidates.xyxy[:, 2]) * 0.5,
                    (candidates.xyxy[:, 1] + candidates.xyxy[:, 3]) * 0.5,
                )
            ).astype(np.float64)
        else:
            centers = np.empty((0, 2), dtype=np.float64)

        static_mask = self._update_static_spots(centers, current_idx)
        on_static_spot = self._is_on_static_spot()

        if self._last_xyxy is None or self._last_frame_idx is None:
            frames_since_last = 1
        else:
            frames_since_last = max(1, current_idx - self._last_frame_idx)

        best_idx = self._select_index(
            candidates,
            self._last_xyxy,
            frames_since_last=frames_since_last,
            static_mask=static_mask,
            on_static_spot=on_static_spot,
            use_score=True,
        )
        if best_idx is None:
            return BallState(xyxy=None, conf=0.0, interpolated=False)

        conf = float(candidates.confidence[best_idx]) if candidates.confidence is not None else 0.0
        state = BallState(
            xyxy=np.asarray(candidates.xyxy[best_idx], dtype=np.float32).copy(),
            conf=conf,
            interpolated=False,
        )
        self._last_xyxy = state.xyxy.copy()
        self._last_frame_idx = current_idx
        self._last_was_static = bool(static_mask[best_idx])
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
