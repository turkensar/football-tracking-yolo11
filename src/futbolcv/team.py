"""Jersey-color team classification using KMeans clustering and majority voting."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

from futbolcv.config import TeamConfig

_HEAD_SKIP_RATIO = 0.15
_MIN_HEAD_SKIP_BOX_H = 12
_GRASS_HUE_MIN = 38
_GRASS_HUE_MAX = 85
_GRASS_SAT_MIN = 45
_GRASS_VAL_MIN = 45
_MIN_NON_GRASS_RATIO = 0.15


def extract_jersey_color(
    frame: np.ndarray,
    xyxy: np.ndarray,
    *,
    color_space: str = "lab",
    top_ratio: float = 0.5,
    center_crop_ratio: float = 0.5,
    exclude_grass: bool = True,
) -> np.ndarray | None:
    """Extract the mean jersey color from the upper-half middle torso region of a player box.

    When ``exclude_grass`` is enabled, green pitch background pixels (OpenCV HSV Hue
    in ``[38, 85]``) are excluded before averaging unless fewer than 15% non-grass
    pixels remain (preserving genuinely green jerseys). Returns a 3-element
    ``float64`` feature vector in Lab or HSV space, or ``None`` for empty boxes.
    """
    if color_space not in ("lab", "hsv"):
        raise ValueError(f"Unsupported color_space {color_space!r}; expected 'lab' or 'hsv'")

    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w, int(round(float(xyxy[0])))))
    y1 = max(0, min(frame_h, int(round(float(xyxy[1])))))
    x2 = max(0, min(frame_w, int(round(float(xyxy[2])))))
    y2 = max(0, min(frame_h, int(round(float(xyxy[3])))))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    head_offset = int(round(box_h * _HEAD_SKIP_RATIO)) if box_h >= _MIN_HEAD_SKIP_BOX_H else 0
    upper_h = max(1, int(round(box_h * top_ratio)))
    crop_y1 = min(y2 - 1, y1 + head_offset)
    crop_y2 = min(y2, max(crop_y1 + 1, y1 + upper_h))

    crop_w = max(1, int(round(box_w * center_crop_ratio)))
    margin_x = (box_w - crop_w) // 2
    crop_x1 = x1 + margin_x
    crop_x2 = min(x2, crop_x1 + crop_w)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    conversion = cv2.COLOR_BGR2LAB if color_space == "lab" else cv2.COLOR_BGR2HSV
    converted = cv2.cvtColor(crop, conversion).astype(np.float64)

    if exclude_grass:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = (
            (hsv[:, :, 0] >= _GRASS_HUE_MIN)
            & (hsv[:, :, 0] <= _GRASS_HUE_MAX)
            & (hsv[:, :, 1] >= _GRASS_SAT_MIN)
            & (hsv[:, :, 2] >= _GRASS_VAL_MIN)
        )
        non_grass = ~grass_mask
        min_pixels = max(3, int(round(non_grass.size * _MIN_NON_GRASS_RATIO)))
        if int(non_grass.sum()) >= min_pixels:
            return converted[non_grass].mean(axis=0)

    return converted.mean(axis=(0, 1))


def _majority_vote(history: deque[int]) -> int:
    """Resolve the winning team label from a sliding window of recent predictions.

    Breaks ties by preferring valid team IDs (``0``/``1``) over unknown (``-1``),
    and then by the most recently observed label.
    """
    counts = Counter(history)
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    if len(tied) == 1:
        return tied[0]

    known_tied = [label for label in tied if label != -1]
    candidates = set(known_tied) if known_tied else set(tied)
    for label in reversed(history):
        if label in candidates:
            return label
    return tied[0]


class TeamClassifier:
    """Unsupervised 2-team jersey classifier using KMeans(k=2) and temporal voting."""

    def __init__(self, cfg: TeamConfig) -> None:
        self.cfg = cfg
        self._centers: np.ndarray | None = None
        self._init_features: list[np.ndarray] = []
        self._init_frames_collected: int = 0
        self._vote_history: dict[int, deque[int]] = {}

    @property
    def is_fitted(self) -> bool:
        """Return True once two team cluster centers have been fitted."""
        return self._centers is not None

    @property
    def cluster_centers(self) -> np.ndarray | None:
        """Return a copy of the fitted ``(2, 3)`` cluster centers, or ``None``."""
        return None if self._centers is None else self._centers.copy()

    @property
    def init_frames_collected(self) -> int:
        """Number of initialization frames accumulated so far."""
        return self._init_frames_collected

    def reset_history(self) -> None:
        """Clear per-``tracker_id`` majority vote windows while keeping fitted centers."""
        self._vote_history.clear()

    def _extract_box_feature(self, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        """Extract one jersey feature vector using this classifier's config."""
        return extract_jersey_color(
            frame,
            xyxy,
            color_space=self.cfg.color_space,
            top_ratio=self.cfg.jersey_top_ratio,
            center_crop_ratio=self.cfg.jersey_center_crop_ratio,
            exclude_grass=self.cfg.exclude_grass,
        )

    def add_init_frame(self, frame: np.ndarray, dets: sv.Detections) -> int:
        """Collect jersey color samples from one frame up to ``cfg.n_init_frames`` frames."""
        if self._init_frames_collected >= self.cfg.n_init_frames:
            return 0

        added = 0
        for xyxy in dets.xyxy:
            feat = self._extract_box_feature(frame, xyxy)
            if feat is not None:
                self._init_features.append(feat)
                added += 1

        self._init_frames_collected += 1
        return added

    def fit(
        self,
        samples: Sequence[tuple[np.ndarray, sv.Detections]] | None = None,
    ) -> TeamClassifier:
        """Fit ``KMeans(k=2)`` on buffered init frames or an explicit ``(frame, dets)`` sequence."""
        if samples is not None:
            self._init_features.clear()
            self._init_frames_collected = 0
            for frame, dets in samples:
                if self._init_frames_collected >= self.cfg.n_init_frames:
                    break
                self.add_init_frame(frame, dets)

        if len(self._init_features) < 2:
            self._centers = None
            return self

        feature_matrix = np.vstack(self._init_features)
        if np.unique(feature_matrix, axis=0).shape[0] < 2:
            self._centers = None
            return self
        kmeans = KMeans(
            n_clusters=2,
            n_init=10,
            random_state=self.cfg.random_state,
        )
        kmeans.fit(feature_matrix)

        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
        # Sort cluster centers deterministically so team 0 / team 1 ordering is reproducible.
        order = sorted(range(2), key=lambda idx: tuple(float(v) for v in centers[idx]))
        self._centers = centers[order]
        self.reset_history()
        return self

    def _predict_raw_label(self, frame: np.ndarray, xyxy: np.ndarray) -> int:
        """Classify a single box to ``0``, ``1``, or ``-1`` (outlier / invalid box)."""
        assert self._centers is not None
        feat = self._extract_box_feature(frame, xyxy)
        if feat is None:
            return -1

        distances = np.linalg.norm(self._centers - feat, axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) > self.cfg.outlier_distance_threshold:
            return -1
        return nearest

    def predict(self, frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
        """Predict team IDs (``0``, ``1``, or ``-1``) and store them in ``dets.data['team_id']``."""
        count = len(dets)
        if count == 0:
            empty = np.empty((0,), dtype=int)
            dets.data["team_id"] = empty
            return empty

        if self._centers is None:
            unknown = np.full(count, -1, dtype=int)
            dets.data["team_id"] = unknown
            return unknown

        tracker_ids = dets.tracker_id
        resolved: list[int] = []
        for idx in range(count):
            raw_label = self._predict_raw_label(frame, dets.xyxy[idx])
            if tracker_ids is not None:
                tid = int(tracker_ids[idx])
                history = self._vote_history.get(tid)
                if history is None:
                    history = deque(maxlen=self.cfg.vote_window)
                    self._vote_history[tid] = history
                history.append(raw_label)
                resolved.append(_majority_vote(history))
            else:
                resolved.append(raw_label)

        team_ids = np.array(resolved, dtype=int)
        dets.data["team_id"] = team_ids
        return team_ids


__all__ = ["TeamClassifier", "extract_jersey_color"]
