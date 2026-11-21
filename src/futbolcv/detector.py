"""YOLO11 object detection wrapper returning supervision Detections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from futbolcv.config import DetectorConfig

DEFAULT_MODELS_DIR = Path("models")


def resolve_device(device: str) -> str:
    """Resolve ``'auto'`` to CUDA device ``'0'`` when available, otherwise ``'cpu'``."""
    normalized = device.strip()
    if normalized.lower() == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return normalized


def _is_bare_filename(weights: str) -> bool:
    """Return True when ``weights`` is a bare filename with no directory component."""
    if "/" in weights or "\\" in weights:
        return False
    weight_path = Path(weights)
    return not weight_path.is_absolute() and len(weight_path.parts) == 1


def resolve_weights_path(weights: str, *, models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """Place bare weight filenames under ``models_dir`` to keep the project root clean."""
    if _is_bare_filename(weights):
        return models_dir / weights
    return Path(weights)


class Detector:
    """YOLO11 wrapper that loads weights once and separates player and ball detections."""

    def __init__(
        self,
        cfg: DetectorConfig,
        *,
        models_dir: Path = DEFAULT_MODELS_DIR,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.weights_path = resolve_weights_path(cfg.weights, models_dir=models_dir)
        if _is_bare_filename(cfg.weights):
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(self.weights_path))

    def detect(self, frame: np.ndarray) -> tuple[sv.Detections, sv.Detections]:
        """Run detection on a BGR frame and return ``(players_and_refs, ball_candidates)``."""
        tracked_person_ids = sorted(
            set(self.cfg.player_class_ids) | set(self.cfg.referee_class_ids)
        )
        target_classes = sorted(set(tracked_person_ids) | set(self.cfg.ball_class_ids))
        if not target_classes:
            return sv.Detections.empty(), sv.Detections.empty()

        infer_conf = min(self.cfg.conf, self.cfg.ball_conf)
        results = self._model.predict(
            source=frame,
            conf=infer_conf,
            imgsz=self.cfg.imgsz,
            classes=target_classes,
            device=self.device,
            verbose=False,
        )
        if not results:
            return sv.Detections.empty(), sv.Detections.empty()

        detections = sv.Detections.from_ultralytics(results[0])
        if len(detections) == 0 or detections.class_id is None:
            return sv.Detections.empty(), sv.Detections.empty()

        class_ids = detections.class_id
        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float32)
        )

        player_mask = np.isin(class_ids, tracked_person_ids) & (confidences >= self.cfg.conf)
        if self.cfg.min_player_box_height_px > 0:
            heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            player_mask &= heights >= self.cfg.min_player_box_height_px

        ball_mask = np.isin(class_ids, self.cfg.ball_class_ids) & (
            confidences >= self.cfg.ball_conf
        )

        return detections[player_mask], detections[ball_mask]


__all__ = ["DEFAULT_MODELS_DIR", "Detector", "resolve_device", "resolve_weights_path"]
