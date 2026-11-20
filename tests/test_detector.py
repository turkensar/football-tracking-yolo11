"""Unit and slow integration tests for the YOLO11 Detector wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import supervision as sv

from futbolcv.config import DetectorConfig
from futbolcv.detector import Detector, resolve_device, resolve_weights_path


class _FakeTensor:
    """Minimal tensor stand-in supporting ``.cpu().numpy()`` for supervision."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeBoxes:
    """Minimal Ultralytics Boxes stand-in for offline unit tests."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = _FakeTensor(xyxy.astype(np.float32))
        self.conf = _FakeTensor(conf.astype(np.float32))
        self.cls = _FakeTensor(cls.astype(np.float32))
        self.id = None


class _FakeResult:
    """Minimal Ultralytics Results stand-in compatible with ``sv.Detections.from_ultralytics``."""

    def __init__(
        self,
        boxes: _FakeBoxes | None,
        names: dict[int, str] | None = None,
    ) -> None:
        self.boxes = boxes
        self.obb = None
        self.masks = None
        self.names = names or {0: "person", 2: "car", 32: "sports ball"}


def test_resolve_device_auto_and_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """'auto' maps to '0' when CUDA is available and 'cpu' otherwise."""
    monkeypatch.setattr("futbolcv.detector.torch.cuda.is_available", lambda: True)
    assert resolve_device("auto") == "0"
    assert resolve_device(" AUTO ") == "0"

    monkeypatch.setattr("futbolcv.detector.torch.cuda.is_available", lambda: False)
    assert resolve_device("auto") == "cpu"

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"


def test_resolve_weights_path_bare_vs_explicit(tmp_path: Path) -> None:
    """Bare filenames resolve under models_dir; explicit paths are preserved."""
    models_dir = tmp_path / "models"
    assert resolve_weights_path("yolo11s.pt", models_dir=models_dir) == models_dir / "yolo11s.pt"
    assert resolve_weights_path("models/yolo11s.pt", models_dir=models_dir) == Path(
        "models/yolo11s.pt"
    )
    assert resolve_weights_path("./custom.pt", models_dir=models_dir) == Path("./custom.pt")

    abs_path = tmp_path / "weights" / "best.pt"
    assert resolve_weights_path(str(abs_path), models_dir=models_dir) == abs_path


def test_detector_loads_once_and_splits_classes(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: np.ndarray,
    tmp_path: Path,
) -> None:
    """Detector loads YOLO once and splits player/ball detections by class and confidence."""
    init_calls: list[str] = []
    predict_calls: list[dict[str, Any]] = []

    fake_result = _FakeResult(
        _FakeBoxes(
            xyxy=np.array(
                [
                    [10, 20, 50, 140],  # person, high conf -> keep
                    [60, 25, 95, 135],  # person, conf=0.30 -> keep
                    [110, 30, 140, 130],  # person, conf=0.18 < 0.25 -> drop
                    [150, 100, 165, 115],  # ball, conf=0.20 >= 0.15 -> keep
                    [170, 100, 185, 115],  # ball, conf=0.10 < 0.15 -> drop
                    [200, 10, 290, 90],  # car (class 2), high conf -> drop
                ],
                dtype=np.float32,
            ),
            conf=np.array([0.85, 0.30, 0.18, 0.20, 0.10, 0.95], dtype=np.float32),
            cls=np.array([0, 0, 0, 32, 32, 2], dtype=np.float32),
        )
    )

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            init_calls.append(weights)

        def predict(self, **kwargs: Any) -> list[_FakeResult]:
            predict_calls.append(kwargs)
            return [fake_result]

    monkeypatch.setattr("futbolcv.detector.YOLO", FakeYOLO)
    monkeypatch.setattr("futbolcv.detector.torch.cuda.is_available", lambda: False)

    models_dir = tmp_path / "models"
    cfg = DetectorConfig(weights="yolo11s.pt", conf=0.25, ball_conf=0.15, imgsz=640, device="auto")
    detector = Detector(cfg, models_dir=models_dir)

    players_1, balls_1 = detector.detect(synthetic_frame)
    players_2, balls_2 = detector.detect(synthetic_frame)

    # Model loaded exactly once under models_dir
    assert init_calls == [str(models_dir / "yolo11s.pt")]
    assert models_dir.is_dir()
    assert len(predict_calls) == 2

    # Inference kwargs honor config and resolved device
    first_call = predict_calls[0]
    assert first_call["conf"] == pytest.approx(0.15)
    assert first_call["imgsz"] == 640
    assert first_call["classes"] == [0, 32]
    assert first_call["device"] == "cpu"
    assert first_call["verbose"] is False

    # Players and ball candidates cleanly separated
    assert isinstance(players_1, sv.Detections)
    assert isinstance(balls_1, sv.Detections)
    assert len(players_1) == 2
    assert len(players_2) == 2
    assert np.array_equal(players_1.class_id, np.array([0, 0]))
    assert np.allclose(players_1.confidence, np.array([0.85, 0.30]))

    assert len(balls_1) == 1
    assert len(balls_2) == 1
    assert np.array_equal(balls_1.class_id, np.array([32]))
    assert np.allclose(balls_1.confidence, np.array([0.20]))


def test_detector_custom_class_ids_and_height_filter(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: np.ndarray,
    tmp_path: Path,
) -> None:
    """Class filters and min_player_box_height_px come from DetectorConfig."""
    fake_result = _FakeResult(
        _FakeBoxes(
            xyxy=np.array(
                [
                    [10, 10, 50, 110],  # class 1 (player), height=100 -> keep
                    [60, 10, 90, 95],  # class 2 (goalkeeper), height=85 -> keep
                    [100, 10, 115, 35],  # class 1 (tiny spectator), height=25 -> drop
                    [140, 140, 152, 152],  # class 3 (ball), height=12 -> keep (not height-filtered)
                ],
                dtype=np.float32,
            ),
            conf=np.array([0.9, 0.8, 0.9, 0.5], dtype=np.float32),
            cls=np.array([1, 2, 1, 3], dtype=np.float32),
        ),
        names={1: "player", 2: "goalkeeper", 3: "ball"},
    )

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights

        def predict(self, **kwargs: Any) -> list[_FakeResult]:
            return [fake_result]

    monkeypatch.setattr("futbolcv.detector.YOLO", FakeYOLO)

    cfg = DetectorConfig(
        weights="custom.pt",
        conf=0.25,
        ball_conf=0.2,
        device="cpu",
        player_class_ids=[1, 2],
        ball_class_ids=[3],
        min_player_box_height_px=40,
    )
    detector = Detector(cfg, models_dir=tmp_path / "models")
    players, balls = detector.detect(synthetic_frame)

    assert len(players) == 2
    assert np.array_equal(players.class_id, np.array([1, 2]))
    assert len(balls) == 1
    assert np.array_equal(balls.class_id, np.array([3]))


def test_detector_handles_empty_results(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: np.ndarray,
    tmp_path: Path,
) -> None:
    """Empty predictions or missing boxes return empty Detections without crashing."""
    empty_boxes = _FakeResult(
        _FakeBoxes(
            xyxy=np.empty((0, 4), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
            cls=np.empty((0,), dtype=np.float32),
        )
    )
    responses: list[list[_FakeResult]] = [[empty_boxes], [_FakeResult(None)], []]

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            self._idx = 0

        def predict(self, **kwargs: Any) -> list[_FakeResult]:
            out = responses[self._idx]
            self._idx += 1
            return out

    monkeypatch.setattr("futbolcv.detector.YOLO", FakeYOLO)

    detector = Detector(DetectorConfig(device="cpu"), models_dir=tmp_path / "models")
    for _ in range(3):
        players, balls = detector.detect(synthetic_frame)
        assert len(players) == 0
        assert len(balls) == 0


@pytest.mark.slow
def test_detector_real_weights_finds_person() -> None:
    """Integration test using real YOLO11 weights on Ultralytics' bundled sample image."""
    import ultralytics

    sample_image = Path(ultralytics.__file__).resolve().parent / "assets" / "zidane.jpg"
    assert sample_image.is_file(), f"Bundled sample image missing: {sample_image}"

    frame = cv2.imread(str(sample_image))
    assert frame is not None

    cfg = DetectorConfig(weights="yolo11s.pt", device="auto")
    detector = Detector(cfg)
    players, _ = detector.detect(frame)

    assert len(players) >= 1
    assert (Path("models") / "yolo11s.pt").is_file()
    assert not Path("yolo11s.pt").exists()
