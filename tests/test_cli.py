"""Tests for the `python -m futbolcv` command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from futbolcv.cli import main
from futbolcv.config import DetectorConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def test_cli_help_succeeds_via_module() -> None:
    """`python -m futbolcv --help` exits cleanly and prints all documented flags."""
    proc = subprocess.run(
        [sys.executable, "-m", "futbolcv", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    for flag in ("--input", "--output", "--config", "--max-frames", "--device"):
        assert flag in proc.stdout
    assert "Traceback" not in proc.stderr


def test_cli_missing_input_prints_clean_error_without_traceback(tmp_path: Path) -> None:
    """Non-existent video path prints a readable error message without a Python traceback."""
    missing = tmp_path / "missing_match.mp4"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "futbolcv",
            "--input",
            str(missing),
            "--output",
            str(tmp_path / "out"),
            "--config",
            str(DEFAULT_CONFIG),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "Error:" in proc.stderr
    assert "missing_match.mp4" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "Traceback" not in proc.stdout


def test_cli_invalid_config_and_flags_return_clean_error(
    synthetic_video_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing config, bad YAML keys, and invalid flags fail cleanly without tracebacks."""
    # Missing config file
    rc = main(
        [
            "--input",
            str(synthetic_video_path),
            "--output",
            str(tmp_path / "out"),
            "--config",
            str(tmp_path / "nope.yaml"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err

    # Unknown YAML key
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("detector:\n  bogus_option: 1\n", encoding="utf-8")
    rc = main(
        [
            "--input",
            str(synthetic_video_path),
            "--output",
            str(tmp_path / "out"),
            "--config",
            str(bad_cfg),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "bogus_option" in captured.err

    # Negative max-frames
    rc = main(
        [
            "--input",
            str(synthetic_video_path),
            "--output",
            str(tmp_path / "out"),
            "--config",
            str(DEFAULT_CONFIG),
            "--max-frames",
            "-3",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "--max-frames" in captured.err

    # Blank --device
    rc = main(
        [
            "--input",
            str(synthetic_video_path),
            "--output",
            str(tmp_path / "out"),
            "--config",
            str(DEFAULT_CONFIG),
            "--device",
            "   ",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "--device" in captured.err


def test_cli_end_to_end_with_mocked_detector(
    synthetic_video_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI wires config, --device override, --max-frames, and pipeline to produce mp4/csv/json."""
    seen_devices: list[str] = []

    class FakeDetector:
        def __init__(self, cfg: DetectorConfig) -> None:
            seen_devices.append(cfg.device)
            self._step = 0

        def detect(self, frame: np.ndarray) -> tuple[sv.Detections, sv.Detections]:
            shift = self._step * 5.0
            self._step += 1
            players = sv.Detections(
                xyxy=np.array([[20.0 + shift, 60.0, 60.0 + shift, 160.0]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([0], dtype=int),
            )
            balls = sv.Detections(
                xyxy=np.array([[100.0 + shift, 110.0, 112.0 + shift, 122.0]], dtype=np.float32),
                confidence=np.array([0.7], dtype=np.float32),
                class_id=np.array([32], dtype=int),
            )
            return players, balls

    monkeypatch.setattr("futbolcv.pipeline.Detector", FakeDetector)

    out_dir = tmp_path / "cli_out"
    rc = main(
        [
            "--input",
            str(synthetic_video_path),
            "--output",
            str(out_dir),
            "--config",
            str(DEFAULT_CONFIG),
            "--max-frames",
            "10",
            "--device",
            "cpu",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "Done." in captured.out
    assert seen_devices == ["cpu"]
    assert (out_dir / "annotated.mp4").is_file()
    assert (out_dir / "tracks.csv").is_file()
    summary_path = out_dir / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_frames"] == 10
    assert summary["video"]["source"] == synthetic_video_path.name


def test_cli_download_connection_error_shows_manual_models_hint(
    synthetic_video_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Download ConnectionError exits 1 with manual models/ hint and no Python traceback."""

    class FailingDetector:
        def __init__(self, cfg: DetectorConfig) -> None:
            raise ConnectionError("Download failure for yolo11s.pt. Environment may be offline.")

    monkeypatch.setattr("futbolcv.pipeline.Detector", FailingDetector)

    rc = main(
        [
            "--input",
            str(synthetic_video_path),
            "--output",
            str(tmp_path / "out_fail"),
            "--config",
            str(DEFAULT_CONFIG),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    assert "models/" in captured.err
    assert "yolo11s.pt" in captured.err
    assert "https://github.com/ultralytics/assets/releases/" in captured.err
