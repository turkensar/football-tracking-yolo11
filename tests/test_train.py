"""Unit tests for scripts/train.py and notebooks/finetune_colab.ipynb."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT_PATH = REPO_ROOT / "scripts" / "train.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "finetune_colab.ipynb"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


def _load_train_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("futbolcv_train_script", TRAIN_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_cli_help_runs() -> None:
    """`python scripts/train.py --help` exits cleanly and lists required flags."""
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for flag in ("--data", "--epochs", "--imgsz", "--batch", "--device"):
        assert flag in result.stdout


def test_train_cli_passes_arguments_to_mocked_yolo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI forwards parsed arguments directly to mocked `ultralytics.YOLO.train`."""
    mod = _load_train_module()
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("nc: 4\nnames: [ball, goalkeeper, player, referee]\n", encoding="utf-8")
    out_project = tmp_path / "runs"

    init_weights: list[str] = []
    train_kwargs: list[dict[str, Any]] = []

    class _FakeYOLO:
        def __init__(self, weights: str) -> None:
            init_weights.append(weights)

        def train(self, **kwargs: Any) -> dict[str, Any]:
            train_kwargs.append(kwargs)
            return {"ok": True}

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(mod, "resolve_base_weights", lambda w: w)

    rc = mod.main(
        [
            "--data",
            str(data_yaml),
            "--weights",
            "yolo11s.pt",
            "--epochs",
            "25",
            "--imgsz",
            "1280",
            "--batch",
            "-1",
            "--device",
            "0",
            "--project",
            str(out_project),
            "--name",
            "exp1",
            "--patience",
            "10",
            "--seed",
            "7",
        ]
    )

    assert rc == 0
    assert init_weights == ["yolo11s.pt"]
    assert len(train_kwargs) == 1
    assert train_kwargs[0] == {
        "data": str(data_yaml),
        "epochs": 25,
        "imgsz": 1280,
        "batch": -1,
        "device": "0",
        "project": str(out_project),
        "name": "exp1",
        "patience": 10,
        "seed": 7,
    }


def test_train_resolve_device_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """`auto` resolves to `'0'` with CUDA and `'cpu'` without CUDA."""
    mod = _load_train_module()

    fake_cuda = ModuleType("cuda")
    fake_torch = ModuleType("torch")
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_cuda.is_available = lambda: True  # type: ignore[attr-defined]
    assert mod.resolve_train_device("auto") == "0"

    fake_cuda.is_available = lambda: False  # type: ignore[attr-defined]
    assert mod.resolve_train_device("auto") == "cpu"
    assert mod.resolve_train_device("cpu") == "cpu"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--epochs", "0"],
        ["--imgsz", "0"],
        ["--batch", "0"],
        ["--patience", "-1"],
    ],
)
def test_train_cli_invalid_args_return_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
) -> None:
    """Invalid hyperparameter values exit with code 1 and no traceback."""
    mod = _load_train_module()
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("nc: 4\n", encoding="utf-8")

    rc = mod.main(["--data", str(data_yaml), *extra_args])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_train_cli_missing_data_file_returns_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing data.yaml returns exit code 1 with a readable message."""
    mod = _load_train_module()
    rc = mod.main(["--data", str(tmp_path / "missing.yaml")])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_colab_notebook_valid_clean_and_secret_safe() -> None:
    """Colab notebook has valid JSON, cleared outputs, exact ultralytics pin, and no secrets."""
    assert NOTEBOOK_PATH.is_file()
    raw_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    nb = json.loads(raw_text)

    assert nb.get("nbformat") == 4
    cells = nb.get("cells")
    assert isinstance(cells, list) and len(cells) >= 5

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(code_cells) >= 5
    for cell in code_cells:
        assert cell.get("outputs") == []
        assert cell.get("execution_count") is None

    full_source = "\n".join("".join(c.get("source", [])) for c in cells)

    # Open in Colab badge and Colab Secrets usage
    assert "colab-badge.svg" in full_source
    assert 'userdata.get("ROBOFLOW_API_KEY")' in full_source

    # No hardcoded Roboflow API keys or local Windows paths
    assert not re.search(r'api_key\s*=\s*["\'][^"\']+["\']', full_source)
    for forbidden in ("C:\\", "C:/", "/Users/", "Projeler", "@gmail", "@hotmail"):
        assert forbidden.lower() not in raw_text.lower()

    # Exact ultralytics pin synchronized with requirements.txt
    req_text = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    match = re.search(r"^ultralytics==([0-9.]+)\s*$", req_text, flags=re.MULTILINE)
    assert match is not None
    expected_pin = f"ultralytics=={match.group(1)}"
    assert expected_pin in full_source
