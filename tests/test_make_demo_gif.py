"""Unit tests for scripts/make_demo_gif.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "make_demo_gif.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_demo_gif", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_demo_gif_creates_valid_gif(
    synthetic_video_path: Path,
    tmp_path: Path,
) -> None:
    """Synthetic video produces an animated GIF with expected width and size under limit."""
    mod = _load_script_module()
    out = tmp_path / "demo.gif"

    result = mod.make_demo_gif(
        synthetic_video_path,
        out,
        start_sec=0.0,
        duration_sec=0.4,
        width=160,
        fps=10.0,
        colors=32,
        max_mb=1.0,
    )

    assert result == out
    assert out.is_file()
    with Image.open(out) as gif:
        assert gif.format == "GIF"
        assert gif.size[0] == 160
        assert gif.n_frames >= 2


def test_make_demo_gif_cli_missing_file_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing input file returns exit code 1 with a clean error message."""
    mod = _load_script_module()
    rc = mod.main(["--input", str(tmp_path / "missing.mp4"), "--output", str(tmp_path / "x.gif")])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err
