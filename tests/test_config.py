"""Unit tests for YAML configuration loading and validation."""

from pathlib import Path

import pytest

from futbolcv.config import (
    AnnotateConfig,
    AppConfig,
    BallConfig,
    DetectorConfig,
    OutputConfig,
    TeamConfig,
    TrackerConfig,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
FAST_CONFIG_PATH = REPO_ROOT / "configs" / "fast.yaml"
FOOTBALL_CONFIG_PATH = REPO_ROOT / "configs" / "football.yaml"


def test_load_fast_config() -> None:
    """Fast CPU preset overrides only detector.imgsz to 640."""
    default_cfg = load_config(DEFAULT_CONFIG_PATH)
    fast_cfg = load_config(FAST_CONFIG_PATH)

    assert fast_cfg.detector.imgsz == 640
    fast_cfg.detector.imgsz = default_cfg.detector.imgsz
    assert fast_cfg == default_cfg


def test_load_football_config() -> None:
    """Fine-tuned football preset maps ball/goalkeeper+player/referee classes cleanly."""
    default_cfg = load_config(DEFAULT_CONFIG_PATH)
    football_cfg = load_config(FOOTBALL_CONFIG_PATH)

    assert football_cfg.detector.weights == "football_yolo11s.pt"
    assert football_cfg.detector.ball_conf == pytest.approx(0.25)
    assert football_cfg.detector.ball_class_ids == [0]
    assert football_cfg.detector.player_class_ids == [1, 2]
    assert football_cfg.detector.referee_class_ids == [3]
    assert default_cfg.detector.referee_class_ids == []

    football_cfg.detector.weights = default_cfg.detector.weights
    football_cfg.detector.ball_conf = default_cfg.detector.ball_conf
    football_cfg.detector.ball_class_ids = default_cfg.detector.ball_class_ids
    football_cfg.detector.player_class_ids = default_cfg.detector.player_class_ids
    football_cfg.detector.referee_class_ids = default_cfg.detector.referee_class_ids
    assert football_cfg == default_cfg


def test_load_default_config() -> None:
    """Default YAML file loads cleanly and populates all required fields."""
    cfg = load_config(DEFAULT_CONFIG_PATH)

    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.detector, DetectorConfig)
    assert isinstance(cfg.tracker, TrackerConfig)
    assert isinstance(cfg.ball, BallConfig)
    assert isinstance(cfg.team, TeamConfig)
    assert isinstance(cfg.annotate, AnnotateConfig)
    assert isinstance(cfg.output, OutputConfig)

    # Required fields specified in PLAN.md T02 (calibrated in T11)
    assert cfg.detector.weights == "yolo11s.pt"
    assert cfg.detector.conf == pytest.approx(0.25)
    assert cfg.detector.imgsz == 1280
    assert cfg.detector.device == "auto"
    assert cfg.detector.player_class_ids == [0]
    assert cfg.detector.ball_class_ids == [32]
    assert cfg.ball.max_gap_frames == 15
    assert cfg.ball.distance_weight == pytest.approx(0.3)
    assert cfg.ball.static_radius_px == pytest.approx(5.0)
    assert cfg.ball.static_frames == 25
    assert cfg.ball.reacquire_conf == pytest.approx(0.5)
    assert cfg.team.n_init_frames == 30
    assert cfg.tracker.frame_rate is None
    assert cfg.tracker.lost_track_buffer == 60
    assert cfg.team.exclude_grass is True

    # BGR colors are normalized to 3-int tuples
    assert cfg.annotate.unknown_color_bgr == (160, 160, 160)
    # Default YAML stays strictly synchronized with dataclass defaults
    assert cfg == AppConfig()


def test_missing_keys_fall_back_to_defaults(tmp_path: Path) -> None:
    """Partial YAML configs override given keys and fill missing ones with defaults."""
    cfg_file = tmp_path / "partial.yaml"
    cfg_file.write_text(
        "detector:\n  conf: 0.4\n  imgsz: 1280\nball:\n  max_gap_frames: 10\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_file)
    defaults = AppConfig()

    assert cfg.detector.conf == pytest.approx(0.4)
    assert cfg.detector.imgsz == 1280
    assert cfg.detector.weights == defaults.detector.weights
    assert cfg.detector.device == defaults.detector.device
    assert cfg.ball.max_gap_frames == 10
    assert cfg.ball.max_jump_px == defaults.ball.max_jump_px
    assert cfg.team == defaults.team
    assert cfg.tracker == defaults.tracker
    assert cfg.annotate == defaults.annotate
    assert cfg.output == defaults.output


def test_empty_yaml_uses_all_defaults(tmp_path: Path) -> None:
    """An empty YAML file returns a full default AppConfig."""
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text("", encoding="utf-8")

    assert load_config(cfg_file) == AppConfig()


def test_null_section_uses_section_defaults(tmp_path: Path) -> None:
    """A section header with no keys (YAML null) falls back to section defaults."""
    cfg_file = tmp_path / "null_section.yaml"
    cfg_file.write_text("detector:\nball:\n", encoding="utf-8")

    assert load_config(cfg_file) == AppConfig()


def test_unknown_top_level_section_raises_value_error(tmp_path: Path) -> None:
    """Unknown top-level YAML sections raise a descriptive ValueError."""
    cfg_file = tmp_path / "bad_section.yaml"
    cfg_file.write_text("unknown_section:\n  foo: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown_section"):
        load_config(cfg_file)


def test_unknown_nested_key_raises_value_error(tmp_path: Path) -> None:
    """Unknown keys inside a valid section raise a descriptive ValueError."""
    cfg_file = tmp_path / "bad_key.yaml"
    cfg_file.write_text("detector:\n  nonexistent_flag: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"detector.*nonexistent_flag"):
        load_config(cfg_file)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Loading a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_non_mapping_root_or_section_raises_value_error(tmp_path: Path) -> None:
    """Non-mapping YAML roots and sections raise ValueError."""
    bad_root = tmp_path / "list_root.yaml"
    bad_root.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad_root)

    bad_section = tmp_path / "scalar_section.yaml"
    bad_section.write_text("detector: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="detector"):
        load_config(bad_section)


def test_device_int_coerced_to_str(tmp_path: Path) -> None:
    """YAML integer GPU index (e.g., `device: 0`) is coerced to string `'0'`."""
    cfg_file = tmp_path / "gpu_idx.yaml"
    cfg_file.write_text("detector:\n  device: 0\n", encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.detector.device == "0"


def test_tracker_frame_rate_override(tmp_path: Path) -> None:
    """Explicit tracker.frame_rate overrides the default None."""
    cfg_file = tmp_path / "fps_override.yaml"
    cfg_file.write_text("tracker:\n  frame_rate: 50.0\n", encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.tracker.frame_rate == pytest.approx(50.0)


@pytest.mark.parametrize(
    "yaml_text",
    [
        "detector:\n  conf: 1.5\n",
        "detector:\n  conf: true\n",
        "detector:\n  conf: high\n",
        "detector:\n  imgsz: 0\n",
        "detector:\n  player_class_ids: [true]\n",
        "detector:\n  referee_class_ids: [true]\n",
        "tracker:\n  frame_rate: 0\n",
        "tracker:\n  frame_rate: true\n",
        "tracker:\n  scene_cut_threshold: 0\n",
        "tracker:\n  scene_cut_threshold: 1.5\n",
        "ball:\n  max_gap_frames: -1\n",
        "ball:\n  distance_weight: -0.1\n",
        "ball:\n  static_radius_px: 0\n",
        "ball:\n  static_frames: 0\n",
        "ball:\n  reacquire_conf: 1.5\n",
        "team:\n  color_space: rgb\n",
        "team:\n  exclude_grass: 'yes'\n",
        "annotate:\n  unknown_color_bgr: [100, 100]\n",
        "annotate:\n  use_jersey_colors: 'yes'\n",
        "annotate:\n  show_hud: 'yes'\n",
        "output:\n  stride: 0\n",
        "output:\n  codec: mp4\n",
    ],
)
def test_invalid_field_values_raise_value_error(tmp_path: Path, yaml_text: str) -> None:
    """Out-of-range or malformed field values fail fast with ValueError."""
    cfg_file = tmp_path / "invalid.yaml"
    cfg_file.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(cfg_file)
