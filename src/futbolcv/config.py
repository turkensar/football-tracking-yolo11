"""YAML configuration loader and typed configuration dataclasses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

_VALID_COLOR_SPACES = ("lab", "hsv")
_BGR_FIELDS = (
    "team_0_color_bgr",
    "team_1_color_bgr",
    "unknown_color_bgr",
    "ball_color_bgr",
)


def _validate_int(name: str, value: Any, *, min_value: int | None = None) -> int:
    """Validate a non-boolean integer, optionally enforcing a lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{name}' must be an integer, got {value!r}")
    if min_value is not None and value < min_value:
        raise ValueError(f"'{name}' must be >= {min_value}, got {value!r}")
    return value


def _validate_float(
    name: str,
    value: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    strict_min: bool = False,
) -> float:
    """Validate a non-boolean real number and return it as float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number, got {value!r}")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"'{name}' must be a finite number, got {value!r}")
    if min_value is not None:
        if strict_min and not (num > min_value):
            raise ValueError(f"'{name}' must be > {min_value}, got {value!r}")
        if not strict_min and not (num >= min_value):
            raise ValueError(f"'{name}' must be >= {min_value}, got {value!r}")
    if max_value is not None and not (num <= max_value):
        raise ValueError(f"'{name}' must be <= {max_value}, got {value!r}")
    return num


def _validate_bgr_tuple(name: str, value: Any) -> tuple[int, int, int]:
    """Validate and convert a BGR color sequence into a 3-int tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"'{name}' must be a 3-element BGR sequence, got {value!r}")
    channels: list[int] = []
    for ch in value:
        if isinstance(ch, bool) or not isinstance(ch, int) or not (0 <= ch <= 255):
            raise ValueError(f"'{name}' channels must be integers in [0, 255], got {value!r}")
        channels.append(ch)
    return (channels[0], channels[1], channels[2])


def _validate_int_list(name: str, value: Any) -> list[int]:
    """Validate that a value is a list of non-boolean integers."""
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"'{name}' must be a list of integers, got {value!r}")
    return list(value)


@dataclass
class DetectorConfig:
    """Object detection settings for YOLO11."""

    weights: str = "yolo11s.pt"
    conf: float = 0.25
    ball_conf: float = 0.15
    imgsz: int = 640
    device: str = "auto"
    player_class_ids: list[int] = field(default_factory=lambda: [0])
    ball_class_ids: list[int] = field(default_factory=lambda: [32])
    min_player_box_height_px: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, int) and not isinstance(self.device, bool):
            self.device = str(self.device)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError(f"'detector.device' must be a non-empty string, got {self.device!r}")
        if not isinstance(self.weights, str) or not self.weights.strip():
            raise ValueError(f"'detector.weights' must be a non-empty string, got {self.weights!r}")
        self.conf = _validate_float("detector.conf", self.conf, min_value=0.0, max_value=1.0)
        self.ball_conf = _validate_float(
            "detector.ball_conf", self.ball_conf, min_value=0.0, max_value=1.0
        )
        self.imgsz = _validate_int("detector.imgsz", self.imgsz, min_value=1)
        self.min_player_box_height_px = _validate_int(
            "detector.min_player_box_height_px", self.min_player_box_height_px, min_value=0
        )
        self.player_class_ids = _validate_int_list(
            "detector.player_class_ids", self.player_class_ids
        )
        self.ball_class_ids = _validate_int_list("detector.ball_class_ids", self.ball_class_ids)


@dataclass
class TrackerConfig:
    """Multi-object player tracking settings for ByteTrack."""

    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: float | None = None
    minimum_consecutive_frames: int = 1
    scene_cut_threshold: float | None = 0.6

    def __post_init__(self) -> None:
        self.track_activation_threshold = _validate_float(
            "tracker.track_activation_threshold",
            self.track_activation_threshold,
            min_value=0.0,
            max_value=1.0,
        )
        self.minimum_matching_threshold = _validate_float(
            "tracker.minimum_matching_threshold",
            self.minimum_matching_threshold,
            min_value=0.0,
            max_value=1.0,
        )
        self.lost_track_buffer = _validate_int(
            "tracker.lost_track_buffer", self.lost_track_buffer, min_value=0
        )
        if self.frame_rate is not None:
            self.frame_rate = _validate_float(
                "tracker.frame_rate", self.frame_rate, min_value=0.0, strict_min=True
            )
        self.minimum_consecutive_frames = _validate_int(
            "tracker.minimum_consecutive_frames", self.minimum_consecutive_frames, min_value=1
        )
        if self.scene_cut_threshold is not None:
            self.scene_cut_threshold = _validate_float(
                "tracker.scene_cut_threshold",
                self.scene_cut_threshold,
                min_value=0.0,
                max_value=1.0,
                strict_min=True,
            )


@dataclass
class BallConfig:
    """Ball candidate selection and temporal interpolation settings."""

    max_gap_frames: int = 15
    max_jump_px: float = 150.0

    def __post_init__(self) -> None:
        self.max_gap_frames = _validate_int("ball.max_gap_frames", self.max_gap_frames, min_value=0)
        self.max_jump_px = _validate_float(
            "ball.max_jump_px", self.max_jump_px, min_value=0.0, strict_min=True
        )


@dataclass
class TeamConfig:
    """Jersey-color team classification settings."""

    n_init_frames: int = 30
    color_space: str = "lab"
    vote_window: int = 15
    outlier_distance_threshold: float = 45.0
    jersey_top_ratio: float = 0.5
    jersey_center_crop_ratio: float = 0.5
    random_state: int = 42

    def __post_init__(self) -> None:
        self.n_init_frames = _validate_int("team.n_init_frames", self.n_init_frames, min_value=1)
        if self.color_space not in _VALID_COLOR_SPACES:
            raise ValueError(
                f"'team.color_space' must be one of {_VALID_COLOR_SPACES}, got {self.color_space!r}"
            )
        self.vote_window = _validate_int("team.vote_window", self.vote_window, min_value=1)
        self.outlier_distance_threshold = _validate_float(
            "team.outlier_distance_threshold",
            self.outlier_distance_threshold,
            min_value=0.0,
            strict_min=True,
        )
        self.jersey_top_ratio = _validate_float(
            "team.jersey_top_ratio",
            self.jersey_top_ratio,
            min_value=0.0,
            max_value=1.0,
            strict_min=True,
        )
        self.jersey_center_crop_ratio = _validate_float(
            "team.jersey_center_crop_ratio",
            self.jersey_center_crop_ratio,
            min_value=0.0,
            max_value=1.0,
            strict_min=True,
        )
        self.random_state = _validate_int("team.random_state", self.random_state, min_value=0)


@dataclass
class AnnotateConfig:
    """Frame annotation style and color settings."""

    team_0_color_bgr: tuple[int, int, int] = (255, 100, 0)
    team_1_color_bgr: tuple[int, int, int] = (0, 60, 230)
    unknown_color_bgr: tuple[int, int, int] = (160, 160, 160)
    ball_color_bgr: tuple[int, int, int] = (0, 230, 255)
    use_jersey_colors: bool = True
    show_hud: bool = True
    font_scale: float = 0.5
    thickness: int = 2

    def __post_init__(self) -> None:
        for attr in _BGR_FIELDS:
            setattr(self, attr, _validate_bgr_tuple(f"annotate.{attr}", getattr(self, attr)))
        if not isinstance(self.use_jersey_colors, bool):
            raise ValueError(
                f"'annotate.use_jersey_colors' must be a boolean, got {self.use_jersey_colors!r}"
            )
        if not isinstance(self.show_hud, bool):
            raise ValueError(f"'annotate.show_hud' must be a boolean, got {self.show_hud!r}")
        self.font_scale = _validate_float(
            "annotate.font_scale", self.font_scale, min_value=0.0, strict_min=True
        )
        self.thickness = _validate_int("annotate.thickness", self.thickness, min_value=1)


@dataclass
class OutputConfig:
    """Video, CSV, and JSON output settings."""

    video_filename: str = "annotated.mp4"
    tracks_filename: str = "tracks.csv"
    summary_filename: str = "summary.json"
    codec: str = "mp4v"
    stride: int = 1

    def __post_init__(self) -> None:
        for attr in ("video_filename", "tracks_filename", "summary_filename"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"'output.{attr}' must be a non-empty string, got {val!r}")
        if not isinstance(self.codec, str) or len(self.codec) != 4:
            raise ValueError(f"'output.codec' must be a 4-character FourCC, got {self.codec!r}")
        self.stride = _validate_int("output.stride", self.stride, min_value=1)


@dataclass
class AppConfig:
    """Top-level application configuration composed of per-module configs."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    annotate: AnnotateConfig = field(default_factory=AnnotateConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


_SECTION_TYPES: dict[str, type[Any]] = {
    "detector": DetectorConfig,
    "tracker": TrackerConfig,
    "ball": BallConfig,
    "team": TeamConfig,
    "annotate": AnnotateConfig,
    "output": OutputConfig,
}


def _build_section(section_name: str, section_cls: type[T], raw: Any) -> T:
    """Validate section keys and instantiate the corresponding config dataclass."""
    if raw is None:
        return section_cls()
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config section '{section_name}' must be a YAML mapping, got {type(raw).__name__}"
        )
    valid_keys = {f.name for f in fields(section_cls)}  # type: ignore[arg-type]
    unknown_keys = set(raw.keys()) - valid_keys
    if unknown_keys:
        unknown_str = ", ".join(sorted(str(k) for k in unknown_keys))
        valid_str = ", ".join(sorted(valid_keys))
        raise ValueError(
            f"Unknown config key(s) in section '{section_name}': {unknown_str}. "
            f"Valid keys: {valid_str}"
        )
    return section_cls(**raw)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an AppConfig from a YAML file.

    Missing sections or keys fall back to dataclass defaults. Unknown sections
    or keys raise a descriptive ``ValueError``.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        return AppConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a YAML mapping, got {type(raw).__name__}")

    unknown_sections = set(raw.keys()) - set(_SECTION_TYPES.keys())
    if unknown_sections:
        unknown_str = ", ".join(sorted(str(k) for k in unknown_sections))
        valid_str = ", ".join(sorted(_SECTION_TYPES.keys()))
        raise ValueError(f"Unknown config section(s): {unknown_str}. Valid sections: {valid_str}")

    kwargs = {
        name: _build_section(name, cls, raw.get(name)) for name, cls in _SECTION_TYPES.items()
    }
    return AppConfig(**kwargs)


__all__ = [
    "AnnotateConfig",
    "AppConfig",
    "BallConfig",
    "DetectorConfig",
    "OutputConfig",
    "TeamConfig",
    "TrackerConfig",
    "load_config",
]
