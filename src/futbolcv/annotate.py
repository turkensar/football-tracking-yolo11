"""Frame annotation overlays for tracked players, ball marker, and HUD."""

from __future__ import annotations

import cv2
import numpy as np

from futbolcv.config import AnnotateConfig
from futbolcv.types import BallState, FrameResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_DARK_BORDER_BGR = (20, 20, 20)
_LIGHT_TEXT_BGR = (255, 255, 255)
_DARK_TEXT_BGR = (15, 15, 15)
_HUD_BG_BGR = (20, 20, 20)
_HUD_ALPHA = 0.55
_LUMINANCE_THRESHOLD = 145.0


def _text_color_for_bg(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Choose dark or white text color based on perceived BGR background luminance."""
    luminance = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    return _DARK_TEXT_BGR if luminance >= _LUMINANCE_THRESHOLD else _LIGHT_TEXT_BGR


def centers_to_bgr(
    cluster_centers: np.ndarray,
    *,
    color_space: str = "lab",
) -> dict[int, tuple[int, int, int]]:
    """Convert fitted ``(2, 3)`` Lab or HSV cluster centers to BGR color tuples."""
    if color_space not in ("lab", "hsv"):
        raise ValueError(f"Unsupported color_space {color_space!r}; expected 'lab' or 'hsv'")

    centers = np.asarray(cluster_centers, dtype=np.float64)
    if centers.shape != (2, 3):
        raise ValueError(f"Expected cluster_centers with shape (2, 3), got {centers.shape}")

    clipped = centers.copy()
    if color_space == "hsv":
        clipped[:, 0] = np.clip(np.round(clipped[:, 0]), 0, 179)
        clipped[:, 1:] = np.clip(np.round(clipped[:, 1:]), 0, 255)
        conversion = cv2.COLOR_HSV2BGR
    else:
        clipped = np.clip(np.round(clipped), 0, 255)
        conversion = cv2.COLOR_LAB2BGR

    swatch = clipped.astype(np.uint8).reshape(1, 2, 3)
    bgr_swatch = cv2.cvtColor(swatch, conversion)[0]
    return {
        0: (int(bgr_swatch[0, 0]), int(bgr_swatch[0, 1]), int(bgr_swatch[0, 2])),
        1: (int(bgr_swatch[1, 0]), int(bgr_swatch[1, 1]), int(bgr_swatch[1, 2])),
    }


class Annotator:
    """Draws player foot ellipses, ID badges, ball markers, and top-left HUD."""

    def __init__(
        self,
        cfg: AnnotateConfig,
        *,
        cluster_centers: np.ndarray | None = None,
        color_space: str = "lab",
    ) -> None:
        self.cfg = cfg
        self._team_colors: dict[int, tuple[int, int, int]] = {
            0: cfg.team_0_color_bgr,
            1: cfg.team_1_color_bgr,
        }
        if cfg.use_jersey_colors and cluster_centers is not None:
            self.set_team_colors_from_centers(cluster_centers, color_space=color_space)

    @property
    def team_colors(self) -> dict[int, tuple[int, int, int]]:
        """Return active BGR colors for team ``0`` and team ``1``."""
        return dict(self._team_colors)

    def set_team_colors_from_centers(
        self,
        cluster_centers: np.ndarray | None,
        *,
        color_space: str = "lab",
    ) -> None:
        """Update team colors from KMeans centers when ``cfg.use_jersey_colors`` is enabled."""
        if not self.cfg.use_jersey_colors or cluster_centers is None:
            self._team_colors = {
                0: self.cfg.team_0_color_bgr,
                1: self.cfg.team_1_color_bgr,
            }
            return
        self._team_colors = centers_to_bgr(cluster_centers, color_space=color_space)

    def _color_for_team(self, team_id: int) -> tuple[int, int, int]:
        """Resolve BGR drawing color for ``team_id`` (``-1`` and unknown map to gray)."""
        return self._team_colors.get(team_id, self.cfg.unknown_color_bgr)

    def _draw_player(
        self,
        canvas: np.ndarray,
        xyxy: np.ndarray,
        *,
        tracker_id: int | None,
        team_id: int,
    ) -> None:
        """Draw a broadcast-style foot ellipse and optional ID badge for one player."""
        frame_h, frame_w = canvas.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round(y2))
        box_w = max(12, int(round(x2 - x1)))

        rx = max(8, int(round(box_w * 0.55)))
        ry = max(4, int(round(box_w * 0.22)))
        color = self._color_for_team(team_id)

        cv2.ellipse(
            canvas,
            center=(cx, cy),
            axes=(rx, ry),
            angle=0.0,
            startAngle=-45.0,
            endAngle=235.0,
            color=color,
            thickness=self.cfg.thickness,
            lineType=cv2.LINE_AA,
        )

        if tracker_id is None:
            return

        label = str(int(tracker_id))
        text_thickness = max(1, self.cfg.thickness - 1)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _FONT, self.cfg.font_scale, text_thickness
        )
        pad_x = 5
        pad_y = 3
        badge_w = text_w + 2 * pad_x
        badge_h = text_h + 2 * pad_y

        rect_x1 = max(0, min(frame_w - badge_w, cx - badge_w // 2))
        rect_y1 = max(0, min(frame_h - badge_h, cy + max(2, ry // 2)))
        rect_x2 = min(frame_w - 1, rect_x1 + badge_w)
        rect_y2 = min(frame_h - 1, rect_y1 + badge_h)

        cv2.rectangle(canvas, (rect_x1, rect_y1), (rect_x2, rect_y2), color, thickness=-1)
        cv2.rectangle(canvas, (rect_x1, rect_y1), (rect_x2, rect_y2), _DARK_BORDER_BGR, thickness=1)

        text_x = rect_x1 + pad_x
        text_y = rect_y2 - pad_y - max(0, baseline // 4)
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            _FONT,
            self.cfg.font_scale,
            _text_color_for_bg(color),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_ball(self, canvas: np.ndarray, ball: BallState) -> None:
        """Draw a downward triangle above the ball (filled if detected, hollow if interpolated)."""
        if ball.xyxy is None:
            return

        x1, y1, x2, _ = (float(v) for v in ball.xyxy)
        bx = int(round((x1 + x2) * 0.5))
        tip_y = max(16, int(round(y1)) - 4)
        half_w = 9
        tri_h = 14

        pts = np.array(
            [
                [bx, tip_y],
                [bx - half_w, tip_y - tri_h],
                [bx + half_w, tip_y - tri_h],
            ],
            dtype=np.int32,
        )

        if ball.interpolated:
            cv2.polylines(
                canvas,
                [pts],
                isClosed=True,
                color=self.cfg.ball_color_bgr,
                thickness=self.cfg.thickness,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.fillPoly(canvas, [pts], color=self.cfg.ball_color_bgr, lineType=cv2.LINE_AA)
            cv2.polylines(
                canvas,
                [pts],
                isClosed=True,
                color=_DARK_BORDER_BGR,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def _draw_hud(self, canvas: np.ndarray, frame_idx: int, fps: float | None) -> None:
        """Draw frame index and optional FPS inside a translucent top-left badge."""
        text = f"Frame: {frame_idx} | FPS: {fps:.1f}" if fps is not None else f"Frame: {frame_idx}"
        hud_scale = max(0.5, self.cfg.font_scale * 1.15)
        text_thickness = max(1, self.cfg.thickness - 1)
        (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, hud_scale, text_thickness)

        margin = 10
        pad_x = 8
        pad_y = 6
        x1 = margin
        y1 = margin
        x2 = min(canvas.shape[1], x1 + text_w + 2 * pad_x)
        y2 = min(canvas.shape[0], y1 + text_h + baseline + 2 * pad_y)
        if x2 <= x1 or y2 <= y1:
            return

        roi = canvas[y1:y2, x1:x2]
        backdrop = np.full_like(roi, _HUD_BG_BGR)
        cv2.addWeighted(backdrop, _HUD_ALPHA, roi, 1.0 - _HUD_ALPHA, 0.0, dst=roi)

        cv2.putText(
            canvas,
            text,
            (x1 + pad_x, y1 + pad_y + text_h),
            _FONT,
            hud_scale,
            _LIGHT_TEXT_BGR,
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    def annotate(
        self,
        frame: np.ndarray,
        result: FrameResult,
        *,
        fps: float | None = None,
    ) -> np.ndarray:
        """Return an annotated copy of ``frame`` without mutating the input array."""
        canvas = frame.copy()

        players = result.players
        count = len(players)
        if count > 0:
            tracker_ids = players.tracker_id
            team_ids = players.data.get("team_id")
            for idx in range(count):
                tid = None if tracker_ids is None else int(tracker_ids[idx])
                team_id = -1 if team_ids is None else int(team_ids[idx])
                self._draw_player(canvas, players.xyxy[idx], tracker_id=tid, team_id=team_id)

        self._draw_ball(canvas, result.ball)

        if self.cfg.show_hud:
            self._draw_hud(canvas, result.frame_idx, fps)

        return canvas


__all__ = ["Annotator", "centers_to_bgr"]
