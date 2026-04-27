"""
CameraView — live preview panel for every available camera.

Consumes frames from RecordWorker.camera_frame(cam_id, bgr, frame_idx, t_ns)
and, when enabled, MediaPipe skeleton overlays from PoseLiveWorker via
``on_pose_overlay(cam_id, kpts33_px, vis33)``. Throttles repaint to
``config.PLOT_UPDATE_MS`` to protect the GUI thread from burst traffic.

Tile count follows ``config.CAMERAS`` at construction time (after startup
camera detection has pruned missing devices).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy,
)

import config
from src.pose.mediapipe_backend import MP33_CONNECTIONS


_VIS_THRESH = 0.5           # drop landmarks with lower visibility
_KPT_COLOR  = (0, 255, 255)  # yellow (BGR)
_EDGE_COLOR = (0, 200, 0)    # green  (BGR)

# V6 — positional cue ring (cognitive_reaction). Drawn as a layered
# circle (outer halo + bright core) to read LED-like at small scales.
# Tuned for the project palette's FITWIN green (#AAF500 ≈ BGR (0,245,170)).
_CUE_HALO_COLOR  = (0, 245, 170)   # BGR — soft halo
_CUE_CORE_COLOR  = (255, 255, 255) # BGR — bright core
_CUE_LABEL_COLOR = (255, 255, 255) # BGR — label text


class _SingleCamTile(QWidget):
    """One cam — header + image area. Keeps latest frame, repaint throttled."""

    def __init__(self, cam_id: str, label: str, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self._latest: Optional[np.ndarray] = None   # BGR
        # Pose overlay buffers (None = don't draw)
        self._kpts:   Optional[np.ndarray] = None
        self._vis:    Optional[np.ndarray] = None
        # V6 — positional cue (normalised image coords). When non-None,
        # draws a glowing ring centered at (x_norm * w, y_norm * h) on
        # every repaint until cleared.
        self._cue_xy: Optional[tuple[float, float]] = None
        self._cue_label: Optional[str] = None
        # ``_cue_phase`` increments on every repaint while a cue is set,
        # so the ring radius can pulse and read more LED-like.
        self._cue_phase: int = 0
        # Dimensions the pose was computed on — may differ from display if the
        # caller feeds us a different-resolution camera. We key all drawing
        # off the frame's own pixel dimensions so rescaling is not needed.

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        header = QLabel(f"{cam_id} · {label}")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            "color:#90caf9; font-weight:bold; font-size:11px; padding:2px;")
        root.addWidget(header)

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet(
            "background:#111; border:1px solid #333; color:#666;")
        self._img.setMinimumSize(240, 180)
        self._img.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._render_placeholder()
        root.addWidget(self._img, stretch=1)

    # ── public API ─────────────────────────────────────────────────────────
    def set_frame(self, bgr: np.ndarray) -> None:
        self._latest = bgr

    def set_pose(self, kpts33: Optional[np.ndarray],
                 vis33: Optional[np.ndarray]) -> None:
        self._kpts = kpts33
        self._vis  = vis33

    def set_positional_cue(self, x_norm: Optional[float],
                            y_norm: Optional[float],
                            label: Optional[str] = None) -> None:
        """Show / hide the V6 positional cue ring on this tile.

        Pass ``None`` for either coord to hide. Coordinates are
        normalised image space (0..1, top-left origin) so they survive
        any resolution change between cameras.
        """
        if x_norm is None or y_norm is None:
            self._cue_xy = None
            self._cue_label = None
            self._cue_phase = 0
            return
        self._cue_xy = (float(x_norm), float(y_norm))
        self._cue_label = label

    def clear(self) -> None:
        self._latest = None
        self._kpts = None
        self._vis = None
        self._cue_xy = None
        self._cue_label = None
        self._cue_phase = 0
        self._render_placeholder()

    def repaint_if_dirty(self) -> None:
        if self._latest is None:
            return
        bgr = self._latest
        self._latest = None
        # Copy once if any overlay is going to draw, so we don't mutate
        # the caller's buffer (recorder thread may still be reading it).
        has_skeleton = (self._kpts is not None and self._vis is not None)
        has_cue      = (self._cue_xy is not None)
        if has_skeleton or has_cue:
            bgr = bgr.copy()
        if has_skeleton:
            self._draw_skeleton(bgr, self._kpts, self._vis)
        if has_cue:
            self._cue_phase = (self._cue_phase + 1) % 60
            self._draw_positional_cue(
                bgr, self._cue_xy, self._cue_label, self._cue_phase)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._img.width(), self._img.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._img.setPixmap(pix)

    # ── drawing ────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_skeleton(bgr: np.ndarray, kpts33: np.ndarray,
                       vis33: np.ndarray) -> None:
        """Draw MP33 landmarks + connections in-place on a BGR frame."""
        if kpts33 is None or vis33 is None:
            return
        if kpts33.shape[0] < 33 or vis33.shape[0] < 33:
            return
        h, w = bgr.shape[:2]
        # Edges first (lines under dots)
        for a, b in MP33_CONNECTIONS:
            if vis33[a] < _VIS_THRESH or vis33[b] < _VIS_THRESH:
                continue
            pa = kpts33[a]; pb = kpts33[b]
            if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                continue
            x1 = int(max(0, min(w - 1, pa[0])))
            y1 = int(max(0, min(h - 1, pa[1])))
            x2 = int(max(0, min(w - 1, pb[0])))
            y2 = int(max(0, min(h - 1, pb[1])))
            cv2.line(bgr, (x1, y1), (x2, y2), _EDGE_COLOR, 2, cv2.LINE_AA)
        # Joint dots
        for i in range(33):
            if vis33[i] < _VIS_THRESH:
                continue
            p = kpts33[i]
            if np.any(np.isnan(p)):
                continue
            x = int(max(0, min(w - 1, p[0])))
            y = int(max(0, min(h - 1, p[1])))
            cv2.circle(bgr, (x, y), 3, _KPT_COLOR, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_positional_cue(bgr: np.ndarray,
                              xy_norm: tuple[float, float],
                              label: Optional[str],
                              phase: int) -> None:
        """Draw an LED-style ring + label at the cued spot.

        ``xy_norm`` is in [0,1] image-normalised coords (top-left origin).
        ``phase`` cycles 0..59 to drive a gentle radius pulse so the cue
        reads as "alive" rather than a static decal.
        """
        h, w = bgr.shape[:2]
        if w <= 1 or h <= 1:
            return
        cx = int(round(xy_norm[0] * (w - 1)))
        cy = int(round(xy_norm[1] * (h - 1)))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        # Base radius scales with the smaller image dimension so the cue
        # stays visible across portrait + landscape camera streams.
        base = max(14, min(w, h) // 18)
        # Pulse: ±15 % over 60 phase ticks (~2 s at 30 Hz)
        pulse = 1.0 + 0.15 * np.sin(2 * np.pi * phase / 60.0)
        r_outer = int(round(base * 1.6 * pulse))
        r_mid   = int(round(base * 1.0 * pulse))
        r_core  = int(round(base * 0.55))
        # Halo — drawn additively for a soft glow effect on darker frames.
        # Use a translucent overlay via copyTo with weighted blend so the
        # underlying camera image still shows through.
        overlay = bgr.copy()
        cv2.circle(overlay, (cx, cy), r_outer, _CUE_HALO_COLOR, -1,
                   cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0, bgr)
        # Crisp ring + bright core
        cv2.circle(bgr, (cx, cy), r_mid,  _CUE_HALO_COLOR, 3, cv2.LINE_AA)
        cv2.circle(bgr, (cx, cy), r_core, _CUE_CORE_COLOR, -1, cv2.LINE_AA)
        # Optional label below the ring
        if label:
            text = label.replace("pos_", "")
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx = max(2, min(w - tw - 2, cx - tw // 2))
            ty = min(h - 4, cy + r_outer + th + 6)
            # Drop-shadow so text reads against any background
            cv2.putText(bgr, text, (tx + 1, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(bgr, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CUE_LABEL_COLOR,
                        2, cv2.LINE_AA)

    def _render_placeholder(self) -> None:
        w, h = 240, 180
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(blank, self.cam_id, (10, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        cv2.putText(blank, "idle", (10, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        img = QImage(blank.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._img.setPixmap(QPixmap.fromImage(img))


class CameraView(QWidget):
    """Row of cam tiles (one per config.CAMERAS entry) + repaint timer.

    T8: minimum size + sizePolicy hints to portrait-rotation 9:16.
    The Logitech StreamCam is mounted in landscape but the capture
    pipeline rotates 90° → portrait 720×1280. Without an aspect-aware
    size hint, the parent layout gives this widget the full available
    width and the video shows with huge left/right letterbox bars.

    By setting Preferred sizePolicy with a min portrait shape, the
    parent layout can choose to keep us narrow (less letterbox).
    """

    # Camera capture is rotated 90° clockwise → final visible aspect
    # 9:16 (height ≈ 1.78 × width).
    _ASPECT_W_OVER_H = 9 / 16   # = 0.5625

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tiles: dict[str, _SingleCamTile] = {}

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        for cam in config.CAMERAS:
            tile = _SingleCamTile(cam["id"], cam["label"])
            self._tiles[cam["id"]] = tile
            lay.addWidget(tile, stretch=1)

        # Tighter aspect-ratio hint so parent layouts don't over-stretch
        # the widget horizontally (T8).
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.MinimumExpanding)
        self.setMinimumSize(220, 380)   # portrait ~9:16 minimum

        # Repaint timer — drives _flush() which pulls latest frames from
        # each tile into the QLabel. Must come AFTER the size policy
        # block so __init__ doesn't return early.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._flush)
        self._timer.start(config.PLOT_UPDATE_MS)   # ~33 ms → 30 fps

    def sizeHint(self) -> QSize:
        # Hint a portrait shape so the parent QHBoxLayout doesn't
        # default to making us very wide. ~340w × 600h is a comfortable
        # 24"-monitor target without letterbox.
        return QSize(340, 600)

    # ── public API ─────────────────────────────────────────────────────────
    def on_camera_frame(self, cam_id: str, bgr: np.ndarray,
                        frame_idx: int, t_ns: int) -> None:
        tile = self._tiles.get(cam_id)
        if tile is not None:
            tile.set_frame(bgr)

    def on_pose_overlay(self, cam_id: str,
                        kpts33: Optional[np.ndarray],
                        vis33: Optional[np.ndarray]) -> None:
        tile = self._tiles.get(cam_id)
        if tile is not None:
            tile.set_pose(kpts33, vis33)

    def clear_overlay(self, cam_id: Optional[str] = None) -> None:
        if cam_id is None:
            for t in self._tiles.values():
                t.set_pose(None, None)
        else:
            t = self._tiles.get(cam_id)
            if t is not None:
                t.set_pose(None, None)

    def set_positional_cue(self, x_norm: Optional[float],
                            y_norm: Optional[float],
                            label: Optional[str] = None,
                            cam_id: Optional[str] = None) -> None:
        """Mirror the V6 cognitive_reaction cue onto every (or one) tile.

        ``x_norm`` / ``y_norm`` are normalised image coords; pass None to
        hide the cue. The recorder fires the cue once per stim from any
        thread; this method is safe to call from the GUI thread via
        signal-slot.
        """
        if cam_id is None:
            for t in self._tiles.values():
                t.set_positional_cue(x_norm, y_norm, label)
        else:
            t = self._tiles.get(cam_id)
            if t is not None:
                t.set_positional_cue(x_norm, y_norm, label)

    def reset(self) -> None:
        for t in self._tiles.values():
            t.clear()

    # ── internals ──────────────────────────────────────────────────────────
    def _flush(self) -> None:
        for t in self._tiles.values():
            t.repaint_if_dirty()
