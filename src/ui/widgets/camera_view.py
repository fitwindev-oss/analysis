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


class _SingleCamTile(QWidget):
    """One cam — header + image area. Keeps latest frame, repaint throttled."""

    def __init__(self, cam_id: str, label: str, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self._latest: Optional[np.ndarray] = None   # BGR
        # Pose overlay buffers (None = don't draw)
        self._kpts:   Optional[np.ndarray] = None
        self._vis:    Optional[np.ndarray] = None
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

    def clear(self) -> None:
        self._latest = None
        self._kpts = None
        self._vis = None
        self._render_placeholder()

    def repaint_if_dirty(self) -> None:
        if self._latest is None:
            return
        bgr = self._latest
        self._latest = None
        # Draw pose overlay in-place on a copy so we don't mutate caller's buffer
        if self._kpts is not None and self._vis is not None:
            bgr = bgr.copy()
            self._draw_skeleton(bgr, self._kpts, self._vis)
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

    def reset(self) -> None:
        for t in self._tiles.values():
            t.clear()

    # ── internals ──────────────────────────────────────────────────────────
    def _flush(self) -> None:
        for t in self._tiles.values():
            t.repaint_if_dirty()
