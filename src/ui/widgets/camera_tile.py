"""Single-camera preview tile used in the 4-column top row."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

import config


class CameraTile(QWidget):
    """Live preview of a single camera. Optionally overlays 2D keypoints."""

    frame_captured = pyqtSignal(object)   # emits (cam_id, timestamp_ns, BGR frame)

    def __init__(self, cam_id: str, cam_index: int, label: str, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.cam_index = cam_index
        self.label = label
        self._cap: cv2.VideoCapture | None = None
        self._active = False
        self._overlay_keypoints: np.ndarray | None = None

        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        header = QLabel(f"{self.cam_id} · {self.label}")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            "color:#90caf9; font-weight:bold; font-size:11px; padding:2px;"
        )
        layout.addWidget(header)

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet(
            "background:#111; border:1px solid #333; color:#666;"
        )
        self._img.setMinimumSize(config.TILE_WIDTH, config.TILE_HEIGHT)
        self._img.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._show_placeholder("No Signal")
        layout.addWidget(self._img, stretch=1)

        self._status = QLabel("Disconnected")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color:#777; font-size:10px;")
        layout.addWidget(self._status)

    # ── public API ───────────────────────────────────────────────────────────
    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = None
            self._set_status(f"idx {self.cam_index} open failed", ok=False)
            return False
        # Enforce a common resolution for all 3 cams (Samsung caps at 480p).
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   config.CAMERA_BUFFERSIZE)
        # Manual exposure (try — some cams ignore). If auto-exposure slips,
        # calibration degrades as focal-length/FOV can shift slightly.
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.CAMERA_AUTO_EXPOSURE)
        self._cap.set(cv2.CAP_PROP_EXPOSURE,      config.CAMERA_EXPOSURE)

        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._active = True
        self._timer.start(int(1000 / config.CAMERA_FPS))
        self._set_status(f"{aw}x{ah} @ {config.CAMERA_FPS}fps", ok=True)
        return True

    def stop(self) -> None:
        self._timer.stop()
        self._active = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._show_placeholder("Stopped")
        self._set_status("Disconnected", ok=False)

    def set_keypoint_overlay(self, kpts: np.ndarray | None) -> None:
        """Set 2D keypoints to overlay on next frame. kpts: (N, 2) pixel coords."""
        self._overlay_keypoints = kpts

    # ── internals ────────────────────────────────────────────────────────────
    def _grab(self):
        if not (self._cap and self._active):
            return
        import time
        t_ns = time.monotonic_ns()
        ok, frame = self._cap.read()
        if not ok:
            return
        self.frame_captured.emit((self.cam_id, t_ns, frame))
        self._render(frame)

    def _render(self, bgr: np.ndarray):
        vis = bgr
        if self._overlay_keypoints is not None:
            vis = bgr.copy()
            for (x, y) in self._overlay_keypoints:
                if not np.isnan(x):
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._img.width(), self._img.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._img.setPixmap(pix)

    def _show_placeholder(self, text: str):
        w, h = config.TILE_WIDTH, config.TILE_HEIGHT
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(blank, f"{self.cam_id}", (10, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        cv2.putText(blank, text, (10, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        img = QImage(blank.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._img.setPixmap(QPixmap.fromImage(img))

    def _set_status(self, text: str, ok: bool):
        color = "#66BB6A" if ok else "#EF5350"
        self._status.setText(text)
        self._status.setStyleSheet(f"color:{color}; font-size:10px;")
