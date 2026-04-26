"""
VideoPlayerWidget — timestamp-synced playback of a session's camera video.

Takes a session directory + cam_id, opens the mp4 with OpenCV, and displays
the frame closest to whatever force-timeline second is set via
``set_time(t_force_s)``. Optionally overlays the MediaPipe skeleton if the
corresponding ``poses_<cam>.npz`` exists.

Not a free-running player — the PlaybackController drives ``set_time``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

from src.analysis.pose2d import (
    resolve_pose_frame, load_session_pose2d,
)
from src.pose.mediapipe_backend import MP33_CONNECTIONS


_VIS_THRESH = 0.5
_KPT_COLOR  = (0, 255, 255)   # yellow BGR
_EDGE_COLOR = (0, 200, 0)     # green  BGR


class VideoPlayerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._session_dir: Optional[Path] = None
        self._cam_id: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._n_frames: int = 0
        self._current_idx: int = -1
        self._overlay_enabled = False
        self._pose_series = None   # optional Pose2DSeries

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self._img = QLabel("(비디오 없음)")
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet(
            "QLabel { background:#0a0a0a; color:#666; border:1px solid #333; }")
        self._img.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._img.setMinimumHeight(240)
        lay.addWidget(self._img)

    # ── public API ─────────────────────────────────────────────────────────
    def load(self, session_dir: str | Path, cam_id: str) -> bool:
        """Open the video. Returns True on success."""
        self.unload()
        sd = Path(session_dir)
        video_path = sd / f"{cam_id}.mp4"
        if not video_path.exists():
            self._img.setText(f"(비디오 없음: {cam_id}.mp4)")
            return False
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._img.setText(f"(비디오 열기 실패: {cam_id}.mp4)")
            return False
        self._cap = cap
        self._session_dir = sd
        self._cam_id = cam_id
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_idx = -1
        # Load pose series if available (for overlay)
        series = load_session_pose2d(sd)
        self._pose_series = series.get(cam_id) if series else None
        # Show first frame
        self.set_time(0.0)
        return True

    def unload(self) -> None:
        if self._cap is not None:
            try: self._cap.release()
            except Exception: pass
        self._cap = None
        self._session_dir = None
        self._cam_id = None
        self._pose_series = None
        self._current_idx = -1
        self._img.clear()
        self._img.setText("(비디오 없음)")

    def set_overlay_enabled(self, on: bool) -> None:
        self._overlay_enabled = bool(on) and self._pose_series is not None
        # Redraw current frame with new overlay state
        if self._cap is not None and self._current_idx >= 0:
            self._redraw_current()

    def has_pose(self) -> bool:
        return self._pose_series is not None

    def duration_s(self) -> float:
        """Session-relative duration (from force record start to video end).

        Uses timestamps.csv when available so it's accurate even if the
        video's internal fps differs from wall fps."""
        if self._session_dir is None or self._cam_id is None:
            return 0.0
        try:
            from src.analysis.pose2d import (
                load_video_timestamps, get_record_start_wall_s,
            )
            walls = load_video_timestamps(self._session_dir, self._cam_id)
            rec  = get_record_start_wall_s(self._session_dir)
            if walls is not None and rec is not None and len(walls) > 0:
                return max(0.0, float(walls[-1] - rec))
        except Exception:
            pass
        # Fallback — assume nominal fps covers wait + duration
        try:
            import json
            meta = json.loads((self._session_dir / "session.json").read_text(
                encoding="utf-8"))
            return float(meta.get("duration_s", 0.0) or 0.0)
        except Exception:
            return 0.0

    # ── playback hook ──────────────────────────────────────────────────────
    def set_time(self, t_force_s: float) -> None:
        if self._cap is None:
            return
        # Timestamp-based mapping (same as analysis sync)
        frame_idx = resolve_pose_frame(
            float(t_force_s), self._session_dir, self._cam_id, self._fps)
        frame_idx = max(0, min(self._n_frames - 1, frame_idx))
        if frame_idx == self._current_idx:
            return
        # Seek: small forward moves can use sequential read; otherwise set
        if 0 < frame_idx - self._current_idx <= 8:
            target = self._current_idx + 1
            while target <= frame_idx:
                ok, frame = self._cap.read()
                if not ok:
                    return
                target += 1
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self._cap.read()
            if not ok:
                return
        self._current_idx = frame_idx
        self._render(frame)

    # ── rendering ──────────────────────────────────────────────────────────
    def _redraw_current(self) -> None:
        if self._cap is None or self._current_idx < 0:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_idx)
        ok, frame = self._cap.read()
        if ok:
            self._render(frame)

    def _render(self, bgr: np.ndarray) -> None:
        if self._overlay_enabled and self._pose_series is not None \
                and 0 <= self._current_idx < len(self._pose_series):
            kpts = self._pose_series.kpts_mp33[self._current_idx]
            vis  = self._pose_series.vis_mp33[self._current_idx]
            bgr = bgr.copy()
            _draw_skeleton(bgr, kpts, vis)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._img.width(), self._img.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._img.setPixmap(pix)


def _draw_skeleton(bgr: np.ndarray, kpts33: np.ndarray,
                   vis33: np.ndarray) -> None:
    h, w = bgr.shape[:2]
    for a, b in MP33_CONNECTIONS:
        if vis33[a] < _VIS_THRESH or vis33[b] < _VIS_THRESH:
            continue
        pa, pb = kpts33[a], kpts33[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        x1 = int(max(0, min(w - 1, pa[0])))
        y1 = int(max(0, min(h - 1, pa[1])))
        x2 = int(max(0, min(w - 1, pb[0])))
        y2 = int(max(0, min(h - 1, pb[1])))
        cv2.line(bgr, (x1, y1), (x2, y2), _EDGE_COLOR, 2, cv2.LINE_AA)
    for i in range(33):
        if vis33[i] < _VIS_THRESH:
            continue
        p = kpts33[i]
        if np.any(np.isnan(p)):
            continue
        x = int(max(0, min(w - 1, p[0])))
        y = int(max(0, min(h - 1, p[1])))
        cv2.circle(bgr, (x, y), 3, _KPT_COLOR, -1, cv2.LINE_AA)
