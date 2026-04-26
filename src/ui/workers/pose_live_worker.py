"""
PoseLiveWorker — realtime BlazePose overlay for one selected camera.

Consumes BGR frames pushed by the caller (typically from RecordWorker's
``camera_frame`` signal), runs MediaPipe BlazePose at a constrained rate,
and emits ``pose_overlay(cam_id, kpts33_px, vis33)`` for the UI.

Design:
  * One worker per MeasureTab session (not per camera).
  * Processes only the configured cam_id; other cam_ids are ignored.
  * Keeps only the latest frame — in-flight frames are discarded when new
    frames arrive faster than inference can keep up (drop-on-full). This
    keeps the overlay fresh at the cost of skipped frames.
  * MediaPipe instance lives inside ``run()`` (thread-local), torn down
    on stop.

Signals:
    pose_overlay(cam_id: str, kpts33_px: np.ndarray, vis33: np.ndarray)
    log_message(str)
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

import config
from src.pose.mediapipe_backend import MPPoseDetector


class PoseLiveWorker(QThread):
    pose_overlay = pyqtSignal(str, object, object)   # cam_id, kpts33_px, vis33
    log_message  = pyqtSignal(str)

    def __init__(self, cam_id: str,
                 complexity: Optional[int] = None,
                 parent=None):
        super().__init__(parent)
        self._cam_id = str(cam_id)
        self._complexity = (config.POSE_REALTIME_COMPLEXITY
                            if complexity is None else int(complexity))
        # Single-slot mailbox (latest frame wins)
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._frame_available = threading.Event()
        self._stop = threading.Event()

    # ── public (GUI-thread) API ────────────────────────────────────────────
    def feed_frame(self, cam_id: str, bgr: np.ndarray) -> None:
        """Called on each camera_frame signal. Non-blocking; drops oldest."""
        if cam_id != self._cam_id or bgr is None:
            return
        with self._lock:
            # copy so we're not holding the caller's buffer after they reuse it
            self._latest = bgr.copy()
        self._frame_available.set()

    def stop(self) -> None:
        self._stop.set()
        self._frame_available.set()

    # ── QThread entry ──────────────────────────────────────────────────────
    def run(self) -> None:
        detector: Optional[MPPoseDetector] = None
        try:
            def _dl(msg: str) -> None:
                self.log_message.emit(f"[pose-live] {msg}")
            detector = MPPoseDetector(
                complexity=self._complexity,
                running_mode="video",
                lr_swap=bool(getattr(config, "CAMERA_MIRROR", False)),
                progress_cb=_dl,
            )
        except Exception as e:
            self.log_message.emit(f"[pose-live] init failed: {e}")
            return

        self.log_message.emit(
            f"[pose-live] start cam={self._cam_id} complexity={self._complexity}")
        ts_ms = 0
        try:
            while not self._stop.is_set():
                # Block until a frame is available or stop is requested
                if not self._frame_available.wait(timeout=0.2):
                    continue
                with self._lock:
                    frame = self._latest
                    self._latest = None
                    self._frame_available.clear()
                if frame is None:
                    continue
                ts_ms += max(1, int(1000.0 / 30.0))
                try:
                    r = detector.detect(frame, timestamp_ms=ts_ms)
                except Exception as e:
                    self.log_message.emit(f"[pose-live] detect err: {e}")
                    continue
                # Even if r.ok is False we emit empties so the overlay clears.
                self.pose_overlay.emit(self._cam_id, r.kpts33, r.vis33)
        finally:
            try:
                if detector is not None:
                    detector.close()
            except Exception:
                pass
            self.log_message.emit("[pose-live] stopped")
