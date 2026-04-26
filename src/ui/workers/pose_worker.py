"""
PoseWorker — background 2D pose detection for all cameras of a session.

Runs MediaPipe BlazePose frame-by-frame on each C<N>.mp4 in the session
folder, computes per-frame joint angles, and saves ``poses_<cam>.npz``.

The npz layout (consumed by ``src/analysis/pose2d.py``):

    kpts_mp33       : (N, 33, 2) pixel coords
    visibility_mp33 : (N, 33)    MediaPipe visibility/presence in [0, 1]
    world_mp33      : (N, 33, 3) world-space landmarks (meters, hip-centered)
    angles          : (N, 12)    joint angles in degrees
    angle_names     : (12,) str
    fps, image_size, cam_id
    backend         : "mediapipe"
    model_complexity: 0 / 1 / 2

Signals:
    progress(cam_id: str, frame: int, total: int)
    cam_done(cam_id: str, out_path: str)
    all_done(session_dir: str, success: bool, error: str)
    log_message(str)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

import config
from src.pose.mediapipe_backend import MPPoseDetector
from src.analysis.pose2d import compute_angles_timeseries, ANGLE_NAMES


class PoseWorker(QThread):
    progress    = pyqtSignal(str, int, int)        # cam_id, frame, total
    cam_done    = pyqtSignal(str, str)             # cam_id, out_path
    all_done    = pyqtSignal(str, bool, str)       # session_dir, success, error
    log_message = pyqtSignal(str)

    def __init__(self, session_dir: str | Path,
                 complexity: Optional[int] = None,
                 overwrite: bool = False,
                 parent=None):
        super().__init__(parent)
        self._session_dir = Path(session_dir)
        self._complexity = (config.POSE_POSTRECORD_COMPLEXITY
                            if complexity is None else int(complexity))
        self._overwrite = overwrite
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    # ── QThread entry point ────────────────────────────────────────────────
    def run(self) -> None:
        sdir = self._session_dir
        if not sdir.exists():
            self.all_done.emit(str(sdir), False, f"session not found: {sdir}")
            return

        def _dl(msg: str) -> None:
            self.log_message.emit(f"[pose] {msg}")

        self.log_message.emit(
            f"[pose] session: {sdir.name}  (complexity={self._complexity})")

        any_ok = False
        last_err = ""
        detector: Optional[MPPoseDetector] = None
        try:
            for cam in config.CAMERAS:
                if self._cancel:
                    self.log_message.emit("[pose] cancelled")
                    break
                cid = cam["id"]
                video_path = sdir / f"{cid}.mp4"
                out_path   = sdir / f"poses_{cid}.npz"

                if not video_path.exists():
                    self.log_message.emit(f"  [{cid}] skip (no video)")
                    continue
                if out_path.exists() and not self._overwrite:
                    self.log_message.emit(f"  [{cid}] cached")
                    any_ok = True
                    self.cam_done.emit(cid, str(out_path))
                    continue

                # Recreate the detector per camera. MediaPipe's VIDEO mode
                # requires monotonically increasing timestamps within one
                # PoseLandmarker instance; each camera feed is a fresh
                # sequence that resets the timeline to 0, so reusing the
                # instance across cameras raises
                # "Input timestamp must be monotonically increasing."
                try:
                    if detector is not None:
                        detector.close()
                        detector = None
                    detector = MPPoseDetector(
                        complexity=self._complexity,
                        running_mode="video",
                        lr_swap=bool(getattr(config, "CAMERA_MIRROR", False)),
                        progress_cb=_dl,
                    )
                except Exception as e:
                    last_err = f"[{cid}] detector init failed: {e}"
                    self.log_message.emit(f"  [{cid}] init error: {e}")
                    continue

                try:
                    self._run_on_video(detector, video_path, out_path, cid)
                    any_ok = True
                    self.cam_done.emit(cid, str(out_path))
                except Exception as e:
                    last_err = f"[{cid}] {e}"
                    self.log_message.emit(f"  [{cid}] error: {e}")
        finally:
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass

        self.all_done.emit(str(sdir), any_ok, last_err)

    # ── per-video loop ─────────────────────────────────────────────────────
    def _run_on_video(self, detector: MPPoseDetector,
                      video_path: Path, out_path: Path,
                      cam_id: str) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        fps   = cap.get(cv2.CAP_PROP_FPS) or config.CAMERA_FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_dt_ms = max(1, int(round(1000.0 / float(fps))))

        kpts_all:  list[np.ndarray] = []
        vis_all:   list[np.ndarray] = []
        world_all: list[np.ndarray] = []
        i = 0
        report_every = max(1, total // 30)   # ~30 progress ticks per video
        try:
            ts_ms = 0
            while True:
                if self._cancel:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                mp_frame = detector.detect(frame, timestamp_ms=ts_ms)
                kpts_all.append(mp_frame.kpts33)
                vis_all.append(mp_frame.vis33)
                world_all.append(mp_frame.world33)
                i += 1
                ts_ms += frame_dt_ms
                if (i % report_every) == 0 or i == total:
                    self.progress.emit(cam_id, i, total)
        finally:
            cap.release()

        if not kpts_all:
            raise RuntimeError("no frames produced")

        kpts  = np.stack(kpts_all,  axis=0).astype(np.float32)
        vis   = np.stack(vis_all,   axis=0).astype(np.float32)
        world = np.stack(world_all, axis=0).astype(np.float32)
        angles = compute_angles_timeseries(kpts, vis, conf_thresh=0.3)

        np.savez(
            out_path,
            cam_id=cam_id,
            kpts_mp33=kpts,
            visibility_mp33=vis,
            world_mp33=world,
            angles=angles,
            angle_names=np.array(ANGLE_NAMES),
            fps=float(fps),
            image_size=np.array([w, h], dtype=np.int32),
            backend="mediapipe",
            model_complexity=int(self._complexity),
        )
        self.log_message.emit(
            f"  [{cam_id}] saved {out_path.name}  "
            f"({i} frames, fps={fps:.1f})"
        )
