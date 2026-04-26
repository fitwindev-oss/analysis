"""
Multi-camera synchronized capture worker.

Design goals:
  - One subprocess per camera (Python GIL + OpenCV blocking reads).
  - Each frame stamped with `time.monotonic_ns()` at read time.
  - Frames pushed to a shared Queue so main/inference processes consume them.
  - Optional disk recording: writes one .mp4 per camera + a timestamps.csv.

This module is the offline-capable backbone: whether we do realtime inference
or just dump video to disk, the same recorder is used.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config


@dataclass
class CameraFrame:
    cam_id: str
    frame_idx: int
    t_ns: int       # time.monotonic_ns() at capture
    bgr: np.ndarray


_ROTATION_CODES = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _orient(frame: np.ndarray, rotation: int, mirror: bool) -> np.ndarray:
    """Apply rotation (deg, clockwise) then horizontal mirror, if enabled."""
    if rotation in _ROTATION_CODES:
        frame = cv2.rotate(frame, _ROTATION_CODES[rotation])
    if mirror:
        frame = cv2.flip(frame, 1)
    return frame


def _camera_process(cam_id: str, cam_index: int, out_queue: mp.Queue,
                    stop_event, record_path: str | None,
                    width: int, height: int, fps: int,
                    rotation: int = 0, mirror: bool = False) -> None:
    """Runs in a child process. Opens one camera, streams frames to queue.

    rotation / mirror are applied per frame BEFORE it is written to mp4 or
    pushed to the queue, so every downstream consumer (GUI preview,
    realtime pose, post-record pose, replay) sees the same oriented frame.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        out_queue.put({"cam_id": cam_id, "error": "open_failed"})
        return
    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   config.CAMERA_BUFFERSIZE)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.CAMERA_AUTO_EXPOSURE)
    cap.set(cv2.CAP_PROP_EXPOSURE,      config.CAMERA_EXPOSURE)

    # Output frame dimensions after rotation (90/270 swap width↔height).
    if rotation in (90, 270):
        out_w, out_h = height, width
    else:
        out_w, out_h = width, height

    writer: cv2.VideoWriter | None = None
    ts_file = None
    ts_writer = None
    if record_path:
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(record_path), fourcc, fps, (out_w, out_h))
        ts_path = record_path.with_suffix(".timestamps.csv")
        ts_file = open(ts_path, "w", newline="")
        ts_writer = csv.writer(ts_file)
        ts_writer.writerow(["frame_idx", "t_monotonic_ns", "t_wall_s"])

    frame_idx = 0
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.001)
                continue
            t_ns = time.monotonic_ns()
            t_wall = time.time()
            frame = _orient(frame, rotation, mirror)
            if writer is not None:
                writer.write(frame)
                ts_writer.writerow([frame_idx, t_ns, t_wall])
            # Drop-on-full: real-time wins over buffering everything
            try:
                out_queue.put_nowait({
                    "cam_id": cam_id,
                    "frame_idx": frame_idx,
                    "t_ns": t_ns,
                    "shape": frame.shape,
                    "bgr": frame,
                })
            except Exception:
                pass
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if ts_file is not None:
            ts_file.close()


class MultiCameraCapture:
    """Spawn one capture process per camera. Frames arrive on a shared queue."""

    def __init__(self, cameras: list[dict] | None = None,
                 record_dir: str | Path | None = None):
        self.cameras = cameras or config.CAMERAS
        self.record_dir = Path(record_dir) if record_dir else None
        self._stop = mp.Event()
        self._queue: mp.Queue = mp.Queue(maxsize=30)   # ~1s @ 30 fps × 3 cams
        self._procs: list[mp.Process] = []

    def start(self):
        self._stop.clear()
        rotation = int(getattr(config, "CAMERA_ROTATION", 0))
        mirror   = bool(getattr(config, "CAMERA_MIRROR", False))
        for cam in self.cameras:
            rec = None
            if self.record_dir:
                rec = str(self.record_dir / f"{cam['id']}.mp4")
            p = mp.Process(
                target=_camera_process,
                args=(cam["id"], cam["index"], self._queue, self._stop, rec,
                      config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS,
                      rotation, mirror),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

    def signal_stop(self) -> None:
        """Set the stop event without waiting for workers to exit.

        Use this when you need to stop multiple recording sources
        simultaneously and want the stop signals to land back-to-back
        on the wall clock (instead of serialised by a 2 s join wait).
        Always pair with ``wait_join()`` afterwards to actually drain
        the worker processes and flush the mp4 writers.
        """
        self._stop.set()

    def wait_join(self, timeout: float = 2.0) -> None:
        """Wait for worker processes to exit.

        Idempotent: safe to call when no workers are running. Workers
        that don't exit within ``timeout`` are terminated forcefully so
        a hung camera can never block session finalisation.
        """
        for p in self._procs:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
        self._procs.clear()

    def stop(self, timeout: float = 2.0):
        """Backward-compatible signal+join in one call.

        Equivalent to ``signal_stop()`` followed by ``wait_join(timeout)``.
        Prefer the split form in code paths that stop multiple hardware
        streams simultaneously (see ``SessionRecorder._finalize``).
        """
        self.signal_stop()
        self.wait_join(timeout)

    def get(self, timeout: float = 0.1) -> dict | None:
        try:
            return self._queue.get(timeout=timeout)
        except Exception:
            return None

    @property
    def queue(self) -> mp.Queue:
        return self._queue
