"""
MediaPipe BlazePose backend (Tasks API).

Runs on CPU by default. Exposes a uniform ``MPPoseDetector`` used by both
offline (``PoseWorker``) and realtime (``PoseLiveWorker``) paths.

Thread-safety:
    A single ``MPPoseDetector`` wraps one ``PoseLandmarker`` instance whose
    internal graph keeps per-frame tracking state. It is NOT safe to share
    across threads — each worker thread must create its own instance and
    call ``close()`` when done.

Model files:
    MediaPipe Tasks requires a ``.task`` file per model complexity. On first
    use the file is auto-downloaded to ``config.POSE_MODEL_CACHE_DIR``.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config


# ─────────────────────────────────────────────────────────────────────────────
# 33-landmark constants (BlazePose ordering)
# ─────────────────────────────────────────────────────────────────────────────

MP33: dict[str, int] = {
    "nose":             0,
    "left_eye_inner":   1,
    "left_eye":         2,
    "left_eye_outer":   3,
    "right_eye_inner":  4,
    "right_eye":        5,
    "right_eye_outer":  6,
    "left_ear":         7,
    "right_ear":        8,
    "mouth_left":       9,
    "mouth_right":      10,
    "left_shoulder":    11,
    "right_shoulder":   12,
    "left_elbow":       13,
    "right_elbow":      14,
    "left_wrist":       15,
    "right_wrist":      16,
    "left_pinky":       17,
    "right_pinky":      18,
    "left_index":       19,
    "right_index":      20,
    "left_thumb":       21,
    "right_thumb":      22,
    "left_hip":         23,
    "right_hip":        24,
    "left_knee":        25,
    "right_knee":       26,
    "left_ankle":       27,
    "right_ankle":      28,
    "left_heel":        29,
    "right_heel":       30,
    "left_foot_index":  31,
    "right_foot_index": 32,
}
MP33_NAMES: list[str] = [
    name for name, _ in sorted(MP33.items(), key=lambda kv: kv[1])
]

# Left/Right index pairs — used to un-swap labels when the input image is
# horizontally mirrored (see ``MPPoseDetector(lr_swap=True)``).
# BlazePose was trained on non-mirrored images; inference on mirrored input
# yields anatomically-flipped L/R labels, which this permutation corrects.
MP33_LR_PAIRS: list[tuple[int, int]] = [
    (1, 4), (2, 5), (3, 6),          # eyes inner/mid/outer
    (7, 8),                          # ears
    (9, 10),                         # mouth corners
    (11, 12), (13, 14), (15, 16),    # shoulder / elbow / wrist
    (17, 18), (19, 20), (21, 22),    # pinky / index / thumb
    (23, 24), (25, 26), (27, 28),    # hip / knee / ankle
    (29, 30), (31, 32),              # heel / foot_index
]


def _build_mirror_permutation() -> np.ndarray:
    perm = np.arange(33, dtype=np.int64)
    for a, b in MP33_LR_PAIRS:
        perm[a], perm[b] = b, a
    return perm


MP33_MIRROR_PERM = _build_mirror_permutation()


# Drawing connections — pared down from MediaPipe's default (hand/face
# connectors removed; we only render torso + limbs + feet).
MP33_CONNECTIONS: list[tuple[int, int]] = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


# ─────────────────────────────────────────────────────────────────────────────
# Frame result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MPPoseFrame:
    ok:        bool                 # True when at least one person was detected
    kpts33:    np.ndarray            # (33, 2) pixel coords in image frame
    vis33:     np.ndarray            # (33,)   visibility / presence score [0, 1]
    world33:   np.ndarray            # (33, 3) world coords (meters, hip-centered)

    @classmethod
    def empty(cls) -> "MPPoseFrame":
        return cls(
            ok=False,
            kpts33=np.full((33, 2), np.nan, dtype=np.float32),
            vis33=np.zeros((33,), dtype=np.float32),
            world33=np.full((33, 3), np.nan, dtype=np.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model file management
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_URLS: dict[int, tuple[str, str]] = {
    # complexity -> (filename, download URL)
    0: ("pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    1: ("pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    2: ("pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"),
}


def ensure_model(complexity: int,
                 progress_cb: Optional[callable] = None) -> Path:
    """Return a local path to the ``.task`` file for ``complexity``.

    Downloads once on first use; subsequent calls hit the cache. Raises on
    unknown complexity or network failure.
    """
    if complexity not in _MODEL_URLS:
        raise ValueError(
            f"POSE complexity must be 0/1/2, got {complexity}")
    fname, url = _MODEL_URLS[complexity]
    cache_dir = Path(config.POSE_MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / fname
    if path.exists() and path.stat().st_size > 1024:
        return path

    tmp = path.with_suffix(".task.part")
    if progress_cb is not None:
        progress_cb(f"다운로드 중: {fname}")

    # Throttle the callback to every 10% so the log doesn't flood with
    # thousands of per-chunk lines.
    _last_bucket = [-1]

    def _hook(block_num, block_size, total_size):
        if progress_cb is None or total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100.0, 100.0 * done / total_size)
        bucket = int(pct // 10)         # 0..10
        if bucket == _last_bucket[0]:
            return
        _last_bucket[0] = bucket
        progress_cb(
            f"{fname}  {pct:3.0f}%  "
            f"({done/1e6:.1f} / {total_size/1e6:.1f} MB)"
        )

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_hook)
        os.replace(tmp, path)
    except Exception as e:
        if tmp.exists():
            try: tmp.unlink()
            except Exception: pass
        raise RuntimeError(f"failed to download {fname}: {e}") from e
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class MPPoseDetector:
    """Per-thread MediaPipe pose detector wrapper.

    Instantiate once per worker thread. Call ``detect(bgr)`` on each frame.
    Call ``close()`` (or use as context manager) when done.
    """

    def __init__(self,
                 complexity: int = 1,
                 running_mode: str = "video",
                 min_det_conf: Optional[float] = None,
                 min_track_conf: Optional[float] = None,
                 lr_swap: bool = False,
                 progress_cb: Optional[callable] = None):
        """Create a detector instance.

        Args:
            lr_swap: Set True when the INPUT frames are horizontally
                mirrored (e.g. ``CAMERA_MIRROR=True`` in config). Applies a
                left↔right index permutation to the output so "left_wrist"
                really is the subject's anatomical left wrist, not the
                image-left wrist.
        """
        self.complexity = int(complexity)
        self.running_mode = running_mode   # "video" | "image" | "live_stream"
        self._lr_swap = bool(lr_swap)
        self._ts_ms = 0

        # Lazy import so unit tests don't drag mediapipe in unless needed
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        model_path = ensure_model(self.complexity, progress_cb=progress_cb)

        mode_map = {
            "image":       vision.RunningMode.IMAGE,
            "video":       vision.RunningMode.VIDEO,
            "live_stream": vision.RunningMode.LIVE_STREAM,
        }
        if running_mode not in mode_map:
            raise ValueError(f"running_mode must be image/video/live_stream")

        det = (config.POSE_MIN_DET_CONF
               if min_det_conf is None else float(min_det_conf))
        trk = (config.POSE_MIN_TRACK_CONF
               if min_track_conf is None else float(min_track_conf))

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path)),
            running_mode=mode_map[running_mode],
            num_poses=1,
            min_pose_detection_confidence=det,
            min_pose_presence_confidence=det,
            min_tracking_confidence=trk,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mp = mp

    # ── lifecycle ──────────────────────────────────────────────────────────
    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    # ── detection ──────────────────────────────────────────────────────────
    def detect(self, bgr: np.ndarray,
               timestamp_ms: Optional[int] = None) -> MPPoseFrame:
        """Run pose detection on one BGR frame. Returns MPPoseFrame."""
        if bgr is None or bgr.size == 0:
            return MPPoseFrame.empty()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == "image":
            result = self._landmarker.detect(mp_image)
        else:
            if timestamp_ms is None:
                self._ts_ms += 1   # monotonic per call
                ts = self._ts_ms
            else:
                ts = int(timestamp_ms)
                self._ts_ms = ts
            if self.running_mode == "video":
                result = self._landmarker.detect_for_video(mp_image, ts)
            else:   # live_stream
                self._landmarker.detect_async(mp_image, ts)
                # LIVE_STREAM returns via a callback that the caller must
                # have configured; we don't support it here.
                return MPPoseFrame.empty()

        if not result.pose_landmarks:
            return MPPoseFrame.empty()

        # One person (num_poses=1)
        lmks = result.pose_landmarks[0]
        kpts = np.full((33, 2), np.nan, dtype=np.float32)
        vis  = np.zeros((33,),   dtype=np.float32)
        for i, lm in enumerate(lmks[:33]):
            # x, y are normalized to [0, 1]
            kpts[i, 0] = float(lm.x) * w
            kpts[i, 1] = float(lm.y) * h
            # Task API exposes both visibility and presence; take the min
            # so one failing channel is enough to downweight the point.
            vis[i] = float(min(lm.visibility or 0.0, lm.presence or 0.0))

        world = np.full((33, 3), np.nan, dtype=np.float32)
        if result.pose_world_landmarks:
            wlmks = result.pose_world_landmarks[0]
            for i, lm in enumerate(wlmks[:33]):
                world[i] = (float(lm.x), float(lm.y), float(lm.z))

        # Mirror compensation — permute L↔R indices so labels reflect the
        # subject's anatomical left/right, not the image-left/right.
        if self._lr_swap:
            kpts  = kpts[MP33_MIRROR_PERM]
            vis   = vis[MP33_MIRROR_PERM]
            world = world[MP33_MIRROR_PERM]

        return MPPoseFrame(ok=True, kpts33=kpts, vis33=vis, world33=world)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience for analysis code
# ─────────────────────────────────────────────────────────────────────────────

def idx(name: str) -> int:
    """Short accessor — MP33['left_knee'] → idx('left_knee')."""
    return MP33[name]
