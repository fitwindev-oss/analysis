"""
Per-camera 2D pose analysis helpers (MediaPipe BlazePose, 33 landmarks).

Each recorded session contains one `poses_<cam_id>.npz` per camera with
MP33 landmarks + precomputed joint angles. This module provides:

  - joint-angle definitions: 10 triangle joint angles + 2 derived = 12 metrics
    (knee/hip/ankle/shoulder/elbow × L/R, trunk_lean, neck_lean)
  - angle computation from MP33 keypoints
  - series loaders + time alignment (force time -> video frame)
  - window summaries (min/max/range/mean) with per-camera + cross-camera
    rollups, consumed by the analyzers (balance.py, cmj.py, squat.py, …)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.pose.mediapipe_backend import MP33, MP33_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Joint-angle definitions — all 2D (image plane), unsigned, in degrees [0, 180]
# ─────────────────────────────────────────────────────────────────────────────

# (angle_name, proximal_kp, joint_kp, distal_kp)
ANGLE_DEFS: list[tuple[str, str, str, str]] = [
    # Lower body
    ("knee_L",     "left_hip",       "left_knee",      "left_ankle"),
    ("knee_R",     "right_hip",      "right_knee",     "right_ankle"),
    ("hip_L",      "left_shoulder",  "left_hip",       "left_knee"),
    ("hip_R",      "right_shoulder", "right_hip",      "right_knee"),
    ("ankle_L",    "left_knee",      "left_ankle",     "left_foot_index"),
    ("ankle_R",    "right_knee",     "right_ankle",    "right_foot_index"),
    # Upper body
    ("shoulder_L", "left_hip",       "left_shoulder",  "left_elbow"),
    ("shoulder_R", "right_hip",      "right_shoulder", "right_elbow"),
    ("elbow_L",    "left_shoulder",  "left_elbow",     "left_wrist"),
    ("elbow_R",    "right_shoulder", "right_elbow",    "right_wrist"),
]
# Derived (not a 3-point joint angle)
DERIVED_ANGLES: list[str] = ["trunk_lean", "neck_lean"]

ANGLE_NAMES: list[str] = [n for n, *_ in ANGLE_DEFS] + DERIVED_ANGLES   # 12
N_ANGLES = len(ANGLE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Angle math
# ─────────────────────────────────────────────────────────────────────────────

def _joint_angle_deg(p_prox: np.ndarray, p_joint: np.ndarray,
                     p_dist: np.ndarray) -> float:
    """Unsigned 2D angle in degrees at p_joint, in [0, 180]."""
    v1 = p_prox - p_joint
    v2 = p_dist - p_joint
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-6:
        return float("nan")
    dot = float(v1[0] * v2[0] + v1[1] * v2[1])
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    return math.degrees(math.atan2(abs(cross), dot))


def _trunk_lean_deg(kpts33: np.ndarray) -> float:
    """Angle of shoulder-midpoint → hip-midpoint from image-vertical."""
    ls = kpts33[MP33["left_shoulder"]]
    rs = kpts33[MP33["right_shoulder"]]
    lh = kpts33[MP33["left_hip"]]
    rh = kpts33[MP33["right_hip"]]
    if (np.any(np.isnan(ls)) or np.any(np.isnan(rs))
            or np.any(np.isnan(lh)) or np.any(np.isnan(rh))):
        return float("nan")
    shoulder = 0.5 * (ls + rs)
    hip      = 0.5 * (lh + rh)
    dx = float(shoulder[0] - hip[0])
    dy = float(shoulder[1] - hip[1])    # image +y points down
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _neck_lean_deg(kpts33: np.ndarray) -> float:
    """Angle of nose → shoulder-midpoint from image-vertical."""
    nose = kpts33[MP33["nose"]]
    ls   = kpts33[MP33["left_shoulder"]]
    rs   = kpts33[MP33["right_shoulder"]]
    if (np.any(np.isnan(nose)) or np.any(np.isnan(ls))
            or np.any(np.isnan(rs))):
        return float("nan")
    shoulder = 0.5 * (ls + rs)
    dx = float(nose[0] - shoulder[0])
    dy = float(nose[1] - shoulder[1])
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def compute_angles_frame(kpts33: np.ndarray, vis33: np.ndarray,
                         conf_thresh: float = 0.3) -> np.ndarray:
    """Compute the 12 angles for a single frame.

    kpts33: (33, 2) pixel coords (NaN for missing)
    vis33:  (33,)   visibility / confidence in [0, 1]

    Returns a (12,) float32 array with NaN where any required keypoint
    has visibility < conf_thresh or coords are NaN.
    """
    out = np.full((N_ANGLES,), np.nan, dtype=np.float32)
    # Triangle angles
    for i, (_name, prox, joint, dist) in enumerate(ANGLE_DEFS):
        ip, ij, id_ = MP33[prox], MP33[joint], MP33[dist]
        if (vis33[ip] < conf_thresh or vis33[ij] < conf_thresh
                or vis33[id_] < conf_thresh):
            continue
        p = kpts33[ip]; j = kpts33[ij]; d = kpts33[id_]
        if np.any(np.isnan(p)) or np.any(np.isnan(j)) or np.any(np.isnan(d)):
            continue
        out[i] = _joint_angle_deg(p, j, d)
    # Derived
    base = len(ANGLE_DEFS)
    trunk_keys = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if all(vis33[MP33[k]] >= conf_thresh for k in trunk_keys):
        out[base + 0] = _trunk_lean_deg(kpts33)
    neck_keys = ("nose", "left_shoulder", "right_shoulder")
    if all(vis33[MP33[k]] >= conf_thresh for k in neck_keys):
        out[base + 1] = _neck_lean_deg(kpts33)
    return out


def compute_angles_timeseries(kpts33: np.ndarray, vis33: np.ndarray,
                              conf_thresh: float = 0.3) -> np.ndarray:
    """Per-frame wrapper for compute_angles_frame.

    kpts33: (N, 33, 2), vis33: (N, 33). Returns (N, 12) float32.
    """
    n = int(kpts33.shape[0])
    out = np.full((n, N_ANGLES), np.nan, dtype=np.float32)
    for i in range(n):
        out[i] = compute_angles_frame(kpts33[i], vis33[i], conf_thresh)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Session loader + time alignment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pose2DSeries:
    cam_id:       str
    kpts_mp33:    np.ndarray          # (N, 33, 2) pixel coords
    vis_mp33:     np.ndarray          # (N, 33)    visibility / presence
    world_mp33:   np.ndarray          # (N, 33, 3) world coords (m)
    angles:       np.ndarray          # (N, 14)
    angle_names:  list[str]
    fps:          float
    image_size:   tuple[int, int]     # (W, H)
    backend:      str = "mediapipe"
    model_complexity: int = 1

    def __len__(self) -> int:
        return int(self.kpts_mp33.shape[0])

    @classmethod
    def load(cls, path: Path) -> "Pose2DSeries":
        z = np.load(path, allow_pickle=False)
        names_arr = z.get("angle_names") if hasattr(z, "get") else None
        # np.load returns NpzFile which supports `.files`; `get` may not exist
        if "angle_names" in z.files:
            names = [str(x) for x in z["angle_names"].tolist()]
        else:
            names = ANGLE_NAMES[:]
        n = int(z["kpts_mp33"].shape[0])
        world = (z["world_mp33"].astype(np.float32)
                 if "world_mp33" in z.files
                 else np.full((n, 33, 3), np.nan, dtype=np.float32))
        return cls(
            cam_id=str(z["cam_id"]),
            kpts_mp33=z["kpts_mp33"].astype(np.float32),
            vis_mp33=z["visibility_mp33"].astype(np.float32),
            world_mp33=world,
            angles=z["angles"].astype(np.float32),
            angle_names=names,
            fps=float(z["fps"]),
            image_size=tuple(z["image_size"].tolist()),
            backend=str(z["backend"]) if "backend" in z.files else "mediapipe",
            model_complexity=int(z["model_complexity"])
                if "model_complexity" in z.files else 1,
        )


def load_session_pose2d(session_dir: str | Path) -> dict[str, Pose2DSeries]:
    """Load every poses_<cam>.npz file in the session directory.

    Returns { cam_id: Pose2DSeries }. Empty dict if no pose files exist.
    Legacy poses2d_<cam>.npz files (RTMPose) are ignored.
    """
    session_dir = Path(session_dir)
    out: dict[str, Pose2DSeries] = {}
    for p in sorted(session_dir.glob("poses_*.npz")):
        try:
            series = Pose2DSeries.load(p)
            out[series.cam_id] = series
        except Exception:
            continue
    return out


def get_wait_offset_s(session_dir: str | Path) -> float:
    """Read the recorder's wait_duration_s from session.json (0 if missing).

    This is the gap between t=0 of the video and t=0 of forces.csv (force
    recording starts after the smart-wait phase completes).
    """
    try:
        meta = json.loads((Path(session_dir) / "session.json").read_text(
            encoding="utf-8"))
        return float(meta.get("wait_duration_s", 0.0) or 0.0)
    except Exception:
        return 0.0


def get_record_start_wall_s(session_dir: str | Path) -> Optional[float]:
    """Wall-clock seconds when force recording started.

    Written by SessionRecorder into session.json. Returned value is
    directly comparable to the `t_wall_s` column of `<cam>.timestamps.csv`.
    """
    try:
        meta = json.loads((Path(session_dir) / "session.json").read_text(
            encoding="utf-8"))
        v = meta.get("record_start_wall_s")
        return float(v) if v is not None else None
    except Exception:
        return None


def load_video_timestamps(session_dir: str | Path,
                          cam_id: str) -> Optional[np.ndarray]:
    """Per-frame wall-clock seconds from ``<cam_id>.timestamps.csv``.

    Returns an ``(N,) float64`` array aligned 1:1 with the video's frames,
    or ``None`` if the file does not exist.
    """
    path = Path(session_dir) / f"{cam_id}.timestamps.csv"
    if not path.exists():
        return None
    try:
        import csv
        walls: list[float] = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                walls.append(float(row["t_wall_s"]))
        if not walls:
            return None
        return np.asarray(walls, dtype=np.float64)
    except Exception:
        return None


def resolve_pose_frame(t_force_s: float,
                       session_dir: str | Path,
                       cam_id: str,
                       pose_fps_fallback: float) -> int:
    """Map force-timeline seconds → video-frame index with hard sync.

    Primary path — use the per-frame wall-clock timestamps written by
    ``camera_worker.py`` (``<cam>.timestamps.csv``) together with the
    recorder's ``record_start_wall_s``. This is accurate regardless of
    the camera's actual capture rate, dropped frames, or per-camera
    start-up skew.

    Fallback path (if either file is missing) — old fps-based math:
    ``(t_force + wait_offset) * fps``. This is what the analyzer used
    before 2026-04-22 and matches pre-sync sessions.
    """
    walls = load_video_timestamps(session_dir, cam_id)
    rec_start = get_record_start_wall_s(session_dir)
    if walls is not None and rec_start is not None and len(walls) > 0:
        t_target = rec_start + float(t_force_s)
        return int(np.argmin(np.abs(walls - t_target)))
    wait_s = get_wait_offset_s(session_dir)
    return int(round((float(t_force_s) + wait_s) * float(pose_fps_fallback)))


# Legacy name kept for callers that haven't migrated yet. Prefer
# resolve_pose_frame(session_dir, cam_id, ...) — it's timestamp-aware.
def force_time_to_pose_frame(t_force_s: float, wait_offset_s: float,
                             fps: float) -> int:
    """Deprecated: fps-math-only mapping. See resolve_pose_frame()."""
    return int(round((float(t_force_s) + float(wait_offset_s)) * float(fps)))


# ─────────────────────────────────────────────────────────────────────────────
# Window summaries
# ─────────────────────────────────────────────────────────────────────────────

def window_summary(pose: Pose2DSeries, i_frame_start: int, i_frame_end: int,
                   angle_names: Optional[list[str]] = None) -> dict:
    """Per-angle min/max/range/mean over [i_frame_start, i_frame_end)."""
    i0 = max(0, i_frame_start)
    i1 = min(len(pose), max(i_frame_end, i0 + 1))
    seg = pose.angles[i0:i1]
    names = angle_names or pose.angle_names
    out: dict = {}
    for j, name in enumerate(names):
        if j >= seg.shape[1]:
            continue
        col = seg[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            out[name] = {"min": None, "max": None, "range": None, "mean": None}
            continue
        out[name] = {
            "min":   float(col.min()),
            "max":   float(col.max()),
            "range": float(col.max() - col.min()),
            "mean":  float(col.mean()),
        }
    return out


def window_summary_for_force_time(pose: Pose2DSeries,
                                  session_dir: str | Path,
                                  t_force_start: float,
                                  t_force_end: float) -> dict:
    """Same as window_summary() but takes force-timeline seconds instead of
    frame indices. Uses ``resolve_pose_frame`` so the mapping is
    timestamp-accurate whenever ``<cam>.timestamps.csv`` is present."""
    i0 = resolve_pose_frame(t_force_start, session_dir, pose.cam_id, pose.fps)
    i1 = resolve_pose_frame(t_force_end,   session_dir, pose.cam_id, pose.fps) + 1
    return window_summary(pose, i0, i1)


def aggregate_cams(per_cam: dict[str, dict]) -> dict:
    """Cross-camera mean of each (angle, stat) pair."""
    if not per_cam:
        return {}
    angle_names = list(next(iter(per_cam.values())).keys())
    out: dict = {}
    for name in angle_names:
        stats_agg: dict = {}
        for stat in ("min", "max", "range", "mean"):
            vals = [per_cam[c][name][stat] for c in per_cam
                    if per_cam[c].get(name, {}).get(stat) is not None]
            stats_agg[stat] = float(np.mean(vals)) if vals else None
        out[name] = stats_agg
    return out
