"""
Proprioception / position reproduction analysis.

Protocol assumption:
  For each trial, the subject is first guided to a TARGET position, then
  asked (eyes closed) to reproduce it. A trial log CSV records
  (trial_idx, t_target_start_ns, t_target_end_ns, t_reproduce_start_ns,
   t_reproduce_end_ns, target_label).

  Position can be tracked via:
    - force CoP on the plate, OR
    - joint keypoint from 3D pose (e.g. an ankle or a wrist)

This module gives two error statistics per trial:
  absolute_error_mm        mean |target - reproduction| distance
  signed_x_error_mm / _y   signed bias
Plus across-trial:
  mean_absolute_error       avg |AE| over trials
  constant_error            mean of signed errors (bias)
  variable_error            std of signed errors (variability)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .common import ForceSession, load_force_session, load_poses3d_world
from .pose2d import (
    load_session_pose2d, window_summary, aggregate_cams,
    resolve_pose_frame,
)


@dataclass
class ProprioTrial:
    trial_idx: int
    label: str
    target_xy_mm: tuple[float, float]
    reproduction_xy_mm: tuple[float, float]
    absolute_error_mm: float
    signed_x_error_mm: float
    signed_y_error_mm: float
    # Per-trial pose summaries for target + reproduction windows
    pose_target_per_cam:       dict = field(default_factory=dict)
    pose_target_mean:          dict = field(default_factory=dict)
    pose_reproduction_per_cam: dict = field(default_factory=dict)
    pose_reproduction_mean:    dict = field(default_factory=dict)
    # Joint-angle reproduction error (degrees) averaged across cameras
    angle_reproduction_error_deg: dict = field(default_factory=dict)


@dataclass
class ProprioResult:
    n_trials: int
    mean_absolute_error_mm: float
    constant_error_x_mm: float
    constant_error_y_mm: float
    variable_error_x_mm: float
    variable_error_y_mm: float
    trials: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trials"] = [asdict(t) for t in self.trials]
        return d


def load_proprio_log(session_dir: Path) -> pd.DataFrame:
    """Columns: trial_idx, t_target_start_ns, t_target_end_ns,
       t_reproduce_start_ns, t_reproduce_end_ns, target_label (optional)."""
    fp = Path(session_dir) / "proprio_log.csv"
    if not fp.exists():
        raise FileNotFoundError(f"missing proprio log: {fp}")
    return pd.read_csv(fp)


def _mean_position_cop(force: ForceSession, t0_s: float, t1_s: float,
                      min_force_n: float = 30.0) -> tuple[float, float]:
    mask = (force.t_s >= t0_s) & (force.t_s <= t1_s) & (force.total > min_force_n)
    if mask.sum() < 3:
        raise RuntimeError(
            f"too few CoP samples in window [{t0_s:.2f}, {t1_s:.2f}]")
    return float(np.nanmean(force.cop_x[mask])), \
           float(np.nanmean(force.cop_y[mask]))


def _mean_position_joint(poses: dict, fps: float, t0_s: float, t1_s: float,
                         joint_idx: int) -> tuple[float, float]:
    kpts = poses["kpts3d"]
    n = len(kpts)
    i0 = int(max(0, t0_s * fps))
    i1 = int(min(n, t1_s * fps))
    if i1 - i0 < 3:
        raise RuntimeError("too few pose samples in window")
    seg = kpts[i0:i1, joint_idx, :2]    # X, Y only
    seg = seg[~np.isnan(seg).any(axis=1)]
    if len(seg) < 3:
        raise RuntimeError("all pose samples NaN in window")
    return float(seg[:, 0].mean()), float(seg[:, 1].mean())


def analyze_proprio(force: ForceSession,
                    trial_log: pd.DataFrame,
                    signal: Literal["cop", "joint"] = "cop",
                    joint_idx: int = 16) -> ProprioResult:
    """
    Compute target vs reproduction errors.

    signal: "cop"  - use force plate center of pressure
            "joint"- use keypoint `joint_idx` from world 3D pose
    """
    poses = None
    fps = 1.0
    if signal == "joint":
        poses = load_poses3d_world(force.session_dir)
        if poses is None:
            raise RuntimeError("joint-signal proprio requires poses3d_world")
        fps = poses["fps"]

    forces_file = force.session_dir / "forces.csv"
    df_force = pd.read_csv(forces_file, usecols=["t_wall"], nrows=1)
    t0_wall = float(df_force["t_wall"].iloc[0])

    # Optional per-cam 2D pose data (from process_pose_for_session.py)
    _pose2d_by_cam = load_session_pose2d(force.session_dir)

    trials: list[ProprioTrial] = []
    for _, row in trial_log.iterrows():
        label = str(row.get("target_label", ""))
        # Accept t_wall or t_ns columns
        if {"t_target_start_wall", "t_target_end_wall"}.issubset(row.index):
            t_target_start = float(row["t_target_start_wall"]) - t0_wall
            t_target_end   = float(row["t_target_end_wall"])   - t0_wall
            t_repro_start  = float(row["t_reproduce_start_wall"]) - t0_wall
            t_repro_end    = float(row["t_reproduce_end_wall"])   - t0_wall
        elif {"t_target_start_ns", "t_target_end_ns"}.issubset(row.index):
            # Assume first force sample's t_ns aligns with t_wall[0]
            t_target_start = float(row["t_target_start_ns"]) / 1e9
            t_target_end   = float(row["t_target_end_ns"])   / 1e9
            t_repro_start  = float(row["t_reproduce_start_ns"]) / 1e9
            t_repro_end    = float(row["t_reproduce_end_ns"])   / 1e9
        else:
            raise RuntimeError(
                "proprio log needs t_target_start_wall / _ns columns")

        if signal == "cop":
            tgt = _mean_position_cop(force, t_target_start, t_target_end)
            rep = _mean_position_cop(force, t_repro_start,  t_repro_end)
        else:
            tgt = _mean_position_joint(poses, fps, t_target_start, t_target_end,
                                       joint_idx)
            rep = _mean_position_joint(poses, fps, t_repro_start, t_repro_end,
                                       joint_idx)

        dx = rep[0] - tgt[0]
        dy = rep[1] - tgt[1]
        ae = float(np.sqrt(dx * dx + dy * dy))

        # ── 2D pose summary for target + reproduction windows ───────────
        pose_target_per_cam: dict = {}
        pose_repro_per_cam:  dict = {}
        pose_target_mean:    dict = {}
        pose_repro_mean:     dict = {}
        angle_err: dict = {}
        if _pose2d_by_cam:
            for cid, ps in _pose2d_by_cam.items():
                f0a = resolve_pose_frame(t_target_start, force.session_dir, cid, ps.fps)
                f1a = resolve_pose_frame(t_target_end,   force.session_dir, cid, ps.fps) + 1
                f0b = resolve_pose_frame(t_repro_start,  force.session_dir, cid, ps.fps)
                f1b = resolve_pose_frame(t_repro_end,    force.session_dir, cid, ps.fps) + 1
                pose_target_per_cam[cid] = window_summary(ps, f0a, f1a)
                pose_repro_per_cam[cid]  = window_summary(ps, f0b, f1b)
            pose_target_mean = aggregate_cams(pose_target_per_cam)
            pose_repro_mean  = aggregate_cams(pose_repro_per_cam)
            # Mean angle at target vs reproduction -> reproduction error
            for angle_name in pose_target_mean:
                a_tgt = pose_target_mean[angle_name].get("mean")
                a_rep = pose_repro_mean.get(angle_name, {}).get("mean")
                if a_tgt is not None and a_rep is not None:
                    angle_err[angle_name] = float(abs(a_rep - a_tgt))

        trials.append(ProprioTrial(
            trial_idx=int(row.get("trial_idx", len(trials))),
            label=label,
            target_xy_mm=tgt,
            reproduction_xy_mm=rep,
            absolute_error_mm=ae,
            signed_x_error_mm=float(dx),
            signed_y_error_mm=float(dy),
            pose_target_per_cam=pose_target_per_cam,
            pose_target_mean=pose_target_mean,
            pose_reproduction_per_cam=pose_repro_per_cam,
            pose_reproduction_mean=pose_repro_mean,
            angle_reproduction_error_deg=angle_err,
        ))

    if not trials:
        raise RuntimeError("no trials processed")

    aes = np.array([t.absolute_error_mm for t in trials])
    dxs = np.array([t.signed_x_error_mm for t in trials])
    dys = np.array([t.signed_y_error_mm for t in trials])
    return ProprioResult(
        n_trials=len(trials),
        mean_absolute_error_mm=float(aes.mean()),
        constant_error_x_mm=float(dxs.mean()),
        constant_error_y_mm=float(dys.mean()),
        variable_error_x_mm=float(dxs.std()),
        variable_error_y_mm=float(dys.std()),
        trials=trials,
    )


def analyze_proprio_file(session_dir, **kw) -> ProprioResult:
    force = load_force_session(session_dir)
    log = load_proprio_log(Path(session_dir))
    return analyze_proprio(force, log, **kw)
