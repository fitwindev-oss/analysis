"""
Static balance / postural sway analysis from CoP trajectory.

Standard posturography metrics (all values in mm unless noted):

  mean_velocity_total   : total path length / duration  (mm/s)
  mean_velocity_ml / _ap: per-axis velocity (mm/s)
  path_length           : total CoP displacement over trial (mm)
  rms_ml / rms_ap       : RMS distance from CoP mean (mm)
  range_ml / range_ap   : full range (mm)
  ellipse95_area        : 95% confidence ellipse (mm^2)
  sway_area_rate        : ellipse area / duration (mm^2/s)

Axis convention (plate world frame):
  X = medio-lateral (ML)    -- left/right
  Y = antero-posterior (AP) -- front/back
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .common import (
    ForceSession, butter_lowpass, confidence_ellipse_area_95,
    load_force_session,
)
from .pose2d import (
    load_session_pose2d, window_summary, aggregate_cams,
    resolve_pose_frame,
)


@dataclass
class BalanceResult:
    duration_s: float
    n_samples: int

    mean_cop_x_mm: float
    mean_cop_y_mm: float

    path_length_mm: float
    mean_velocity_mm_s: float
    mean_velocity_ml_mm_s: float     # X axis
    mean_velocity_ap_mm_s: float     # Y axis

    rms_ml_mm: float
    rms_ap_mm: float
    range_ml_mm: float
    range_ap_mm: float

    ellipse95_area_mm2: float
    sway_area_rate_mm2_s: float

    # Per-board contribution during trial
    mean_board1_pct: float
    mean_board2_pct: float

    # Optional per-camera 2D pose angle summary over the analysis window
    pose_per_cam: dict = None          # type: ignore
    pose_mean:    dict = None          # type: ignore

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_balance(force: ForceSession,
                    t_start: Optional[float] = None,
                    t_end: Optional[float] = None,
                    min_force_n: float = 200.0,
                    cutoff_hz: float = 5.0,
                    auto_trim: bool = True,
                    warmup_s: float = 1.0) -> BalanceResult:
    """
    Compute static-balance / CoP-sway metrics over [t_start, t_end].

    Samples are retained only where:
      - total force >= min_force_n (subject clearly on plate)
      - CoP is within plate footprint (rejects stray division artifacts)
      - both ankles' worth of CoP are present (not NaN)

    The first ``warmup_s`` seconds after the analysis start are discarded
    unconditionally. This matters especially for single-leg stances
    (balance_eo / balance_ec with stance=left|right), where the recorder
    now releases after only 0.5 s of loading and the subject may still be
    settling into the pose during the first second of recording.

    If auto_trim=True (default), the longest contiguous stable-standing
    segment is selected from within [t_start, t_end]; this additionally
    skips any pre/post-recording overhead where the subject is getting on
    or off the plate.

    Range statistics use percentile-based (p5-p95) spans rather than raw
    min-max, to be robust against occasional spurious CoP values.
    """
    import config as _cfg

    if t_start is not None or t_end is not None:
        t0 = t_start if t_start is not None else force.t_s[0]
        t1 = t_end   if t_end   is not None else force.t_s[-1]
        force = force.time_slice(t0, t1)

    # Warmup discard — drop the first `warmup_s` seconds from the analysis.
    if warmup_s and warmup_s > 0 and len(force.t_s) > 0:
        t0 = float(force.t_s[0]) + float(warmup_s)
        if t0 < float(force.t_s[-1]):
            force = force.time_slice(t0, float(force.t_s[-1]))

    # Basic validity mask:
    #  - subject must have >=min_force_n on plate
    #  - CoP must be within plate AND well away from edge (samples snapped to
    #    the edge by the clipping step of recompute/live CoP are artefacts,
    #    not real motion)
    EDGE_MARGIN_MM = 20.0
    mask = (force.total >= min_force_n) & \
           (~np.isnan(force.cop_x)) & (~np.isnan(force.cop_y)) & \
           (force.cop_x > EDGE_MARGIN_MM) & \
           (force.cop_x < _cfg.PLATE_TOTAL_WIDTH_MM  - EDGE_MARGIN_MM) & \
           (force.cop_y > EDGE_MARGIN_MM) & \
           (force.cop_y < _cfg.PLATE_TOTAL_HEIGHT_MM - EDGE_MARGIN_MM)

    # Additional MAD-based outlier rejection: reject samples whose CoP lies
    # > 6*MAD from the median. This catches isolated wild jumps that remain
    # inside plate bounds but are not part of real postural sway.
    if mask.any():
        cx_v = force.cop_x[mask]
        cy_v = force.cop_y[mask]
        med_x = float(np.median(cx_v))
        med_y = float(np.median(cy_v))
        mad_x = float(np.median(np.abs(cx_v - med_x))) + 1e-6
        mad_y = float(np.median(np.abs(cy_v - med_y))) + 1e-6
        # 1.4826 * MAD ~ 1 sigma for Gaussian; use 6*MAD -> ~4 sigma cutoff
        mad_mask = (np.abs(force.cop_x - med_x) < 6.0 * mad_x) & \
                   (np.abs(force.cop_y - med_y) < 6.0 * mad_y)
        mask = mask & mad_mask

    if auto_trim:
        # Pick the longest contiguous run of valid samples
        if mask.any():
            runs = []
            i = 0
            while i < len(mask):
                if mask[i]:
                    j = i
                    while j < len(mask) and mask[j]:
                        j += 1
                    runs.append((i, j))
                    i = j
                else:
                    i += 1
            if runs:
                best = max(runs, key=lambda ij: ij[1] - ij[0])
                run_mask = np.zeros_like(mask)
                run_mask[best[0]:best[1]] = True
                mask = mask & run_mask

    if mask.sum() < 30:
        raise RuntimeError(f"too few valid samples: {mask.sum()}")

    t   = force.t_s[mask]
    cx  = force.cop_x[mask]
    cy  = force.cop_y[mask]
    b1t = force.b1_total[mask]
    b2t = force.b2_total[mask]

    # Low-pass filter
    cx_f = butter_lowpass(cx, cutoff_hz, force.fs)
    cy_f = butter_lowpass(cy, cutoff_hz, force.fs)

    dt = np.diff(t)
    dx = np.diff(cx_f)
    dy = np.diff(cy_f)
    step_dist = np.sqrt(dx * dx + dy * dy)
    path_length = float(step_dist.sum())
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    mean_vel = path_length / duration if duration > 0 else 0.0

    # Per-axis velocity (mean of |dx/dt| etc.)
    v_ml = float(np.mean(np.abs(dx / dt))) if duration > 0 else 0.0
    v_ap = float(np.mean(np.abs(dy / dt))) if duration > 0 else 0.0

    # RMS distance from mean
    cx_c = cx_f - cx_f.mean()
    cy_c = cy_f - cy_f.mean()
    rms_ml = float(np.sqrt(np.mean(cx_c ** 2)))
    rms_ap = float(np.sqrt(np.mean(cy_c ** 2)))

    # Range = p95-p5 (robust against residual outliers)
    range_ml = float(np.percentile(cx_f, 95) - np.percentile(cx_f, 5))
    range_ap = float(np.percentile(cy_f, 95) - np.percentile(cy_f, 5))

    area = confidence_ellipse_area_95(cx_f, cy_f)

    # Board contribution
    total_f = b1t + b2t
    pct_b1 = float(np.mean(b1t / np.where(total_f > 0, total_f, 1.0)) * 100)
    pct_b2 = 100.0 - pct_b1

    # ── 2D pose angle summary over full analysis window ────────────────────
    pose_per_cam: dict = {}
    pose_mean: dict = {}
    pose2d_by_cam = load_session_pose2d(force.session_dir)
    if pose2d_by_cam:
        t0 = float(t[0]); t1 = float(t[-1])
        for cid, ps in pose2d_by_cam.items():
            f0 = resolve_pose_frame(t0, force.session_dir, cid, ps.fps)
            f1 = resolve_pose_frame(t1, force.session_dir, cid, ps.fps) + 1
            pose_per_cam[cid] = window_summary(ps, f0, f1)
        pose_mean = aggregate_cams(pose_per_cam)

    return BalanceResult(
        duration_s=duration, n_samples=int(len(t)),
        mean_cop_x_mm=float(cx_f.mean()),
        mean_cop_y_mm=float(cy_f.mean()),
        path_length_mm=path_length,
        mean_velocity_mm_s=float(mean_vel),
        mean_velocity_ml_mm_s=v_ml,
        mean_velocity_ap_mm_s=v_ap,
        rms_ml_mm=rms_ml, rms_ap_mm=rms_ap,
        range_ml_mm=range_ml, range_ap_mm=range_ap,
        ellipse95_area_mm2=area,
        sway_area_rate_mm2_s=area / duration if duration > 0 else 0.0,
        mean_board1_pct=pct_b1, mean_board2_pct=pct_b2,
        pose_per_cam=pose_per_cam,
        pose_mean=pose_mean,
    )


def romberg_ratio(result_eyes_open: BalanceResult,
                  result_eyes_closed: BalanceResult,
                  metric: str = "mean_velocity_mm_s") -> float:
    """
    Romberg ratio = metric(closed) / metric(open). Higher means more
    dependent on vision for balance.
    """
    return getattr(result_eyes_closed, metric) / getattr(result_eyes_open, metric)


def analyze_balance_file(session_dir, t_start=None, t_end=None,
                         min_force_n=30.0, cutoff_hz=10.0) -> BalanceResult:
    """Convenience: load session and run analyze_balance."""
    force = load_force_session(session_dir)
    return analyze_balance(force, t_start, t_end, min_force_n, cutoff_hz)
