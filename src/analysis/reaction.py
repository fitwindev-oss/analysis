"""
Reaction time + balance-recovery analysis.

For each stimulus fired during recording, we compute:
  - response_time_ms      : time from stimulus to force onset (baseline + 5sigma)
  - peak_displacement_mm  : maximum CoP deviation from pre-stimulus baseline
                            during a post-stimulus window (default 3 s)
  - recovery_time_s       : time until the CoP returns and stays within
                            +/- 5 mm of the pre-stimulus baseline for 0.5 s
  - stability_rms_after   : CoP RMS over the last 2 s of the post-stimulus
                            window (how jittery the subject still is)

Per-response-type aggregates are reported (left_shift / right_shift / jump).

stimulus_log.csv columns (written by record_session.py):
    trial_idx, t_wall, t_ns, stimulus_type, response_type
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import ForceSession, butter_lowpass, load_force_session
from .pose2d import (
    load_session_pose2d, window_summary, aggregate_cams,
    resolve_pose_frame,
)


@dataclass
class RTTrial:
    trial_idx: int
    stimulus_type: str
    response_type: str
    t_stim_s: float
    t_response_s: Optional[float]
    rt_ms: Optional[float]
    premature: bool
    no_response: bool
    peak_displacement_mm: Optional[float] = None
    recovery_time_s: Optional[float] = None
    stability_rms_after_mm: Optional[float] = None
    # Per-trial per-camera pose summary over [t_stim, t_stim + post_window_s]
    pose_per_cam: dict = field(default_factory=dict)
    pose_mean:    dict = field(default_factory=dict)


@dataclass
class ReactionResult:
    n_trials: int
    n_valid: int
    n_premature: int
    n_no_response: int

    mean_rt_ms: float
    median_rt_ms: float
    std_rt_ms: float
    min_rt_ms: float
    max_rt_ms: float

    # Balance-recovery aggregates across all valid trials
    mean_peak_displacement_mm: float
    mean_recovery_time_s: float
    mean_stability_rms_after_mm: float

    # Per-response-type breakdown (optional)
    per_response: dict = field(default_factory=dict)

    trials: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trials"] = [asdict(t) for t in self.trials]
        return d


def load_stimulus_log(session_dir: Path) -> pd.DataFrame:
    fp = Path(session_dir) / "stimulus_log.csv"
    if not fp.exists():
        raise FileNotFoundError(f"missing stimulus log: {fp}")
    return pd.read_csv(fp)


def _detect_response_time(f: np.ndarray, i_stim: int, fs: float,
                          baseline_window_s: float, max_rt_s: float,
                          sigma_multiplier: float
                          ) -> Optional[float]:
    """Return response time (s) or None if no response within window."""
    i_base_start = max(0, int(i_stim - baseline_window_s * fs))
    if i_base_start >= i_stim:
        return None
    baseline = f[i_base_start:i_stim]
    if len(baseline) < 5:
        return None
    bl_mean = float(baseline.mean())
    bl_sd   = float(baseline.std()) + 1e-3
    thr = sigma_multiplier * bl_sd
    i_end = min(len(f), int(i_stim + max_rt_s * fs))
    window = f[i_stim:i_end]
    idx_cross = np.where(np.abs(window - bl_mean) > thr)[0]
    if len(idx_cross) == 0:
        return None
    return float(idx_cross[0]) / fs


def _cop_balance_metrics(cop_x: np.ndarray, cop_y: np.ndarray,
                         i_stim: int, fs: float,
                         baseline_window_s: float,
                         post_window_s: float,
                         recovery_band_mm: float,
                         recovery_hold_s: float,
                         ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (peak_displacement_mm, recovery_time_s, stability_rms_after_mm).
    All three may be None if data is invalid around the stimulus.
    """
    i_base0 = max(0, int(i_stim - baseline_window_s * fs))
    i_end   = min(len(cop_x), int(i_stim + post_window_s * fs))
    if i_base0 >= i_stim or i_end - i_stim < int(0.5 * fs):
        return None, None, None

    base_x = cop_x[i_base0:i_stim]
    base_y = cop_y[i_base0:i_stim]
    base_x = base_x[~np.isnan(base_x)]
    base_y = base_y[~np.isnan(base_y)]
    if len(base_x) < 5 or len(base_y) < 5:
        return None, None, None
    bx = float(base_x.mean()); by = float(base_y.mean())

    post_x = cop_x[i_stim:i_end]
    post_y = cop_y[i_stim:i_end]
    valid = ~(np.isnan(post_x) | np.isnan(post_y))
    post_x = post_x[valid]; post_y = post_y[valid]
    if len(post_x) < 5:
        return None, None, None

    disp = np.sqrt((post_x - bx) ** 2 + (post_y - by) ** 2)
    peak = float(disp.max())

    # Recovery: find first time t after which CoP stays within ±recovery_band
    # of baseline for at least recovery_hold_s seconds
    within = disp < recovery_band_mm
    hold_n = max(int(recovery_hold_s * fs), 3)
    recovery_t = None
    for i in range(len(within) - hold_n):
        if within[i:i + hold_n].all():
            recovery_t = i / fs
            break

    # Stability RMS over the last 2 s of post window
    tail_samples = min(int(2.0 * fs), len(disp))
    if tail_samples > 5:
        tail = disp[-tail_samples:]
        rms_after = float(np.sqrt(np.mean(tail ** 2)))
    else:
        rms_after = None

    return peak, recovery_t, rms_after


def analyze_reaction(force: ForceSession,
                     stimuli: pd.DataFrame,
                     max_rt_s: float = 1.0,
                     premature_cutoff_s: float = 0.10,
                     sigma_multiplier: float = 5.0,
                     baseline_window_s: float = 0.50,
                     post_window_s: float = 3.0,
                     recovery_band_mm: float = 5.0,
                     recovery_hold_s: float = 0.5) -> ReactionResult:
    fs = force.fs
    t_s = force.t_s
    f = butter_lowpass(force.total, 30.0, fs)

    # Align stimulus times to force time base
    if "t_wall" in stimuli.columns:
        t_stim_wall = stimuli["t_wall"].to_numpy(np.float64)
        forces_file = force.session_dir / "forces.csv"
        df_force = pd.read_csv(forces_file, usecols=["t_wall"], nrows=1)
        t0_wall = float(df_force["t_wall"].iloc[0])
        t_stim_s = t_stim_wall - t0_wall
    elif "t_ns" in stimuli.columns:
        t_stim_s = stimuli["t_ns"].to_numpy(np.int64) / 1e9
    else:
        raise RuntimeError("stimulus log needs t_wall or t_ns column")

    stim_types = stimuli.get("stimulus_type",
                             pd.Series(["unknown"] * len(stimuli))).to_numpy()
    resp_types = stimuli.get("response_type",
                             pd.Series([""] * len(stimuli))).to_numpy()

    # Load per-cam 2D pose data (optional)
    pose2d_by_cam = load_session_pose2d(force.session_dir)

    trials: list[RTTrial] = []
    for i, ts in enumerate(t_stim_s):
        i_stim = int(np.searchsorted(t_s, ts))
        rt_s = _detect_response_time(
            f, i_stim, fs, baseline_window_s, max_rt_s, sigma_multiplier)

        # Balance metrics are computed regardless of RT detection success,
        # because we still want to see how CoP moved post-stimulus.
        peak_mm, rec_s, rms_after = _cop_balance_metrics(
            force.cop_x, force.cop_y, i_stim, fs,
            baseline_window_s, post_window_s,
            recovery_band_mm, recovery_hold_s,
        )

        # Per-trial 2D pose summary over [ts, ts + post_window_s]
        pose_per_cam: dict = {}
        pose_mean: dict = {}
        if pose2d_by_cam:
            t0 = float(ts); t1 = float(ts + post_window_s)
            for cid, ps in pose2d_by_cam.items():
                f0 = resolve_pose_frame(t0, force.session_dir, cid, ps.fps)
                f1 = resolve_pose_frame(t1, force.session_dir, cid, ps.fps) + 1
                pose_per_cam[cid] = window_summary(ps, f0, f1)
            pose_mean = aggregate_cams(pose_per_cam)

        if rt_s is None:
            trials.append(RTTrial(
                trial_idx=i, stimulus_type=str(stim_types[i]),
                response_type=str(resp_types[i]),
                t_stim_s=float(ts),
                t_response_s=None, rt_ms=None,
                premature=False, no_response=True,
                peak_displacement_mm=peak_mm,
                recovery_time_s=rec_s,
                stability_rms_after_mm=rms_after,
                pose_per_cam=pose_per_cam,
                pose_mean=pose_mean,
            ))
            continue

        is_prem = rt_s < premature_cutoff_s
        trials.append(RTTrial(
            trial_idx=i, stimulus_type=str(stim_types[i]),
            response_type=str(resp_types[i]),
            t_stim_s=float(ts),
            t_response_s=float(ts + rt_s),
            rt_ms=rt_s * 1000.0,
            premature=is_prem, no_response=False,
            peak_displacement_mm=peak_mm,
            recovery_time_s=rec_s,
            stability_rms_after_mm=rms_after,
            pose_per_cam=pose_per_cam,
            pose_mean=pose_mean,
        ))

    # Overall aggregates
    valid = [t for t in trials if t.rt_ms is not None and not t.premature]
    rts = np.array([t.rt_ms for t in valid], dtype=np.float64)

    if len(rts) == 0:
        mean_rt = median_rt = std_rt = min_rt = max_rt = float("nan")
    else:
        mean_rt   = float(rts.mean())
        median_rt = float(np.median(rts))
        std_rt    = float(rts.std())
        min_rt    = float(rts.min())
        max_rt    = float(rts.max())

    peaks = [t.peak_displacement_mm for t in valid
             if t.peak_displacement_mm is not None]
    recs  = [t.recovery_time_s for t in valid
             if t.recovery_time_s is not None]
    rmss  = [t.stability_rms_after_mm for t in valid
             if t.stability_rms_after_mm is not None]
    mean_peak    = float(np.mean(peaks)) if peaks else float("nan")
    mean_recov   = float(np.mean(recs)) if recs else float("nan")
    mean_rms_aft = float(np.mean(rmss)) if rmss else float("nan")

    # Per-response-type aggregates
    per_resp: dict = {}
    for t in valid:
        key = t.response_type or "unknown"
        per_resp.setdefault(key, []).append(t)
    per_response_summary = {}
    for key, group in per_resp.items():
        rt_vals     = [x.rt_ms                  for x in group if x.rt_ms is not None]
        peak_vals   = [x.peak_displacement_mm   for x in group if x.peak_displacement_mm is not None]
        rec_vals    = [x.recovery_time_s        for x in group if x.recovery_time_s is not None]
        rms_vals    = [x.stability_rms_after_mm for x in group if x.stability_rms_after_mm is not None]
        per_response_summary[key] = {
            "n":                int(len(group)),
            "mean_rt_ms":       float(np.mean(rt_vals))     if rt_vals     else float("nan"),
            "mean_peak_mm":     float(np.mean(peak_vals))   if peak_vals   else float("nan"),
            "mean_recovery_s":  float(np.mean(rec_vals))    if rec_vals    else float("nan"),
            "mean_rms_after":   float(np.mean(rms_vals))    if rms_vals    else float("nan"),
        }

    return ReactionResult(
        n_trials=len(trials),
        n_valid=len(valid),
        n_premature=sum(1 for t in trials if t.premature),
        n_no_response=sum(1 for t in trials if t.no_response),
        mean_rt_ms=mean_rt, median_rt_ms=median_rt,
        std_rt_ms=std_rt, min_rt_ms=min_rt, max_rt_ms=max_rt,
        mean_peak_displacement_mm=mean_peak,
        mean_recovery_time_s=mean_recov,
        mean_stability_rms_after_mm=mean_rms_aft,
        per_response=per_response_summary,
        trials=trials,
    )


def analyze_reaction_file(session_dir, **kw) -> ReactionResult:
    force = load_force_session(session_dir)
    stim = load_stimulus_log(Path(session_dir))
    return analyze_reaction(force, stim, **kw)
