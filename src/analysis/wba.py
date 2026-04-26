"""
Weight Bearing Asymmetry (WBA) analysis.

Compares left/right force plate loading to detect asymmetry:

  wba_pct       = 100 * |F_L - F_R| / (F_L + F_R)  (absolute asymmetry, %)
  load_ratio    = F_L / F_R
  symmetry_idx  = 100 * (F_smaller / F_larger)     (>95% = near symmetric)

Board1 = LEFT plate, Board2 = RIGHT plate (per project convention).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .common import ForceSession, load_force_session


@dataclass
class WBAResult:
    duration_s: float
    n_samples: int

    mean_f_left_n: float           # Board1
    mean_f_right_n: float          # Board2
    mean_total_n: float

    mean_left_pct: float           # (F_L / F_total) * 100
    mean_right_pct: float
    mean_wba_pct: float            # 100 * |F_L - F_R| / F_total

    std_wba_pct: float             # variability over trial
    max_wba_pct: float

    load_ratio_l_over_r: float
    symmetry_index_pct: float      # (min/max) * 100

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_wba(force: ForceSession,
                t_start: Optional[float] = None,
                t_end: Optional[float] = None,
                min_force_n: float = 50.0) -> WBAResult:
    if t_start is not None or t_end is not None:
        t0 = t_start if t_start is not None else force.t_s[0]
        t1 = t_end   if t_end   is not None else force.t_s[-1]
        force = force.time_slice(t0, t1)

    # Clip per-board total to >=0 (vertical force is physically non-negative;
    # negatives come from per-channel drift after zero-calibration).
    fl_raw = force.b1_total
    fr_raw = force.b2_total
    fl = np.maximum(fl_raw, 0.0)
    fr = np.maximum(fr_raw, 0.0)
    ft = fl + fr

    mask = ft >= min_force_n
    if mask.sum() < 10:
        raise RuntimeError(f"too few weight-bearing samples: {mask.sum()}")

    fl = fl[mask]; fr = fr[mask]; ft = ft[mask]

    mean_fl = float(fl.mean()); mean_fr = float(fr.mean())
    mean_ft = mean_fl + mean_fr

    # per-sample WBA%
    wba_pct_series = 100.0 * np.abs(fl - fr) / ft
    mean_wba = float(wba_pct_series.mean())
    std_wba  = float(wba_pct_series.std())
    max_wba  = float(wba_pct_series.max())

    left_pct  = 100.0 * mean_fl / mean_ft
    right_pct = 100.0 - left_pct
    ratio_lr  = mean_fl / mean_fr if mean_fr > 1 else float("inf")
    sym_idx   = 100.0 * min(mean_fl, mean_fr) / max(mean_fl, mean_fr)

    return WBAResult(
        duration_s=float(force.t_s[mask][-1] - force.t_s[mask][0]),
        n_samples=int(mask.sum()),
        mean_f_left_n=mean_fl, mean_f_right_n=mean_fr, mean_total_n=mean_ft,
        mean_left_pct=left_pct, mean_right_pct=right_pct,
        mean_wba_pct=mean_wba, std_wba_pct=std_wba, max_wba_pct=max_wba,
        load_ratio_l_over_r=float(ratio_lr),
        symmetry_index_pct=float(sym_idx),
    )


def analyze_wba_file(session_dir, **kw) -> WBAResult:
    return analyze_wba(load_force_session(session_dir), **kw)
