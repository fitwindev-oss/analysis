"""
Counter Movement Jump (CMJ) analysis from vertical ground-reaction force.

Method follows the impulse-momentum approach (Linthorne 2001,
McMahon et al. 2018):

  1. Quiet standing phase -> subject body weight (BW) from mean vGRF (5-8 Hz LP).
  2. Unweighting onset = first time GRF drops > (threshold N) below BW.
  3. Eccentric phase = unweighting down to CoM minimum velocity (v_min < 0).
  4. Concentric phase = from v_min up to takeoff (GRF < 20 N).
  5. Flight = GRF < 20 N until GRF rebounds > landing threshold.
  6. Jump height computed two ways:
        h_imp   from take-off velocity: h = v_to^2 / (2g)
        h_flight from flight time:       h = g * T_flight^2 / 8

Metrics reported:
  bw_n, bw_kg
  peak_force_n, peak_force_bw
  peak_rfd_n_s           Rate of Force Development (peak of dF/dt)
  net_impulse_ns         eccentric + concentric impulse (N·s)
  takeoff_velocity_m_s
  jump_height_m_impulse  jump height from v_to
  jump_height_m_flight   jump height from flight time
  peak_power_w           peak concentric power
  eccentric_duration_s
  concentric_duration_s
  flight_time_s
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .common import (
    ForceSession, G, butter_lowpass, numerical_derivative, load_force_session,
)
from .pose2d import (
    load_session_pose2d, window_summary, aggregate_cams,
    resolve_pose_frame,
)


@dataclass
class CMJResult:
    bw_n: float
    bw_kg: float

    peak_force_n: float
    peak_force_bw: float
    peak_rfd_n_s: float
    net_impulse_ns: float

    takeoff_velocity_m_s: float
    jump_height_m_impulse: float
    jump_height_m_flight: float
    peak_power_w: float
    peak_power_bw_w_kg: float

    eccentric_duration_s: float
    concentric_duration_s: float
    flight_time_s: float

    ecc_con_ratio: float              # ecc_dur / con_dur

    # Timings (seconds into the trial)
    t_unweight_onset_s: float
    t_takeoff_s: float
    t_landing_s: float

    # Optional per-camera pose summary from unweight onset to landing.
    pose_per_cam: dict = None    # type: ignore
    pose_mean:    dict = None    # type: ignore

    def to_dict(self) -> dict:
        return asdict(self)


def _find_quiet_standing(force: ForceSession,
                         min_duration_s: float = 1.0,
                         sway_std_n: float = 10.0) -> tuple[int, int]:
    """Find a quiet-standing window at start of trial for BW estimation."""
    n_win = max(int(min_duration_s * force.fs), 50)
    ft = butter_lowpass(force.total, 10.0, force.fs)
    best = (0, n_win, np.inf)
    step = max(n_win // 5, 5)
    for i in range(0, len(ft) - n_win, step):
        seg = ft[i:i + n_win]
        if seg.mean() < 100:
            continue  # no person on plate
        sd = float(seg.std())
        if sd < best[2]:
            best = (i, i + n_win, sd)
        if sd < sway_std_n:
            return (i, i + n_win)
    return (best[0], best[1])


def analyze_cmj(force: ForceSession,
                bw_override_kg: Optional[float] = None,
                unweight_threshold_n: float = 30.0,
                flight_threshold_n: float = 20.0) -> CMJResult:
    """
    Full CMJ analysis. Expects a single trial containing:
      quiet standing (>=1 s) -> unweight/countermovement -> propulsion ->
      flight -> landing.
    """
    fs = force.fs
    # Low-pass the vGRF to suppress high-freq noise (keep RFD peaks)
    f = butter_lowpass(force.total, 50.0, fs, order=4)
    t = force.t_s

    # ── Body weight from quiet standing ─────────────────────────────────────
    if bw_override_kg is not None:
        bw_n = bw_override_kg * G
    else:
        i0, i1 = _find_quiet_standing(force)
        bw_n = float(np.mean(f[i0:i1]))
        # Sanity: a plausible adult has BW >= 400 N (~40 kg). If auto-detect
        # finds something lower, the subject likely wasn't fully on the plate
        # during the quiet window. Fail loudly rather than produce garbage.
        if bw_n < 400.0:
            raise RuntimeError(
                f"auto-detected body weight = {bw_n:.0f} N ({bw_n/G:.1f} kg) "
                f"is implausibly low. The subject was probably not fully on "
                f"the plate during the quiet-standing phase. Re-record with "
                f"subject standing still from t=0, or pass --bw-kg <value> "
                f"to override.")
    bw_kg = bw_n / G

    # ── Unweighting onset: first sustained drop below BW - threshold ───────
    below = f < (bw_n - unweight_threshold_n)
    # first index of sustained (>50ms) drop
    onset_idx = None
    run_min = max(int(0.05 * fs), 3)
    run = 0
    for i, v in enumerate(below):
        if v:
            run += 1
            if run >= run_min:
                onset_idx = i - run + 1
                break
        else:
            run = 0
    if onset_idx is None:
        raise RuntimeError("no unweighting phase detected; is this a CMJ trial?")

    # ── Takeoff: GRF drops below flight_threshold ───────────────────────────
    take_idx = None
    for i in range(onset_idx, len(f)):
        if f[i] < flight_threshold_n:
            take_idx = i
            break
    if take_idx is None:
        raise RuntimeError("no takeoff detected (GRF never dropped below flight threshold)")

    # ── Landing: GRF rises back above a landing threshold ──────────────────
    land_thr = max(bw_n * 0.5, 150.0)
    land_idx = None
    for i in range(take_idx + 1, len(f)):
        if f[i] > land_thr:
            land_idx = i
            break
    if land_idx is None:
        # fallback: end of trial
        land_idx = len(f) - 1

    # ── Velocity via impulse-momentum from unweighting onset to takeoff ─────
    m = bw_kg
    dt = 1.0 / fs
    # net force during propulsion
    net_f = f - bw_n
    # integrate from onset (assume v=0 at onset)
    v = np.zeros_like(f)
    for i in range(onset_idx + 1, take_idx + 1):
        v[i] = v[i - 1] + (net_f[i - 1] / m) * dt
    v_takeoff = float(v[take_idx])
    # Find v_min (deepest negative velocity = end of eccentric phase)
    v_segment = v[onset_idx:take_idx + 1]
    v_min_idx_rel = int(np.argmin(v_segment))
    v_min_idx = onset_idx + v_min_idx_rel

    # ── Net impulse from onset to takeoff ───────────────────────────────────
    net_impulse = float(np.sum(net_f[onset_idx:take_idx]) * dt)

    # ── Phases ──────────────────────────────────────────────────────────────
    ecc_dur = float((v_min_idx - onset_idx) / fs)
    con_dur = float((take_idx - v_min_idx) / fs)

    # ── Peak force, peak RFD, peak power ───────────────────────────────────
    peak_f = float(np.max(f[onset_idx:take_idx]))
    rfd = numerical_derivative(f, fs)
    peak_rfd = float(np.max(rfd[onset_idx:take_idx]))

    # Power = F * v   (positive during concentric)
    power = f * v
    peak_power = float(np.max(power[onset_idx:take_idx]))

    # ── Jump heights ────────────────────────────────────────────────────────
    h_impulse = max(v_takeoff, 0.0) ** 2 / (2 * G)
    flight_time = float((land_idx - take_idx) / fs)
    h_flight = G * flight_time ** 2 / 8.0

    # ── Per-camera pose summary over the jump window ───────────────────────
    pose_per_cam: dict = {}
    pose_mean: dict = {}
    pose2d_by_cam = load_session_pose2d(force.session_dir)
    if pose2d_by_cam:
        t0 = float(t[onset_idx]); t1 = float(t[land_idx])
        for cid, ps in pose2d_by_cam.items():
            f0 = resolve_pose_frame(t0, force.session_dir, cid, ps.fps)
            f1 = resolve_pose_frame(t1, force.session_dir, cid, ps.fps) + 1
            pose_per_cam[cid] = window_summary(ps, f0, f1)
        pose_mean = aggregate_cams(pose_per_cam)

    return CMJResult(
        bw_n=bw_n, bw_kg=bw_kg,
        peak_force_n=peak_f,
        peak_force_bw=peak_f / bw_n,
        peak_rfd_n_s=peak_rfd,
        net_impulse_ns=net_impulse,
        takeoff_velocity_m_s=v_takeoff,
        jump_height_m_impulse=h_impulse,
        jump_height_m_flight=h_flight,
        peak_power_w=peak_power,
        peak_power_bw_w_kg=peak_power / bw_kg,
        eccentric_duration_s=ecc_dur,
        concentric_duration_s=con_dur,
        flight_time_s=flight_time,
        ecc_con_ratio=ecc_dur / con_dur if con_dur > 0 else float("nan"),
        t_unweight_onset_s=float(t[onset_idx]),
        t_takeoff_s=float(t[take_idx]),
        t_landing_s=float(t[land_idx]),
        pose_per_cam=pose_per_cam,
        pose_mean=pose_mean,
    )


def analyze_cmj_file(session_dir, **kw) -> CMJResult:
    return analyze_cmj(load_force_session(session_dir), **kw)
