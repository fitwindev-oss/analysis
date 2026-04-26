"""
Squat / Overhead-squat technique analysis.

Two complementary data streams:
  (a) Force plate - vGRF + CoP -> symmetry, depth surrogate, rep timing
  (b) Encoder    - bar / tool vertical displacement (if a barbell is used)
  (c) 3D skeleton - horizontal joint positions (hip, knee, ankle, arm)

Because the current camera setup yields anisotropic 3D reconstruction, we
report 3D quantities **in the horizontal plane only** (world XY). Vertical
(Z) values from the skeleton are unreliable; we prefer encoder or force-based
depth when available.

Metrics:
  per_rep:
    t_start_s, t_bottom_s, t_end_s
    ecc_duration_s, con_duration_s
    peak_vgrf_n, peak_vgrf_bw
    mean_wba_pct                      avg L/R asymmetry during rep
    depth_from_encoder_mm             bar vertical ROM (if encoder available)
    depth_from_hip_mm                 hip height drop (if skeleton available)
    knee_over_toe_mm_mean             horizontal offset knee-ankle in world
    hip_over_cop_mm_mean              horizontal offset hip-CoP
    symmetry_knee_pct                 L vs R knee forward extension
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .common import (
    ForceSession, butter_lowpass, load_force_session, load_poses3d_world,
)
from .encoder import detect_reps, _encoder_available
from .pose2d import (
    load_session_pose2d, window_summary, aggregate_cams,
    resolve_pose_frame,
)


# COCO-17 joint indices we use
L_HIP, R_HIP     = 11, 12
L_KNEE, R_KNEE   = 13, 14
L_ANKLE, R_ANKLE = 15, 16
L_WRIST, R_WRIST = 9, 10


@dataclass
class SquatRep:
    idx: int
    t_start_s: float
    t_bottom_s: float
    t_end_s: float
    ecc_duration_s: float
    con_duration_s: float

    peak_vgrf_n: float
    peak_vgrf_bw: float
    mean_wba_pct: float

    depth_from_encoder_mm: Optional[float]
    depth_from_hip_mm: Optional[float]
    knee_over_toe_mm_mean: Optional[float]
    hip_over_cop_mm_mean: Optional[float]
    symmetry_knee_pct: Optional[float]

    # ── Patent-derived precision metrics (Phase S1d) ──────────────────
    # RMSE from the session's mean CoP trajectory across reps (mm). High
    # RMSE = this rep's CoP path deviates from the subject's norm.
    rmse_ap_mm:     Optional[float] = None
    rmse_ml_mm:     Optional[float] = None
    # Tempo ratio — eccentric:concentric time ratio. 2.0 = 2 s down / 1 s up.
    tempo_ratio:    Optional[float] = None
    # Impulse L/R asymmetry per phase (%). 0 = perfect, 100 = entire
    # impulse on one side only.
    impulse_asym_ecc_pct: Optional[float] = None
    impulse_asym_con_pct: Optional[float] = None
    # RFD metrics — force-onset-aligned intervals during concentric phase.
    rfd_n_s:        dict = field(default_factory=dict)   # {20: N/s, 40: ..., ...}
    peak_rfd_n_s:   Optional[float] = None
    # VRT (Visual/Auditory Reaction Time, ms) from stim trigger to
    # concentric force onset. None when no stim log is available.
    vrt_ms:         Optional[float] = None

    # Per-camera 2D pose angle summaries over this rep window.
    # Shape: {cam_id -> {angle_name -> {min, max, range, mean}}}
    pose_per_cam: dict = field(default_factory=dict)
    pose_mean:    dict = field(default_factory=dict)


@dataclass
class SquatResult:
    n_reps: int
    bw_n: float
    reps: list = field(default_factory=list)

    mean_depth_mm: float = 0.0
    mean_peak_vgrf_bw: float = 0.0
    mean_wba_pct: float = 0.0

    # ── Session-level precision metrics (Phase S1d) ───────────────────
    # CMC (Coefficient of Multiple Correlation) — CoP trajectory
    # consistency across all reps. 0-1, higher = more repeatable.
    # Separate values per axis (anteroposterior / mediolateral).
    cmc_ap:          Optional[float] = None
    cmc_ml:          Optional[float] = None
    mean_rmse_ap_mm: float = 0.0
    mean_rmse_ml_mm: float = 0.0
    mean_tempo_ratio:        float = 0.0
    mean_impulse_asym_ecc_pct: float = 0.0
    mean_impulse_asym_con_pct: float = 0.0
    mean_peak_rfd_n_s: float = 0.0
    mean_vrt_ms:     Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reps"] = [asdict(r) for r in self.reps]
        return d


def _detect_squat_reps_from_vgrf(total_n: np.ndarray, fs: float, bw: float,
                                 min_rep_s: float = 1.0,
                                 min_prominence_n: float = 40.0,
                                 band_n: float = 10.0,
                                 min_push_above_bw_n: float = 60.0,
                                 ) -> list[tuple[int, int, int]]:
    """
    Detect squat reps via local-minima (bottom-of-squat) detection.

    For each valley i_bot in the smoothed vGRF:
      - i_start = latest index before i_bot where v returned to BW
      - i_end   = first index after the concentric push peak where v
                  returns to BW (i.e. the end of the ascent)

    Using find_peaks with a minimum inter-valley distance guarantees
    one detection per squat and avoids the micro-split + over-merge
    pitfalls of threshold-crossing approaches on slow, low-amplitude
    controlled squats.

    Args:
        min_rep_s: minimum interval between consecutive squat bottoms
        min_prominence_n: minimum GRF dip (N) to count as a valley
        band_n: how close to BW (+/- band_n) the signal must return to
                mark the rep boundary (start and end)
    """
    from scipy.signal import find_peaks

    v = butter_lowpass(total_n, 2.0, fs)   # heavier LPF helps slow squats
    n = len(v)

    # Find squat bottoms (local minima)
    min_distance = max(int(min_rep_s * fs), 5)
    valleys, _ = find_peaks(-v, distance=min_distance,
                            prominence=min_prominence_n)
    if len(valleys) == 0:
        return []

    reps: list[tuple[int, int, int]] = []
    for k, i_bot in enumerate(valleys):
        # Lower bound for this rep's start = halfway back to previous valley
        # (to avoid reaching into the prior rep)
        prev_bot = valleys[k - 1] if k > 0 else 0
        back_bound = max(0, (prev_bot + i_bot) // 2)
        # Walk back from i_bot until GRF returns to near BW (end of prior standing)
        i_start = back_bound
        for j in range(int(i_bot) - 1, int(back_bound) - 1, -1):
            if v[j] >= bw - band_n:
                i_start = j
                break

        # Upper bound for this rep's end = halfway to next valley
        next_bot = valleys[k + 1] if k + 1 < len(valleys) else n
        fwd_bound = min(n - 1, (i_bot + next_bot) // 2)
        # Find concentric push peak between bottom and fwd_bound
        seg = v[int(i_bot):int(fwd_bound) + 1]
        if len(seg) < 2:
            continue
        push_peak_rel = int(np.argmax(seg))
        push_peak = int(i_bot) + push_peak_rel
        # From push peak, walk forward until GRF returns to near BW
        i_end = int(push_peak)
        for j in range(int(push_peak), int(fwd_bound) + 1):
            if v[j] <= bw + band_n:
                i_end = j
                break
        else:
            i_end = int(fwd_bound)

        # Final filter: a real squat rep must have a concentric push peak
        # that clearly exceeds BW. Without a push, this "valley" is just
        # standing-state noise and should be discarded.
        rep_peak = float(v[i_bot:i_end + 1].max())
        if rep_peak < bw + min_push_above_bw_n:
            continue

        reps.append((int(i_start), int(i_bot), int(i_end)))
    return reps


def _estimate_bw_n(force: ForceSession) -> float:
    """Median of smoothed total force during first 1s on plate."""
    fs = force.fs
    n = min(int(1.0 * fs), len(force.total))
    ft = butter_lowpass(force.total[:n], 5.0, fs)
    ft = ft[ft > 100]
    return float(np.median(ft)) if len(ft) > 0 else 700.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase S1d — Precision metrics (patent 2 §4)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_rep_cop(cop_x: np.ndarray, cop_y: np.ndarray,
                       i_s: int, i_e: int,
                       n_samples: int = 101) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Resample one rep's CoP trail to a fixed length along normalised
    time (0%-100% of the rep). Foot-center reference is approximated as
    the rep's mean CoP (proper foot-center would need pose or physical
    marker; subtracting per-rep mean is the standard fallback).

    Returns ``(x_norm, y_norm)`` shape (n_samples,), or None if the rep
    has fewer than 3 valid CoP samples.
    """
    x_seg = cop_x[i_s:i_e + 1]
    y_seg = cop_y[i_s:i_e + 1]
    mask = (~np.isnan(x_seg)) & (~np.isnan(y_seg))
    if mask.sum() < 3:
        return None
    x_valid = x_seg[mask]
    y_valid = y_seg[mask]
    # Center on rep's mean CoP (relative coord, per patent §4 Step 1)
    x_valid = x_valid - x_valid.mean()
    y_valid = y_valid - y_valid.mean()
    n_valid = len(x_valid)
    if n_valid < 2:
        return None
    src = np.linspace(0.0, 1.0, n_valid)
    dst = np.linspace(0.0, 1.0, n_samples)
    return np.interp(dst, src, x_valid), np.interp(dst, src, y_valid)


def _compute_cmc(trajs: list[np.ndarray]) -> Optional[float]:
    """Coefficient of Multiple Correlation (Kadaba 1989).

    Each entry of ``trajs`` is a (N,) array representing one rep's
    trajectory of the same axis (AP or ML), normalised to N samples
    across time. CMC ≈ 1 means very consistent across reps; CMC ≈ 0
    means random variation dominates. Requires ≥ 2 reps.
    """
    if len(trajs) < 2:
        return None
    Y = np.asarray(trajs, dtype=np.float64)      # (G reps, N samples)
    # Mean trajectory (one value per sample index, averaged across reps)
    Y_mean_sample = Y.mean(axis=0)
    # Grand mean (scalar)
    Y_grand_mean = Y.mean()
    num = float(((Y - Y_mean_sample[None, :]) ** 2).sum())
    den = float(((Y - Y_grand_mean) ** 2).sum())
    if den <= 1e-12:
        return None
    cmc_sq = 1.0 - num / den
    # Numerical floor
    return float(np.sqrt(max(0.0, min(1.0, cmc_sq))))


def _compute_rmse_per_rep(trajs: list[np.ndarray]) -> list[Optional[float]]:
    """Per-rep RMS deviation from the mean trajectory of the same session."""
    if len(trajs) < 2:
        return [None] * len(trajs)
    Y = np.asarray(trajs, dtype=np.float64)
    Y_mean = Y.mean(axis=0)
    return [float(np.sqrt(np.mean((y - Y_mean) ** 2))) for y in Y]


def _compute_rfd_intervals(total_n: np.ndarray, fs: float,
                           i_bot: int, i_end: int, bw_n: float,
                           intervals_ms: tuple[int, ...] = (20, 40, 60, 80, 100),
                           onset_threshold_n: float = 50.0,
                           ) -> tuple[dict[int, Optional[float]],
                                       Optional[float], Optional[int]]:
    """Interval-based RFD from concentric force onset.

    Onset = first sample in the concentric phase (``[i_bot, i_end]``)
    where vGRF exceeds ``bw + onset_threshold_n`` (default 50 N above
    body weight). Returns:

      - ``rfd_map``  : {20: N/s, 40: ..., ...} (None where window
                       extends beyond rep end)
      - ``peak_rfd`` : max N/s over any 10-ms sliding window in the
                       concentric ascent, or None if onset not found
      - ``onset_idx``: absolute index in ``total_n`` of the force onset,
                       None if not detected (used by VRT)
    """
    if i_end <= i_bot + 1:
        return {ms: None for ms in intervals_ms}, None, None
    seg = total_n[i_bot:i_end + 1]
    thresh = bw_n + onset_threshold_n
    above = np.where(seg > thresh)[0]
    if len(above) == 0:
        return {ms: None for ms in intervals_ms}, None, None
    onset_abs = int(i_bot + above[0])
    f_onset = float(total_n[onset_abs])

    rfd_map: dict[int, Optional[float]] = {}
    for ms in intervals_ms:
        dt = ms / 1000.0
        n_samp = max(1, int(round(dt * fs)))
        end_idx = onset_abs + n_samp
        if end_idx >= len(total_n):
            rfd_map[ms] = None
            continue
        f_end = float(total_n[end_idx])
        rfd_map[ms] = (f_end - f_onset) / dt

    # Peak RFD over 10 ms sliding window in the full concentric phase
    window = max(1, int(round(0.010 * fs)))
    if i_end - onset_abs <= window:
        peak = None
    else:
        # Vectorised sliding-window slope: diff by `window` then divide by dt
        tail = total_n[onset_abs:i_end + 1]
        if len(tail) <= window:
            peak = None
        else:
            dt = window / fs
            slopes = (tail[window:] - tail[:-window]) / dt
            peak = float(slopes.max())
    return rfd_map, peak, onset_abs


def _compute_impulse_asymmetry(b1_total: np.ndarray, b2_total: np.ndarray,
                                fs: float, i_s: int, i_b: int, i_e: int,
                                ) -> tuple[float, float]:
    """Left/right impulse asymmetry (%) in the eccentric and concentric
    phases separately. 0 = perfectly symmetric, 100 = entire impulse on
    one board. Clipped to 0-100."""
    dt = 1.0 / max(fs, 1e-6)

    def _seg_asym(l0, l1, r0, r1):
        # Clip negative values from noise before integrating
        bl = np.maximum(b1_total[l0:l1 + 1], 0.0)
        br = np.maximum(b2_total[r0:r1 + 1], 0.0)
        imp_l = float(bl.sum() * dt)
        imp_r = float(br.sum() * dt)
        tot = imp_l + imp_r
        if tot < 1e-6:
            return 0.0
        return float(abs(imp_l - imp_r) / tot * 100.0)

    asym_ecc = _seg_asym(i_s, i_b, i_s, i_b)
    asym_con = _seg_asym(i_b, i_e, i_b, i_e)
    return asym_ecc, asym_con


def _load_squat_stim_log(session_dir) -> tuple[list[dict], Optional[float]]:
    """Load stimulus events for VRT computation.

    Returns ``(events, record_start_wall_s)``. Empty list + None when
    no stim log / meta is present (pre-S1d-2 sessions).

    The ``t_wall`` field is absolute (seconds since epoch); the
    analyzer's ForceSession.t_s is relative (0 at recording start).
    We convert via ``record_start_wall_s`` from session.json so the
    stim timestamps align with force time-series indices.
    """
    from pathlib import Path
    import csv, json
    session_dir = Path(session_dir)
    log = session_dir / "stimulus_log.csv"
    if not log.exists():
        return [], None
    meta_path = session_dir / "session.json"
    rec_start = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rec_start = meta.get("record_start_wall_s")
            if rec_start is not None:
                rec_start = float(rec_start)
        except Exception:
            rec_start = None
    events: list[dict] = []
    try:
        with open(log, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                events.append(row)
    except Exception:
        return [], rec_start
    return events, rec_start


def _match_vrt_for_rep(stim_events: list[dict], rec_start_wall_s: Optional[float],
                       t_bottom_s: float, onset_t_s: float,
                       max_lag_s: float = 2.0,
                       ) -> Optional[float]:
    """Match a concentric-ascent stim event near this rep's bottom and
    return VRT (ms) = onset_t - stim_t. Times are both converted to
    "seconds since recording start" so they compare cleanly.
    None if no matching event or rec_start not resolvable.
    """
    if not stim_events or rec_start_wall_s is None:
        return None
    best_dt: Optional[float] = None
    for ev in stim_events:
        try:
            typ = str(ev.get("trigger_type") or ev.get("response_type") or "")
            if "ascent" not in typ.lower() and "up" not in typ.lower():
                continue
            t_wall = ev.get("t_wall")
            if t_wall is None:
                continue
            # Convert absolute wall time → relative (seconds since record start)
            t_stim_rel = float(t_wall) - rec_start_wall_s
        except (TypeError, ValueError):
            continue
        # Stim must come just before ascent onset, within max_lag
        lag = onset_t_s - t_stim_rel
        if 0.0 <= lag <= max_lag_s:
            # Also near the rep bottom
            if abs(t_stim_rel - t_bottom_s) < max_lag_s:
                best_dt = lag if best_dt is None else min(best_dt, lag)
    if best_dt is None:
        return None
    return float(best_dt * 1000.0)   # seconds → ms


def analyze_squat(force: ForceSession,
                  use_encoder: int = 0,             # 1 or 2, or 0 to skip
                  bw_override_n: Optional[float] = None) -> SquatResult:
    """Squat analysis on force + (optionally) encoder + (optionally) 3D pose."""
    fs = force.fs
    t = force.t_s

    bw = bw_override_n if bw_override_n is not None else _estimate_bw_n(force)
    poses = load_poses3d_world(force.session_dir)
    have_pose = poses is not None

    # Load per-camera 2D pose data (if produced by process_pose_for_session.py)
    pose2d_by_cam = load_session_pose2d(force.session_dir)

    # Honor hardware availability: fall back to vGRF-based rep detection if
    # the caller asked for an encoder that is flagged broken in config.
    if use_encoder in (1, 2) and not _encoder_available(use_encoder):
        print(f"[squat] use_encoder={use_encoder} requested but "
              f"ENCODER{use_encoder}_AVAILABLE=False - falling back to vGRF")
        use_encoder = 0

    # Rep detection: prefer encoder, else use vGRF signal with extended
    # window covering BOTH eccentric dip AND concentric push spike.
    if use_encoder in (1, 2):
        disp = force.enc1 if use_encoder == 1 else force.enc2
        reps_idx = detect_reps(disp, fs, min_rom_mm=200.0)
    else:
        reps_idx = _detect_squat_reps_from_vgrf(force.total, fs, bw)

    # ── Precision prep (Phase S1d) ────────────────────────────────────
    # Normalize each rep's CoP to a common length so we can compute
    # cross-rep CMC + per-rep RMSE. AP = Y axis, ML = X axis.
    traj_cache: list[Optional[tuple[np.ndarray, np.ndarray]]] = []
    for (i_s, _i_b, i_e) in reps_idx:
        traj_cache.append(
            _normalize_rep_cop(force.cop_x, force.cop_y, i_s, i_e))
    trajs_ml = [tr[0] for tr in traj_cache if tr is not None]
    trajs_ap = [tr[1] for tr in traj_cache if tr is not None]
    cmc_ml = _compute_cmc(trajs_ml)
    cmc_ap = _compute_cmc(trajs_ap)
    rmse_ml_list = _compute_rmse_per_rep(trajs_ml)
    rmse_ap_list = _compute_rmse_per_rep(trajs_ap)
    # Map back to per-rep (some reps may have been skipped for NaN)
    rmse_lookup: dict[int, tuple[Optional[float], Optional[float]]] = {}
    valid_k = 0
    for k, tr in enumerate(traj_cache):
        if tr is None:
            rmse_lookup[k] = (None, None)
        else:
            rmse_lookup[k] = (rmse_ap_list[valid_k] if valid_k < len(rmse_ap_list) else None,
                              rmse_ml_list[valid_k] if valid_k < len(rmse_ml_list) else None)
            valid_k += 1

    stim_events, stim_rec_start = _load_squat_stim_log(force.session_dir)

    # Per-rep metrics
    rep_list: list[SquatRep] = []
    for k, (i_s, i_b, i_e) in enumerate(reps_idx):
        vgrf_seg = force.total[i_s:i_e + 1]
        if len(vgrf_seg) < 3:
            continue
        peak_vgrf = float(vgrf_seg.max())

        # WBA during the rep
        fl = force.b1_total[i_s:i_e + 1]
        fr = force.b2_total[i_s:i_e + 1]
        tot = fl + fr
        wba_series = 100.0 * np.abs(fl - fr) / np.where(tot > 0, tot, 1.0)
        mean_wba = float(wba_series.mean())

        dep_enc = None
        if use_encoder in (1, 2):
            disp = force.enc1 if use_encoder == 1 else force.enc2
            dep_enc = float(disp[i_s:i_e + 1].max() - disp[i_s:i_e + 1].min())

        dep_hip = None
        knee_toe = None
        hip_cop = None
        sym_knee = None
        if have_pose and poses["kpts3d"] is not None:
            k3 = poses["kpts3d"]
            # Align pose indices to force indices via simple proportional mapping
            # (both streams are dense, roughly synchronized by recording pipeline)
            n_pose = len(k3)
            n_force = len(force.t_s)
            p_s = int(i_s * n_pose / n_force)
            p_e = int(i_e * n_pose / n_force)
            if p_e > p_s + 2:
                kseg = k3[p_s:p_e + 1]
                hip_mid = 0.5 * (kseg[:, L_HIP] + kseg[:, R_HIP])
                # Depth from hip z range (less reliable due to anisotropy)
                dep_hip = float(np.nanmax(hip_mid[:, 2]) - np.nanmin(hip_mid[:, 2]))

                # Knee over ankle (horizontal)
                knee_mid  = 0.5 * (kseg[:, L_KNEE]  + kseg[:, R_KNEE])
                ankle_mid = 0.5 * (kseg[:, L_ANKLE] + kseg[:, R_ANKLE])
                # Horizontal distance knee to ankle mean
                dxy = knee_mid[:, :2] - ankle_mid[:, :2]
                knee_toe = float(np.nanmean(np.linalg.norm(dxy, axis=1)))

                # Hip over CoP (horizontal)
                cop_seg = np.stack([force.cop_x[i_s:i_e + 1],
                                    force.cop_y[i_s:i_e + 1]], axis=1)
                # Resample pose segment to match force length
                if len(hip_mid) > 1:
                    idx_r = np.linspace(0, len(hip_mid) - 1,
                                         len(cop_seg)).astype(int)
                    hip_xy = hip_mid[idx_r, :2]
                    hip_cop = float(np.nanmean(np.linalg.norm(hip_xy - cop_seg,
                                                               axis=1)))
                # Symmetry of knee forward extension
                l_forward = float(np.nanmean(
                    kseg[:, L_KNEE, 1] - kseg[:, L_ANKLE, 1]))
                r_forward = float(np.nanmean(
                    kseg[:, R_KNEE, 1] - kseg[:, R_ANKLE, 1]))
                if max(abs(l_forward), abs(r_forward)) > 1:
                    sym_knee = 100.0 * min(abs(l_forward), abs(r_forward)) / \
                        max(abs(l_forward), abs(r_forward))

        ecc_t = float((t[i_b] - t[i_s]))
        con_t = float((t[i_e] - t[i_b]))

        # ── 2D pose summary per camera over this rep window ──────────────
        pose_per_cam: dict = {}
        pose_mean: dict = {}
        if pose2d_by_cam:
            t_start_rel = float(t[i_s])
            t_end_rel   = float(t[i_e])
            for cid, ps in pose2d_by_cam.items():
                f0 = resolve_pose_frame(t_start_rel, force.session_dir, cid, ps.fps)
                f1 = resolve_pose_frame(t_end_rel,   force.session_dir, cid, ps.fps) + 1
                pose_per_cam[cid] = window_summary(ps, f0, f1)
            pose_mean = aggregate_cams(pose_per_cam)

        # ── Phase S1d precision metrics for this rep ─────────────────
        rmse_ap_v, rmse_ml_v = rmse_lookup.get(k, (None, None))
        tempo_ratio = (ecc_t / con_t) if con_t > 1e-3 else None
        asym_ecc, asym_con = _compute_impulse_asymmetry(
            force.b1_total, force.b2_total, fs, i_s, i_b, i_e)
        rfd_map, peak_rfd, onset_idx = _compute_rfd_intervals(
            force.total, fs, i_b, i_e, bw)
        vrt_ms = None
        if onset_idx is not None:
            onset_t = float(t[onset_idx])
            vrt_ms = _match_vrt_for_rep(
                stim_events, stim_rec_start,
                t_bottom_s=float(t[i_b]), onset_t_s=onset_t)

        rep_list.append(SquatRep(
            idx=k,
            t_start_s=float(t[i_s]),
            t_bottom_s=float(t[i_b]),
            t_end_s=float(t[i_e]),
            ecc_duration_s=ecc_t,
            con_duration_s=con_t,
            peak_vgrf_n=peak_vgrf,
            peak_vgrf_bw=peak_vgrf / bw,
            mean_wba_pct=mean_wba,
            depth_from_encoder_mm=dep_enc,
            depth_from_hip_mm=dep_hip,
            knee_over_toe_mm_mean=knee_toe,
            hip_over_cop_mm_mean=hip_cop,
            symmetry_knee_pct=sym_knee,
            rmse_ap_mm=rmse_ap_v,
            rmse_ml_mm=rmse_ml_v,
            tempo_ratio=tempo_ratio,
            impulse_asym_ecc_pct=asym_ecc,
            impulse_asym_con_pct=asym_con,
            rfd_n_s={str(k): v for k, v in (rfd_map or {}).items()},
            peak_rfd_n_s=peak_rfd,
            vrt_ms=vrt_ms,
            pose_per_cam=pose_per_cam,
            pose_mean=pose_mean,
        ))

    result = SquatResult(n_reps=len(rep_list), bw_n=bw, reps=rep_list,
                          cmc_ap=cmc_ap, cmc_ml=cmc_ml)
    if rep_list:
        depths = [r.depth_from_encoder_mm or r.depth_from_hip_mm or 0.0
                  for r in rep_list]
        result.mean_depth_mm = float(np.mean(depths))
        result.mean_peak_vgrf_bw = float(np.mean(
            [r.peak_vgrf_bw for r in rep_list]))
        result.mean_wba_pct = float(np.mean([r.mean_wba_pct for r in rep_list]))

        # ── Phase S1d session-level means ────────────────────────────
        def _mean_not_none(xs):
            vals = [x for x in xs if x is not None]
            return float(np.mean(vals)) if vals else 0.0

        result.mean_rmse_ap_mm = _mean_not_none(
            [r.rmse_ap_mm for r in rep_list])
        result.mean_rmse_ml_mm = _mean_not_none(
            [r.rmse_ml_mm for r in rep_list])
        result.mean_tempo_ratio = _mean_not_none(
            [r.tempo_ratio for r in rep_list])
        result.mean_impulse_asym_ecc_pct = _mean_not_none(
            [r.impulse_asym_ecc_pct for r in rep_list])
        result.mean_impulse_asym_con_pct = _mean_not_none(
            [r.impulse_asym_con_pct for r in rep_list])
        result.mean_peak_rfd_n_s = _mean_not_none(
            [r.peak_rfd_n_s for r in rep_list])
        vrts = [r.vrt_ms for r in rep_list if r.vrt_ms is not None]
        result.mean_vrt_ms = float(np.mean(vrts)) if vrts else None
    return result


def analyze_squat_file(session_dir, **kw) -> SquatResult:
    return analyze_squat(load_force_session(session_dir), **kw)
