"""
World frame alignment using force plate CoP.

Reads the UNSCALED extrinsics + triangulated 3D poses + force plate CoP from
a calibration session, finds the similarity transform (scale s, rotation R,
translation t) that aligns the ankle-midpoint 3D trajectory to the CoP
trajectory during the STAND STILL phase, then writes the final
world_frame.npz.

Assumptions:
  - Session was recorded with record_calibration_session.py (includes the
    standard protocol timeline - STAND STILL is the last 10 seconds).
  - DAQ forces.csv is present and has CoP in plate-world mm.
  - Both ankles are well-triangulated during STAND STILL (L/R ankle joints
    15, 16 in COCO-17).

Sanity checks after alignment:
  - Mean ankle Z near zero
  - Head (nose) Z positive (person is upright, not upside-down)
  - Standing subject height roughly in [1.3, 2.1] m

Usage:
    python scripts/align_world_from_force.py --session session_20260421_150000

Output:
    data/calibration/world_frame.npz

Overwrites any previous file.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


STAND_STILL_START_S = 50.0    # per DEFAULT_PROTOCOL in record script
STAND_STILL_END_S   = 60.0

# Dynamic portion used for Umeyama fit (march + turn + arms + squat).
# During these phases both CoP and ankle-midpoint move significantly so
# the similarity transform is well-conditioned. Pure STAND STILL is NOT
# used for fitting - it has near-zero variance and gives unstable scale.
DYNAMIC_START_S = 14.0
DYNAMIC_END_S   = 50.0
MIN_FORCE_N     = 100.0   # subject must be on the plate


def umeyama_similarity(src: np.ndarray, dst: np.ndarray,
                       with_scaling: bool = True):
    """
    Generic Umeyama (supports any dim). y ≈ s * R * x + t.
    """
    assert src.shape == dst.shape, f"{src.shape} vs {dst.shape}"
    n, d = src.shape
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    if with_scaling:
        var_s = (src_c ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_with_stature(kpts3d_full: np.ndarray,
                       src_valid_frames: np.ndarray,
                       src_pairs: np.ndarray,      # (N, 3) ankle-mid, 3D
                       dst_pairs: np.ndarray,      # (N, 3) CoP XY with z=0
                       stature_mm: float,
                       still_mask_on_pairs: np.ndarray):
    """
    Stature-anchored alignment for anisotropic 3D reconstructions.

    Workflow:
      1. Scale  s  from median |nose - ankle_midpoint| during STAND STILL
         against the known subject stature (ankle-to-nose ~= 0.895 * stature).
      2. World-Z direction = mean (nose - ankle_mid) during STAND STILL.
      3. Horizontal (world-XY) rotation from 2D rotation-only Procrustes
         between ankle_mid horizontal motion and CoP horizontal motion.
      4. Translation: ankle mean XY ↔ CoP mean XY; ankle mean Z = 0.

    All scalars in mm, all 3D in original unscaled src frame.
    """
    ankle_to_nose_mm = 0.895 * stature_mm   # empirical for adults

    # Nose and ankle mid for STATURE and UP-DIR (from STAND STILL frames only)
    n_kp = kpts3d_full.shape[0]
    l_ankle = kpts3d_full[:, 15]
    r_ankle = kpts3d_full[:, 16]
    nose    = kpts3d_full[:,  0]
    ankle_mid_full = 0.5 * (l_ankle + r_ankle)

    # STAND STILL: both ankles and nose non-NaN + mask
    valid_still = (~np.isnan(ankle_mid_full).any(-1)) & (~np.isnan(nose).any(-1))
    # still_mask_on_pairs gives which of our src_pairs rows are during STAND STILL;
    # we need a full-frame equivalent. Build one from scratch.
    # src_valid_frames[i] is the frame index that produced src_pairs[i].
    # So still_frames = frame indices of src_pairs rows where still_mask_on_pairs is True.
    still_frame_idxs = src_valid_frames[still_mask_on_pairs]
    still_mask_full = np.zeros(n_kp, dtype=bool)
    still_mask_full[still_frame_idxs] = True
    still_mask_full &= valid_still

    if still_mask_full.sum() < 20:
        raise RuntimeError(
            f"need >=20 STAND STILL frames with both ankles + nose visible, "
            f"got {still_mask_full.sum()}")

    stature_vecs = nose[still_mask_full] - ankle_mid_full[still_mask_full]
    stature_lens = np.linalg.norm(stature_vecs, axis=1)
    # robust: median, and also reject outlier frames
    med = np.median(stature_lens)
    mad = np.median(np.abs(stature_lens - med))
    keep = np.abs(stature_lens - med) < 5 * (mad + 1e-6)
    stature_vecs_keep = stature_vecs[keep]
    stature_unscaled = np.median(np.linalg.norm(stature_vecs_keep, axis=1))
    s = ankle_to_nose_mm / stature_unscaled
    print(f"  stature_unscaled (median) = {stature_unscaled:.4f}", flush=True)
    print(f"  scale from stature        = {s:.2f} mm/unit", flush=True)

    # Up direction = mean of unit stature vectors (in src frame)
    unit_stat = stature_vecs_keep / np.linalg.norm(
        stature_vecs_keep, axis=1, keepdims=True)
    up_src = unit_stat.mean(axis=0)
    up_src /= np.linalg.norm(up_src)
    print(f"  world-Z direction in src  = {up_src}", flush=True)

    # ── Build rotation whose Z row = up_src, X/Y rows TBD ────────────────
    # Start with an arbitrary perpendicular basis, then refine X/Y via
    # horizontal CoP correlation.
    ref = np.array([1.0, 0, 0])
    if abs(np.dot(ref, up_src)) > 0.9:
        ref = np.array([0.0, 1, 0])
    x_axis_tmp = ref - np.dot(ref, up_src) * up_src
    x_axis_tmp /= np.linalg.norm(x_axis_tmp)
    y_axis_tmp = np.cross(up_src, x_axis_tmp)

    R_init = np.stack([x_axis_tmp, y_axis_tmp, up_src], axis=0)    # (3,3)

    # ── Filter outliers on the dynamic src_pairs/dst_pairs using MAD ─────
    # in src frame (robust to big triangulation glitches)
    dist_from_med = np.linalg.norm(src_pairs - np.median(src_pairs, axis=0), axis=1)
    mad2 = np.median(np.abs(dist_from_med - np.median(dist_from_med)))
    mad2 = max(mad2, 1e-6)
    inlier = dist_from_med < np.median(dist_from_med) + 6 * mad2
    print(f"  dynamic pairs inlier: {inlier.sum()}/{len(src_pairs)} "
          f"(dropped {len(src_pairs) - inlier.sum()})", flush=True)
    src_in = src_pairs[inlier]
    dst_in = dst_pairs[inlier]

    # ── Horizontal rotation via 2D Procrustes ───────────────────────────────
    # Apply s * R_init to src, then align XY with dst.
    proj_init = (s * (R_init @ src_in.T)).T        # (N, 3) in "world-candidate" frame
    ankle_xy = proj_init[:, :2]
    cop_xy = dst_in[:, :2]

    # 2D rotation-only Procrustes:  cop ≈ R_2d @ ankle  (no scale, just rotation)
    ankle_xy_c = ankle_xy - ankle_xy.mean(0)
    cop_xy_c   = cop_xy - cop_xy.mean(0)
    cov_2d = cop_xy_c.T @ ankle_xy_c             # (2, 2)
    U2, _, Vt2 = np.linalg.svd(cov_2d)
    R_2d = U2 @ Vt2
    if np.linalg.det(R_2d) < 0:
        Vt2 = Vt2.copy()
        Vt2[-1, :] *= -1
        R_2d = U2 @ Vt2

    R_2d_3d = np.eye(3)
    R_2d_3d[:2, :2] = R_2d
    R_total = R_2d_3d @ R_init

    # ── Translation ─────────────────────────────────────────────────────────
    predicted = (s * (R_total @ src_in.T)).T     # (N, 3)
    t_xy = cop_xy.mean(0) - predicted.mean(0)[:2]
    t_z  = 0.0 - predicted.mean(0)[2]
    t = np.array([t_xy[0], t_xy[1], t_z])

    return s, R_total, t


def umeyama_plate(src: np.ndarray, dst: np.ndarray):
    """
    Plate-aware similarity fit: src (N, 3) -> dst (N, 3) where dst's Z is
    always 0 (CoP on plate surface).

    Standard 3D Umeyama breaks down because reconstruction noise in src
    along directions that do NOT match dst inflates var(src) and collapses
    the scale. PCA-based pre-rotation also fails when noise is larger than
    real motion.

    The correct formulation is a **3D source → 2D target** Procrustes:
       find R in SO(3), s > 0, t in R^3  minimizing
         sum_i || (s R src_i + t)[:2] - cop_i ||^2

    Closed-form solution:
       cov = src_c.T @ cop_c / n    (3 x 2)
       SVD: cov = U_32 * Sigma_2 * V_22^T     (thin)
       First two ROWS of R (the x/y axes in world basis expressed in src
       frame) come from U_32 @ V_22^T (a 3x2 orthogonal matrix with
       orthonormal columns). Third row is the cross product.
       Scale  s = sum(Sigma) / var(src @ R[:2,:].T) .
       Translation: XY aligns centroids, Z sets mean predicted Z to 0
       (ankles on plate).

    This approach uses ONLY the correlation between src and dst in the
    scale denominator, so perpendicular src noise doesn't break scale.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 10:
        raise ValueError(f"need >=10 pairs, got {n}")

    src_mean = src.mean(axis=0)
    cop_mean = dst[:, :2].mean(axis=0)
    src_c = src - src_mean
    cop_c = dst[:, :2] - cop_mean

    # ── 3D → 2D Procrustes: cross-covariance SVD ────────────────────────
    M = (src_c.T @ cop_c) / n                     # (3, 2)
    U, D, Vt = np.linalg.svd(M, full_matrices=False)    # U (3,2), D (2,), Vt (2,2)
    # A = U @ Vt is 3x2 with orthonormal columns.
    A = U @ Vt

    # First 2 rows of R (world X, Y directions expressed in src frame) = A.T
    r1 = A[:, 0]
    r2 = A[:, 1]
    r3 = np.cross(r1, r2)     # world Z = right-hand-rule perpendicular
    R_total = np.stack([r1, r2, r3], axis=0)

    # Numerical safety: orthonormalize via polar decomposition
    U_p, _, Vt_p = np.linalg.svd(R_total)
    R_total = U_p @ Vt_p
    if np.linalg.det(R_total) < 0:
        R_total[2, :] *= -1

    # ── Scale: use only the 2D-matched variance ────────────────────────────
    proj = src_c @ R_total.T          # (N, 3); first 2 cols aligned with cop
    var_proj_xy = (proj[:, :2] ** 2).sum() / n
    s = float(np.sum(D) / var_proj_xy) if var_proj_xy > 1e-12 else 1.0

    # ── Translation ─────────────────────────────────────────────────────────
    predicted_full = (s * (R_total @ src.T)).T    # (N, 3)
    t_xy = cop_mean - predicted_full.mean(axis=0)[:2]
    t_z  = 0.0      - predicted_full.mean(axis=0)[2]
    t = np.array([t_xy[0], t_xy[1], t_z])
    return s, R_total, t


def transform_extrinsic(R_old: np.ndarray, t_old: np.ndarray,
                        s: float, R_align: np.ndarray, t_align: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Given old extrinsics (R_old, t_old) in unscaled frame and similarity
    transform world_new = s * R_align * world_old + t_align, return new
    extrinsics in the aligned world frame.

        R_new = R_old @ R_align.T
        t_new = -R_new @ t_align + s * t_old
    """
    R_new = R_old @ R_align.T
    t_new = -R_new @ t_align + s * t_old
    return R_new, t_new


def load_camera_timestamps(session_dir: Path, cam_ids: list[str]) -> dict:
    """Return dict cam_id -> (frame_idx ndarray, t_ns ndarray)."""
    out = {}
    for cid in cam_ids:
        ts_path = session_dir / f"{cid}.timestamps.csv"
        if not ts_path.exists():
            raise FileNotFoundError(f"missing: {ts_path}")
        df = pd.read_csv(ts_path)
        out[cid] = (df["frame_idx"].to_numpy(dtype=np.int64),
                    df["t_monotonic_ns"].to_numpy(dtype=np.int64))
    return out


def auto_find_still_segment(forces_df: pd.DataFrame,
                            min_duration_s: float = 3.0,
                            sway_threshold_mm: float = 30.0) -> tuple[int, int]:
    """
    Fallback: find a stretch of force data with low CoP sway.

    Returns (start_row, end_row) inclusive-exclusive slice.
    """
    cop_x = forces_df["cop_world_x_mm"].to_numpy(dtype=np.float64)
    cop_y = forces_df["cop_world_y_mm"].to_numpy(dtype=np.float64)
    t_wall = forces_df["t_wall"].to_numpy(dtype=np.float64)
    n = len(t_wall)
    if n < 10:
        return (0, n)
    fs = 1.0 / np.median(np.diff(t_wall))
    win = max(int(min_duration_s * fs), 10)
    best = (0, win, np.inf)
    for i in range(0, n - win, win // 4):
        xs = cop_x[i:i + win]
        ys = cop_y[i:i + win]
        if np.isnan(xs).any() or np.isnan(ys).any():
            continue
        sway = float(np.sqrt(xs.var() + ys.var()))
        if sway < best[2]:
            best = (i, i + win, sway)
    if best[2] > sway_threshold_mm:
        print(f"  [warn] best auto-found still sway = {best[2]:.1f} mm "
              f"(above threshold {sway_threshold_mm})", flush=True)
    return (best[0], best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--dyn-start", type=float, default=DYNAMIC_START_S,
                    help="start of DYNAMIC fit window (march+squat region)")
    ap.add_argument("--dyn-end",   type=float, default=DYNAMIC_END_S)
    ap.add_argument("--still-start", type=float, default=STAND_STILL_START_S,
                    help="start of STAND STILL sanity window")
    ap.add_argument("--still-end",   type=float, default=STAND_STILL_END_S)
    ap.add_argument("--auto",        action="store_true",
                    help="auto-detect high-motion segment for fit (override --dyn-*)")
    ap.add_argument("--stature-mm",  type=float, default=None,
                    help="subject stature in mm - if set, uses stature-anchored "
                         "alignment instead of Umeyama. Recommended when cameras "
                         "are not well-separated (anisotropic 3D reconstruction).")
    args = ap.parse_args()

    session_dir = config.CALIB_DIR / args.session
    if not session_dir.exists():
        alt = config.CALIB_DIR / f"session_{args.session}"
        if alt.exists():
            session_dir = alt
        else:
            raise FileNotFoundError(f"no such session: {session_dir}")

    print(f"\n[align] Session: {session_dir}", flush=True)

    # ── Load unscaled extrinsics ────────────────────────────────────────────
    extr_path = config.CALIB_DIR / "extrinsics_unscaled.npz"
    if not extr_path.exists():
        raise FileNotFoundError(
            f"{extr_path} missing. Run calibrate_from_poses.py first.")
    e = np.load(extr_path, allow_pickle=True)
    cam_ids = [str(x) for x in e["cams"].tolist()]
    extr = {
        cam_ids[0]: {"R": e["R0"], "t": e["t0"], "K": e["K0"]},
        cam_ids[1]: {"R": e["R1"], "t": e["t1"], "K": e["K1"]},
        cam_ids[2]: {"R": e["R2"], "t": e["t2"], "K": e["K2"]},
    }

    # ── Load triangulated 3D poses (time-aligned) ───────────────────────────
    poses_path = config.CALIB_DIR / f"poses3d_{session_dir.name}.npz"
    if not poses_path.exists():
        raise FileNotFoundError(
            f"{poses_path} missing. Run calibrate_from_poses.py first.")
    p = np.load(poses_path, allow_pickle=True)
    kpts3d = p["kpts3d"]              # (N_aligned, 17, 3)
    fps    = float(p["fps"])
    n_frames = kpts3d.shape[0]
    # Timestamp of each aligned row (in ns, same clock as forces.csv t_ns).
    if "t_ref_ns" in p:
        t_ref_ns_array = p["t_ref_ns"].astype(np.int64)
        print(f"[align] poses3d uses time-aligned axis "
              f"({len(t_ref_ns_array)} frames)", flush=True)
    else:
        t_ref_ns_array = None
        print(f"[align] [warn] poses3d has no t_ref_ns; falling back to "
              f"cam0 frame index alignment", flush=True)

    # ── Load force plate CoP ────────────────────────────────────────────────
    forces_path = session_dir / "forces.csv"
    if not forces_path.exists():
        raise FileNotFoundError(
            f"{forces_path} missing. Re-record with DAQ connected.")
    forces = pd.read_csv(forces_path)
    if "cop_world_x_mm" not in forces.columns:
        raise RuntimeError("forces.csv does not contain CoP columns")

    # ── Build per-aligned-frame timestamps (ns) ─────────────────────────────
    if t_ref_ns_array is not None:
        ref_ts_ns = t_ref_ns_array
    else:
        # legacy: use cam0 timestamps directly
        cam_ts = load_camera_timestamps(session_dir, cam_ids)
        ref_ts_ns = cam_ts[cam_ids[0]][1][:n_frames]

    # Session start_ns (from metadata if available, else first ref frame)
    meta_path = session_dir / "session.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rec_start_ns = meta.get("record_start_monotonic_ns")
    else:
        meta = {}
        rec_start_ns = None
    if rec_start_ns is None:
        rec_start_ns = int(ref_ts_ns[0])
    rec_start_ns = int(rec_start_ns)

    # Map aligned frame i -> seconds from rec_start
    cam0_sec = (ref_ts_ns - rec_start_ns) / 1e9
    cam0_ts  = ref_ts_ns

    # ── Build DYNAMIC-phase mask (march + turn + arms + squat) ──────────────
    if args.auto:
        print("[align] Auto-detecting high-motion segment from force data...",
              flush=True)
        i0, i1 = auto_find_still_segment(forces)   # (returns still - invert)
        # Take everything OUTSIDE the still segment as dynamic
        t_dyn_start_s = 5.0  # skip first few seconds of N-pose
        t_dyn_end_s   = (forces["t_ns"].iloc[i0] - rec_start_ns) / 1e9
        if t_dyn_end_s - t_dyn_start_s < 10:
            t_dyn_end_s = 50.0
        print(f"  auto dynamic: {t_dyn_start_s:.2f}-{t_dyn_end_s:.2f}s",
              flush=True)
    else:
        t_dyn_start_s = args.dyn_start
        t_dyn_end_s   = args.dyn_end

    cam_mask = (cam0_sec >= t_dyn_start_s) & (cam0_sec <= t_dyn_end_s)
    cam_idxs_dyn = np.where(cam_mask)[0]
    print(f"[align] dynamic fit window cam frames: {len(cam_idxs_dyn)} "
          f"(t={t_dyn_start_s:.1f}-{t_dyn_end_s:.1f}s)", flush=True)

    if len(cam_idxs_dyn) < 100:
        raise RuntimeError(
            f"too few dynamic frames: {len(cam_idxs_dyn)}. Re-record with "
            f"a longer march/squat segment.")

    # ── Match ankle midpoint ↔ CoP for every dynamic frame ───────────────────
    force_t = forces["t_ns"].to_numpy(dtype=np.int64)
    force_cx = forces["cop_world_x_mm"].to_numpy(dtype=np.float64)
    force_cy = forces["cop_world_y_mm"].to_numpy(dtype=np.float64)
    force_total = forces["total_n"].to_numpy(dtype=np.float64)

    src_pts, dst_pts, src_frames = [], [], []
    l_ankle_idx, r_ankle_idx = 15, 16
    # Gather dynamic + still together so we can mask later
    all_idxs = np.concatenate([cam_idxs_dyn,
                                np.where((cam0_sec >= args.still_start) &
                                         (cam0_sec <= args.still_end))[0]])
    all_idxs = np.unique(all_idxs)
    for fi in all_idxs:
        L = kpts3d[fi, l_ankle_idx]
        R = kpts3d[fi, r_ankle_idx]
        if np.isnan(L).any() or np.isnan(R).any():
            continue
        mid_3d = 0.5 * (L + R)
        t_ns = int(cam0_ts[fi])
        j = int(np.searchsorted(force_t, t_ns))
        if j >= len(force_t):
            j = len(force_t) - 1
        if force_total[j] < MIN_FORCE_N:
            continue
        cx = force_cx[j]; cy = force_cy[j]
        if np.isnan(cx) or np.isnan(cy):
            continue
        src_pts.append(mid_3d)
        dst_pts.append([cx, cy, 0.0])
        src_frames.append(int(fi))

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    src_frames = np.asarray(src_frames, dtype=np.int32)
    # Flags for which pairs are STAND STILL
    still_mask_pairs = (cam0_sec[src_frames] >= args.still_start) & \
                       (cam0_sec[src_frames] <= args.still_end)
    dyn_mask_pairs = ~still_mask_pairs
    print(f"[align] matched pairs: {len(src)} total  "
          f"(dynamic {dyn_mask_pairs.sum()}, still {still_mask_pairs.sum()})",
          flush=True)
    if len(src) < 50:
        raise RuntimeError(
            f"only {len(src)} valid pairs. Check 3D ankle detection + "
            f"force data coverage.")

    # Diagnostic
    src_std = src.std(axis=0)
    dst_std = dst.std(axis=0)
    print(f"  src std (unscaled): x={src_std[0]:.4f} y={src_std[1]:.4f} "
          f"z={src_std[2]:.4f}", flush=True)
    print(f"  dst std (mm):       x={dst_std[0]:.1f}   y={dst_std[1]:.1f}   "
          f"z={dst_std[2]:.1f}", flush=True)

    # ── Solve: pick algorithm based on --stature-mm ─────────────────────────
    if args.stature_mm is not None:
        print(f"\n[align] Using STATURE-ANCHORED alignment "
              f"(stature = {args.stature_mm:.0f} mm)...", flush=True)
        s, R_align, t_align = align_with_stature(
            kpts3d, src_frames, src, dst,
            stature_mm=args.stature_mm,
            still_mask_on_pairs=still_mask_pairs,
        )
    else:
        # Restrict to dynamic-only for Umeyama
        src_dyn = src[dyn_mask_pairs]
        dst_dyn = dst[dyn_mask_pairs]
        s, R_align, t_align = umeyama_plate(src_dyn, dst_dyn)
    print(f"[align] result:  s = {s:.4f}   "
          f"|t_align| = {np.linalg.norm(t_align):.1f}", flush=True)

    def _apply_transform(s_, R_, t_):
        out = np.full_like(kpts3d, np.nan)
        valid_m = ~np.isnan(kpts3d).any(axis=-1)
        flat_in_ = kpts3d[valid_m]
        flat_out_ = (s_ * (R_ @ flat_in_.T)).T + t_
        out[valid_m] = flat_out_.astype(np.float32)
        return out

    def _zstats(world_pts):
        az = np.concatenate([world_pts[:, l_ankle_idx, 2],
                             world_pts[:, r_ankle_idx, 2]])
        az = az[~np.isnan(az)]
        nz = world_pts[:, 0, 2]
        nz = nz[~np.isnan(nz)]
        return (float(np.mean(az)) if len(az) else float("nan"),
                float(np.mean(nz)) if len(nz) else float("nan"))

    kpts3d_world = _apply_transform(s, R_align, t_align)
    ankle_z_mean, nose_z_mean = _zstats(kpts3d_world)
    print(f"  ankle Z mean (should be ~0 mm): {ankle_z_mean:.1f}", flush=True)
    print(f"  nose   Z mean (should be >1000 mm): {nose_z_mean:.1f}",
          flush=True)

    # If subject is inverted (nose below ankle), flip Z.
    if nose_z_mean < ankle_z_mean:
        print("  [fix] subject appears inverted - flipping Z axis", flush=True)
        R_flip = np.diag([1.0, 1.0, -1.0])
        R_align = R_flip @ R_align
        t_align = R_flip @ t_align
        kpts3d_world = _apply_transform(s, R_align, t_align)
        ankle_z_mean, nose_z_mean = _zstats(kpts3d_world)
        print(f"  after flip: ankle Z = {ankle_z_mean:.1f}, "
              f"nose Z = {nose_z_mean:.1f}", flush=True)

    # ── Transform camera extrinsics ─────────────────────────────────────────
    new_extr = {}
    for cid in cam_ids:
        R_old = extr[cid]["R"]; t_old = extr[cid]["t"]
        R_new, t_new = transform_extrinsic(R_old, t_old, s, R_align, t_align)
        new_extr[cid] = {"R": R_new, "t": t_new, "K": extr[cid]["K"]}

    # ── Save final world frame ──────────────────────────────────────────────
    out_path = config.CALIB_DIR / "world_frame.npz"
    np.savez(
        out_path,
        cams=np.array(cam_ids),
        R0=new_extr[cam_ids[0]]["R"], t0=new_extr[cam_ids[0]]["t"],
        R1=new_extr[cam_ids[1]]["R"], t1=new_extr[cam_ids[1]]["t"],
        R2=new_extr[cam_ids[2]]["R"], t2=new_extr[cam_ids[2]]["t"],
        K0=new_extr[cam_ids[0]]["K"], K1=new_extr[cam_ids[1]]["K"],
        K2=new_extr[cam_ids[2]]["K"],
        dist0=np.zeros(5), dist1=np.zeros(5), dist2=np.zeros(5),
        scale=float(s),
        R_align=R_align, t_align=t_align,
        ankle_z_mean=ankle_z_mean,
        nose_z_mean=nose_z_mean,
        n_pairs=len(src),
        source="skeleton+force plate",
        session=session_dir.name,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"\n[align] saved: {out_path}", flush=True)

    # Also save session 3D trajectory in world mm for downstream analysis
    traj_path = config.CALIB_DIR / f"poses3d_world_{session_dir.name}.npz"
    np.savez(
        traj_path,
        kpts3d_world=kpts3d_world,
        fps=fps,
        joint_names=np.array(config.KPT_NAMES),
    )
    print(f"[align] saved trajectory: {traj_path}", flush=True)

    # Final sanity readout
    if abs(ankle_z_mean) > 100:
        print(f"\n  [warn] ankle Z mean = {ankle_z_mean:.1f} mm "
              f"(expected near 0). The alignment may be off. Consider "
              f"running with --auto or re-recording the session.",
              flush=True)
    else:
        print(f"\n  [OK] ankle Z mean within +/-100 mm of plate surface",
              flush=True)

    if nose_z_mean < 1300 or nose_z_mean > 2100:
        print(f"  [warn] nose Z mean = {nose_z_mean:.1f} mm (expected 1400-1900)",
              flush=True)
    else:
        print(f"  [OK] subject height plausible", flush=True)


if __name__ == "__main__":
    main()
