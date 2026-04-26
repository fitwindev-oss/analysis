"""
Diagnostic report for a skeleton-based calibration.

Usage:
    python scripts/diagnose_calibration.py --session session_YYYYMMDD_HHMMSS

Reports:
  - Per-camera pose (position in unscaled world / cam0 frame)
  - 3D ankle midpoint statistics over the session (std per axis, range)
  - Correlation between ankle 3D and CoP over time
  - Estimated scale from 3 different methods:
      (a) Correlation-based (Procrustes)
      (b) Range ratio  (bbox_cop / bbox_ankle_principal)
      (c) Ankle-distance (stance width heuristic)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    args = ap.parse_args()

    session_dir = config.CALIB_DIR / args.session
    if not session_dir.exists():
        alt = config.CALIB_DIR / f"session_{args.session}"
        if alt.exists():
            session_dir = alt
        else:
            raise FileNotFoundError(session_dir)
    print(f"[diag] session: {session_dir}\n")

    # Extrinsics (unscaled)
    extr = np.load(config.CALIB_DIR / "extrinsics_unscaled.npz", allow_pickle=True)
    cam_ids = [str(x) for x in extr["cams"].tolist()]
    for i, cid in enumerate(cam_ids):
        R = extr[f"R{i}"]
        t = extr[f"t{i}"]
        K = extr[f"K{i}"]
        cam_pos = -R.T @ t        # camera center in world (unscaled)
        cam_forward = R.T @ np.array([0, 0, 1.0])
        print(f"{cid}:")
        print(f"  focal: {K[0,0]:.1f} px")
        print(f"  pos (unscaled): {cam_pos}")
        print(f"  forward dir  : {cam_forward}   (where cam is looking)")

    print()

    # 3D poses
    poses_path = config.CALIB_DIR / f"poses3d_{session_dir.name}.npz"
    p = np.load(poses_path, allow_pickle=True)
    kpts3d = p["kpts3d"]
    n_frames = kpts3d.shape[0]

    # Ankle analysis (JOINTS 15, 16 in COCO17 = L, R ankle)
    l_ankle = kpts3d[:, 15]
    r_ankle = kpts3d[:, 16]
    mid = 0.5 * (l_ankle + r_ankle)
    valid = ~np.isnan(mid).any(axis=-1)
    mid_valid = mid[valid]
    print(f"ankle midpoint: {valid.sum()}/{n_frames} valid frames")
    print(f"  mean  (unscaled): {mid_valid.mean(axis=0)}")
    print(f"  std   (unscaled): {mid_valid.std(axis=0)}")
    print(f"  range (unscaled): {mid_valid.ptp(axis=0)}")
    # PCA eigvals
    centered = mid_valid - mid_valid.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    print(f"  PCA eigvals (asc): {eigvals}")
    print(f"  PCA eigvecs cols:\n{eigvecs}")

    # Inter-ankle distance stats
    la_valid = ~np.isnan(l_ankle).any(axis=-1)
    ra_valid = ~np.isnan(r_ankle).any(axis=-1)
    both = la_valid & ra_valid
    if both.sum() > 10:
        d = np.linalg.norm(l_ankle[both] - r_ankle[both], axis=1)
        print(f"\n|L_ankle - R_ankle| unscaled:")
        print(f"  mean: {d.mean():.4f}  std: {d.std():.4f}")
        print(f"  range: [{d.min():.4f}, {d.max():.4f}]")

    # Shoulder-hip vertical span (for implied scale via body proportions)
    l_sh = kpts3d[:, 5]; r_sh = kpts3d[:, 6]
    l_hip = kpts3d[:, 11]; r_hip = kpts3d[:, 12]
    shoulder_mid = 0.5 * (l_sh + r_sh)
    hip_mid = 0.5 * (l_hip + r_hip)
    ok = ~(np.isnan(shoulder_mid).any(axis=-1) | np.isnan(hip_mid).any(axis=-1))
    if ok.sum() > 10:
        d = np.linalg.norm(shoulder_mid[ok] - hip_mid[ok], axis=1)
        print(f"\n|shoulder_mid - hip_mid| (torso length):")
        print(f"  mean: {d.mean():.4f}  std: {d.std():.4f}")

    # Ankle-to-nose (stature minus foot)
    nose = kpts3d[:, 0]
    ok = valid & ~np.isnan(nose).any(axis=-1)
    if ok.sum() > 10:
        d = np.linalg.norm(nose[ok] - mid[ok], axis=1)
        print(f"\n|nose - ankle_mid| (approx stature):")
        print(f"  mean: {d.mean():.4f}  std: {d.std():.4f}")

    # Force data
    fp = session_dir / "forces.csv"
    if fp.exists():
        f = pd.read_csv(fp)
        cx = pd.to_numeric(f["cop_world_x_mm"], errors="coerce").dropna()
        cy = pd.to_numeric(f["cop_world_y_mm"], errors="coerce").dropna()
        print(f"\nforces.csv: {len(f)} rows")
        print(f"  CoP X (mm):  mean={cx.mean():.1f}  std={cx.std():.1f}  "
              f"range=[{cx.min():.1f}, {cx.max():.1f}]")
        print(f"  CoP Y (mm):  mean={cy.mean():.1f}  std={cy.std():.1f}  "
              f"range=[{cy.min():.1f}, {cy.max():.1f}]")

        # Rough implied scale
        print("\nImplied scale estimates:")
        print(f"  (using largest PCA eigval of ankle midpoint)")
        sigma_src = np.sqrt(eigvals[-1])
        sigma_cop_max = max(cx.std(), cy.std())
        scale_corr = sigma_cop_max / sigma_src
        print(f"  scale ~ {scale_corr:.1f}  mm/unit (from largest std match)")

        d_stature_estimated_human = 1600  # mm, average ankle-to-nose for adults
        if ok.sum() > 10:
            d = np.linalg.norm(nose[ok] - mid[ok], axis=1)
            scale_stature = d_stature_estimated_human / d.mean()
            print(f"  scale ~ {scale_stature:.1f}  mm/unit (from ~1.6m ankle-to-nose)")


if __name__ == "__main__":
    main()
