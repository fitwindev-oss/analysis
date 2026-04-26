"""
Recompute cop_world_{x,y}_mm and total_n in an already-recorded forces.csv
using the current (clipped-corner) CoP formula.

Use this after fixing the CoP calculation in DaqFrame, to avoid having to
re-record valid sessions. A .bak backup of the original CSV is written.

Usage:
    python scripts/recompute_cop.py --session balance_20260421_225447
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def _board_cop(corners_n: np.ndarray, w: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized board-local CoP with negative-corner clipping."""
    clipped = np.maximum(corners_n, 0.0)                  # (N, 4)
    total = clipped.sum(axis=1)                           # (N,)
    tl = clipped[:, 0]; tr = clipped[:, 1]
    bl = clipped[:, 2]; br = clipped[:, 3]
    cx = np.where(total > 10.0, (tr + br) * w / np.maximum(total, 1e-6), w / 2)
    cy = np.where(total > 10.0, (tl + tr) * h / np.maximum(total, 1e-6), h / 2)
    cx = np.clip(cx, 0, w)
    cy = np.clip(cy, 0, h)
    return cx, cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    args = ap.parse_args()

    base = None
    for root in (config.SESSIONS_DIR, config.CALIB_DIR):
        c = root / args.session
        if c.exists():
            base = c; break
    if base is None:
        raise FileNotFoundError(args.session)
    fp = base / "forces.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)

    df = pd.read_csv(fp)
    print(f"[patch] {fp}  ({len(df)} samples)")

    b1 = df[["b1_tl_N", "b1_tr_N", "b1_bl_N", "b1_br_N"]].to_numpy(dtype=np.float64)
    b2 = df[["b2_tl_N", "b2_tr_N", "b2_bl_N", "b2_br_N"]].to_numpy(dtype=np.float64)

    # Board-local CoPs
    W, H = config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM
    b1_cx, b1_cy = _board_cop(b1, W, H)
    b2_cx, b2_cy = _board_cop(b2, W, H)

    # To world (Board2 offset in X by plate1 width)
    b1_wx = b1_cx + config.BOARD1_ORIGIN_MM[0]
    b1_wy = b1_cy + config.BOARD1_ORIGIN_MM[1]
    b2_wx = b2_cx + config.BOARD2_ORIGIN_MM[0]
    b2_wy = b2_cy + config.BOARD2_ORIGIN_MM[1]

    # Per-board total (using RAW = unclipped signed forces, matches stored total)
    b1_total_raw = b1.sum(axis=1)
    b2_total_raw = b2.sum(axis=1)
    # For CoP weighting, use clipped totals
    b1_tc = np.maximum(b1, 0.0).sum(axis=1)
    b2_tc = np.maximum(b2, 0.0).sum(axis=1)

    w_total = b1_tc + b2_tc
    valid = w_total > 10.0
    cx = np.full(len(df), np.nan)
    cy = np.full(len(df), np.nan)
    cx[valid] = (b1_wx[valid] * b1_tc[valid] + b2_wx[valid] * b2_tc[valid]) / w_total[valid]
    cy[valid] = (b1_wy[valid] * b1_tc[valid] + b2_wy[valid] * b2_tc[valid]) / w_total[valid]

    # Preview outlier reduction
    old_cx = pd.to_numeric(df["cop_world_x_mm"], errors="coerce").to_numpy()
    old_cy = pd.to_numeric(df["cop_world_y_mm"], errors="coerce").to_numpy()
    print(f"  OLD cop_x  p5={np.nanpercentile(old_cx,5):.1f}  "
          f"p95={np.nanpercentile(old_cx,95):.1f}  "
          f"min={np.nanmin(old_cx):.1f}  max={np.nanmax(old_cx):.1f}")
    print(f"  NEW cop_x  p5={np.nanpercentile(cx,5):.1f}  "
          f"p95={np.nanpercentile(cx,95):.1f}  "
          f"min={np.nanmin(cx):.1f}  max={np.nanmax(cx):.1f}")
    print(f"  OLD cop_y  p5={np.nanpercentile(old_cy,5):.1f}  "
          f"p95={np.nanpercentile(old_cy,95):.1f}  "
          f"min={np.nanmin(old_cy):.1f}  max={np.nanmax(old_cy):.1f}")
    print(f"  NEW cop_y  p5={np.nanpercentile(cy,5):.1f}  "
          f"p95={np.nanpercentile(cy,95):.1f}  "
          f"min={np.nanmin(cy):.1f}  max={np.nanmax(cy):.1f}")

    # Backup + write
    backup = fp.with_suffix(".csv.bak")
    if not backup.exists():
        shutil.copy(fp, backup)
        print(f"  backed up: {backup}")
    df["cop_world_x_mm"] = [f"{v:.2f}" if not np.isnan(v) else "" for v in cx]
    df["cop_world_y_mm"] = [f"{v:.2f}" if not np.isnan(v) else "" for v in cy]
    df.to_csv(fp, index=False)
    print(f"  saved: {fp}")


if __name__ == "__main__":
    main()
