"""
Diagnostic: raw force/CoP distribution for a recorded session.

Usage:
    python scripts/diagnose_session.py --session <name>

Shows percentiles of each signal and flags outliers.
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

    base = None
    for root in (config.SESSIONS_DIR, config.CALIB_DIR):
        c = root / args.session
        if c.exists():
            base = c; break
    if base is None:
        raise FileNotFoundError(args.session)
    fp = base / "forces.csv"
    print(f"[diag] {fp}\n")

    df = pd.read_csv(fp)
    n = len(df)
    t = df["t_wall"].to_numpy(copy=True)
    t = t - t[0]
    fs = 1.0 / np.median(np.diff(t))
    print(f"samples: {n}   duration: {t[-1]:.2f} s   fs: {fs:.1f} Hz\n")

    def _stats(name: str):
        v = pd.to_numeric(df[name], errors="coerce").dropna().to_numpy()
        if len(v) == 0:
            print(f"  {name:20s}: (all NaN)")
            return
        p1 = np.percentile(v, 1)
        p5 = np.percentile(v, 5)
        p50 = np.percentile(v, 50)
        p95 = np.percentile(v, 95)
        p99 = np.percentile(v, 99)
        print(f"  {name:20s}: "
              f"min={v.min():10.2f}  "
              f"p1={p1:8.2f}  "
              f"p5={p5:8.2f}  "
              f"p50={p50:8.2f}  "
              f"p95={p95:8.2f}  "
              f"p99={p99:8.2f}  "
              f"max={v.max():10.2f}  "
              f"std={v.std():8.2f}")

    print("Force channels (N):")
    for c in ["b1_tl_N", "b1_tr_N", "b1_bl_N", "b1_br_N",
              "b2_tl_N", "b2_tr_N", "b2_bl_N", "b2_br_N"]:
        _stats(c)

    print("\nDerived:")
    _stats("total_n")
    _stats("cop_world_x_mm")
    _stats("cop_world_y_mm")
    _stats("enc1_mm")
    _stats("enc2_mm")

    # Outlier count for CoP
    cx = pd.to_numeric(df["cop_world_x_mm"], errors="coerce")
    cy = pd.to_numeric(df["cop_world_y_mm"], errors="coerce")
    tot = df["total_n"].to_numpy()

    # "plausible" CoP: inside plate +/- 50mm tolerance
    plausible = (cx > -50) & (cx < config.PLATE_TOTAL_WIDTH_MM + 50) & \
                (cy > -50) & (cy < config.PLATE_TOTAL_HEIGHT_MM + 50)
    n_impl = int((~plausible).sum())
    print(f"\nCoP samples outside plate+50mm margin: {n_impl}/{n} "
          f"({100*n_impl/n:.1f}%)")

    # Force on each board percentiles
    b1_total = df[["b1_tl_N", "b1_tr_N", "b1_bl_N", "b1_br_N"]].sum(axis=1)
    b2_total = df[["b2_tl_N", "b2_tr_N", "b2_bl_N", "b2_br_N"]].sum(axis=1)
    print(f"\nBoard totals:")
    print(f"  board1  mean={b1_total.mean():.1f}  min={b1_total.min():.1f}  "
          f"max={b1_total.max():.1f}")
    print(f"  board2  mean={b2_total.mean():.1f}  min={b2_total.min():.1f}  "
          f"max={b2_total.max():.1f}")
    low_f1 = int((b1_total < 50).sum())
    low_f2 = int((b2_total < 50).sum())
    print(f"  board1<50N: {low_f1} samples ({100*low_f1/n:.1f}%)")
    print(f"  board2<50N: {low_f2} samples ({100*low_f2/n:.1f}%)")


if __name__ == "__main__":
    main()
