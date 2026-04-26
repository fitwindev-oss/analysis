"""
Fix CoP unit bug in an already-recorded forces.csv.

The original record_calibration_session.py multiplied cop values by 1000
before writing, but DaqFrame.cop_world_mm() already returns mm. This script
divides the cop_world_x_mm and cop_world_y_mm columns by 1000 in-place (it
writes a backup copy of the original first).

Usage:
    python scripts/fix_forces_csv_units.py --session session_YYYYMMDD_HHMMSS

Detection:
    If the cop std in the file is > 10_000 (absurd for a 558x432 mm plate),
    the script treats it as buggy and divides by 1000.  Otherwise it declines
    to touch the file.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--force", action="store_true",
                    help="patch even if the file does not look buggy")
    args = ap.parse_args()

    session_dir = config.CALIB_DIR / args.session
    if not session_dir.exists():
        alt = config.CALIB_DIR / f"session_{args.session}"
        if alt.exists():
            session_dir = alt
        else:
            raise FileNotFoundError(session_dir)

    fp = session_dir / "forces.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)

    df = pd.read_csv(fp)
    if "cop_world_x_mm" not in df.columns:
        raise RuntimeError(f"{fp} has no CoP columns")

    cx = pd.to_numeric(df["cop_world_x_mm"], errors="coerce")
    cy = pd.to_numeric(df["cop_world_y_mm"], errors="coerce")
    cx_std = float(cx.std(skipna=True))
    cy_std = float(cy.std(skipna=True))

    print(f"[fix] {fp}")
    print(f"  current  cop std: x={cx_std:.2f}  y={cy_std:.2f}")
    looks_buggy = cx_std > 10_000 or cy_std > 10_000
    if not looks_buggy and not args.force:
        print("  CoP magnitudes look already correct; nothing to do.")
        print("  (use --force to divide anyway)")
        return

    backup = fp.with_suffix(".csv.bak")
    shutil.copy(fp, backup)
    print(f"  backed up original to: {backup}")

    df["cop_world_x_mm"] = cx / 1000.0
    df["cop_world_y_mm"] = cy / 1000.0
    df.to_csv(fp, index=False)

    new_cx_std = float(df["cop_world_x_mm"].std(skipna=True))
    new_cy_std = float(df["cop_world_y_mm"].std(skipna=True))
    print(f"  corrected cop std: x={new_cx_std:.2f}  y={new_cy_std:.2f}")
    print(f"  saved: {fp}")

    if new_cx_std > 500 or new_cy_std > 500:
        print("  [warn] values still look large; CoP std of >500 mm is unusual "
              "for a 558x432 mm plate.")


if __name__ == "__main__":
    main()
