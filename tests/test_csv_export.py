"""
Unit tests for src.analysis.csv_export.

Builds tiny synthetic sessions on disk (forces.csv, stimulus_log.csv,
events.csv, optional poses_*.npz) and runs export_session_csv on them.
Verifies:
  - timeseries CSV exists, starts with UTF-8 BOM, has the agreed
    column order (time → events → force → encoder → CoP → pose...)
  - event_stim values land on the row closest to each stim's t_wall
  - event_off_plate is 1 inside the events.csv intervals and 0 outside
  - NaN floats render as empty strings
  - pose_native CSV is generated only when poses_*.npz exists

Run:
    python tests/test_csv_export.py
"""
from __future__ import annotations

import csv as _csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.csv_export import (
    export_session_csv, _build_off_plate_col, _build_stim_event_cols,
    _fmt, _digits_for,
)


# ────────────────────────────────────────────────────────────────────────────
# Synthetic session builders
# ────────────────────────────────────────────────────────────────────────────
def _make_min_session(tmp: Path, *,
                       n_samples: int = 200,
                       fs: float = 100.0,
                       rec_start_wall: float = 1000.0,
                       stims: list[dict] | None = None,
                       off_plate_intervals: list[tuple[float, float]] | None = None,
                       with_pose: bool = False) -> Path:
    """Build a minimal session folder with everything csv_export can
    consume. Returns the session directory path.

    Force data: total_n alternates above/below the 20 N gate so the
    on_plate column has variation. Encoder + CoP fixed.
    """
    sd = tmp / "session"
    sd.mkdir()
    # forces.csv with all expected columns
    rows = ["t_ns,t_wall,b1_tl_N,b1_tr_N,b1_bl_N,b1_br_N,"
            "b2_tl_N,b2_tr_N,b2_bl_N,b2_br_N,enc1_mm,enc2_mm,"
            "total_n,cop_world_x_mm,cop_world_y_mm,on_plate"]
    for i in range(n_samples):
        t = rec_start_wall + i / fs
        # 8 corners — equal split, total_n ≈ 200N (well above 20N gate)
        c = 25.0
        total = c * 8
        cop_x = 280.0 + 0.05 * np.sin(i / 10.0)
        cop_y = 215.0 + 0.05 * np.cos(i / 10.0)
        rows.append(
            f"{i*int(1e7)},{t:.6f},{c:.3f},{c:.3f},{c:.3f},{c:.3f},"
            f"{c:.3f},{c:.3f},{c:.3f},{c:.3f},10.000,20.000,"
            f"{total:.3f},{cop_x:.2f},{cop_y:.2f},1")
    (sd / "forces.csv").write_text("\n".join(rows) + "\n",
                                     encoding="utf-8")

    # session.json (minimal)
    import json
    meta = {
        "test": "cognitive_reaction" if stims else "balance_eo",
        "record_start_wall_s": rec_start_wall,
        "duration_s": n_samples / fs,
        "subject_id": "test", "subject_name": "테스트",
        "subject_kg": 80.0, "n_daq_samples": n_samples,
        "n_stimuli": len(stims) if stims else 0,
    }
    (sd / "session.json").write_text(json.dumps(meta), encoding="utf-8")

    # stimulus_log.csv
    if stims:
        srows = ["trial_idx,t_wall,t_ns,stimulus_type,response_type,"
                 "target_x_norm,target_y_norm,target_label"]
        for i, s in enumerate(stims):
            srows.append(
                f"{i},{rec_start_wall + s['t_offs']:.6f},{i},"
                f"audio_visual,{s.get('label', 'pos_N')},"
                f"{s.get('tx', 0.5):.4f},{s.get('ty', 0.5):.4f},"
                f"{s.get('label', 'pos_N')}")
        (sd / "stimulus_log.csv").write_text("\n".join(srows) + "\n",
                                                encoding="utf-8")

    # events.csv (optional off-plate intervals)
    if off_plate_intervals:
        erows = ["trial_idx,t_start_s,t_end_s,t_start_wall,"
                 "t_end_wall,duration_s,n_samples"]
        for i, (s, e) in enumerate(off_plate_intervals):
            erows.append(
                f"{i},{s},{e},{rec_start_wall+s},{rec_start_wall+e},"
                f"{e-s},{int((e-s)*fs)}")
        (sd / "events.csv").write_text("\n".join(erows) + "\n",
                                          encoding="utf-8")

    # poses_C0.npz (optional)
    if with_pose:
        n_frames = 60
        kpts = np.full((n_frames, 33, 2), np.nan, dtype=np.float32)
        vis  = np.zeros((n_frames, 33), dtype=np.float32)
        # Right wrist visible, oscillating across the frame
        wrist_idx = 16
        kpts[:, wrist_idx, 0] = np.linspace(100, 540, n_frames)
        kpts[:, wrist_idx, 1] = 240.0
        vis[:, wrist_idx] = 1.0
        world = np.full((n_frames, 33, 3), np.nan, dtype=np.float32)
        angles = np.full((n_frames, 12), np.nan, dtype=np.float32)
        from src.analysis.pose2d import ANGLE_NAMES
        np.savez(sd / "poses_C0.npz",
                  cam_id="C0",
                  kpts_mp33=kpts,
                  visibility_mp33=vis,
                  world_mp33=world,
                  angles=angles,
                  angle_names=np.array(ANGLE_NAMES),
                  fps=30.0,
                  image_size=np.array([640, 480], dtype=np.int32),
                  backend="mediapipe",
                  model_complexity=1)
        # timestamps.csv covering the same wall window
        trows = ["frame_idx,t_monotonic_ns,t_wall_s"]
        for i in range(n_frames):
            t = rec_start_wall + i * (n_samples / fs) / n_frames
            trows.append(f"{i},{i*int(1e7)},{t:.6f}")
        (sd / "C0.timestamps.csv").write_text(
            "\n".join(trows) + "\n", encoding="utf-8")

    return sd


# ────────────────────────────────────────────────────────────────────────────
# _fmt + _digits_for
# ────────────────────────────────────────────────────────────────────────────
def test_fmt_handles_none_and_nan():
    assert _fmt(None, 3) == ""
    assert _fmt(float("nan"), 3) == ""
    assert _fmt(float("inf"), 3) == ""


def test_fmt_rounds_to_digits():
    assert _fmt(1.234567, 3) == "1.235"
    assert _fmt(1.0, 6) == "1.000000"


def test_digits_for_picks_sensible_defaults():
    assert _digits_for("t_s") == 6
    assert _digits_for("b1_tl_N") == 3
    assert _digits_for("enc1_mm") == 3
    assert _digits_for("right_wrist_x_px") == 2
    assert _digits_for("right_wrist_vis") == 3
    assert _digits_for("angle_knee_L_deg") == 2


# ────────────────────────────────────────────────────────────────────────────
# Event column builders
# ────────────────────────────────────────────────────────────────────────────
def test_off_plate_col_marks_intervals():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(
            Path(tmp), n_samples=200, fs=100.0,
            off_plate_intervals=[(0.50, 0.80)])
        t_s = np.arange(200, dtype=np.float64) / 100.0
        col = _build_off_plate_col(sd, t_s)
        # Samples 50..80 inclusive should be 1
        assert col[49] == 0
        assert col[50] == 1
        assert col[79] == 1
        assert col[80] == 1
        assert col[81] == 0


def test_stim_event_col_lands_on_closest_sample():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(
            Path(tmp), n_samples=600, fs=100.0,
            stims=[{"t_offs": 1.234, "label": "pos_N",
                     "tx": 0.5, "ty": 0.2},
                    {"t_offs": 4.567, "label": "pos_E",
                     "tx": 0.85, "ty": 0.5}])
        t_s = np.arange(600, dtype=np.float64) / 100.0
        labels, txs, tys = _build_stim_event_cols(sd, t_s, t0_wall=1000.0)
        # Stim 1 at t=1.234s → sample 123 (closest)
        assert labels[123] == "pos_N"
        assert txs[123] == "0.5000"
        assert tys[123] == "0.2000"
        # Stim 2 at t=4.567s → sample 457
        assert labels[457] == "pos_E"
        # Other rows must be empty
        assert labels[0] == ""
        assert labels[100] == ""


def test_off_plate_col_no_events_csv_returns_zeros():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(Path(tmp), n_samples=100, fs=100.0)
        t_s = np.arange(100, dtype=np.float64) / 100.0
        col = _build_off_plate_col(sd, t_s)
        assert all(v == 0 for v in col)


# ────────────────────────────────────────────────────────────────────────────
# End-to-end CSV export
# ────────────────────────────────────────────────────────────────────────────
def test_export_writes_utf8_bom():
    """Excel & Notepad both happy if the file starts with EF BB BF."""
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(Path(tmp), n_samples=50, fs=100.0)
        out = export_session_csv(sd)
        ts = out["timeseries"]
        with open(ts, "rb") as f:
            head = f.read(3)
        assert head == b"\xef\xbb\xbf"


def test_header_order_events_immediately_after_time():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(
            Path(tmp), n_samples=50, fs=100.0,
            stims=[{"t_offs": 0.10, "label": "pos_N",
                     "tx": 0.5, "ty": 0.2}],
            off_plate_intervals=[(0.20, 0.30)])
        out = export_session_csv(sd)
        with open(out["timeseries"], "r",
                   encoding="utf-8-sig", newline="") as f:
            r = _csv.reader(f)
            header = next(r)
        # First six columns: time + events
        assert header[:6] == [
            "t_s", "t_wall_s",
            "event_stim", "event_target_x_norm",
            "event_target_y_norm", "event_off_plate",
        ]
        # Force columns immediately after
        assert header[6] == "b1_tl_N"


def test_export_event_columns_populated_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(
            Path(tmp), n_samples=200, fs=100.0,
            stims=[{"t_offs": 0.50, "label": "pos_E",
                     "tx": 0.85, "ty": 0.5}],
            off_plate_intervals=[(1.00, 1.20)])
        out = export_session_csv(sd)
        with open(out["timeseries"], "r",
                   encoding="utf-8-sig", newline="") as f:
            r = _csv.DictReader(f)
            rows = list(r)
        # Sample 50 should carry the stim label + target xy
        assert rows[50]["event_stim"] == "pos_E"
        assert rows[50]["event_target_x_norm"] == "0.8500"
        assert rows[50]["event_target_y_norm"] == "0.5000"
        # Surrounding rows empty
        assert rows[49]["event_stim"] == ""
        assert rows[51]["event_stim"] == ""
        # Off-plate window 1.00-1.20s → samples 100..120 marked 1
        assert rows[100]["event_off_plate"] == "1"
        assert rows[120]["event_off_plate"] == "1"
        assert rows[99]["event_off_plate"] == "0"
        assert rows[121]["event_off_plate"] == "0"


def test_export_writes_pose_native_when_poses_present():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(Path(tmp), n_samples=200, fs=100.0,
                                  with_pose=True)
        out = export_session_csv(sd)
        assert out["pose_native"] is not None
        assert out["pose_native"].exists()
        # The pose-native CSV must contain right_wrist_x_px column
        with open(out["pose_native"], "r",
                   encoding="utf-8-sig", newline="") as f:
            header = next(_csv.reader(f))
        assert "right_wrist_x_px" in header
        assert "right_wrist_y_px" in header
        # Time + event columns at the front
        assert header[:7] == [
            "frame_idx", "t_s", "t_wall_s",
            "event_stim", "event_target_x_norm",
            "event_target_y_norm", "event_off_plate",
        ]


def test_export_no_pose_skips_pose_native():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(Path(tmp), n_samples=50, fs=100.0,
                                  with_pose=False)
        out = export_session_csv(sd)
        assert out["pose_native"] is None


def test_export_nan_renders_as_empty_string():
    """Force a NaN through the pipeline by writing a forces.csv whose
    last sample has empty CoP fields, then verify the timeseries cells
    for that row are empty (not 'nan')."""
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(Path(tmp), n_samples=30, fs=100.0)
        # Patch forces.csv last row → blank CoP cells (NaN)
        fp = sd / "forces.csv"
        lines = fp.read_text(encoding="utf-8").splitlines()
        head = lines[0]
        last = lines[-1].split(",")
        # cop_world_x_mm, cop_world_y_mm columns 13, 14 (0-indexed)
        last[13] = ""; last[14] = ""
        lines[-1] = ",".join(last)
        fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = export_session_csv(sd)
        with open(out["timeseries"], "r",
                   encoding="utf-8-sig", newline="") as f:
            rows = list(_csv.DictReader(f))
        assert rows[-1]["cop_x_mm"] == ""
        assert rows[-1]["cop_y_mm"] == ""


def test_export_row_count_matches_force_samples():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_min_session(Path(tmp), n_samples=137, fs=100.0)
        out = export_session_csv(sd)
        with open(out["timeseries"], "r",
                   encoding="utf-8-sig", newline="") as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == 137


# ────────────────────────────────────────────────────────────────────────────
# Direct runner
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items()
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print()
    if failed:
        print(f"=== {failed}/{len(fns)} tests failed ===")
        sys.exit(1)
    print(f"=== All {len(fns)} tests passed ===")
