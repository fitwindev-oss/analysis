"""
Per-session plain-text CSV export — readable in Notepad / VS Code / vim
without Excel.

Mirrors the data shape of ``excel_export.py`` (TimeSeries + Pose_native
sheets) but writes UTF-8-BOM CSV files instead of an .xlsx workbook.
The headline difference for the operator is the **event columns** placed
right next to ``t_s`` so timeline anomalies (stimuli, off-plate events)
read directly off the row:

    t_s , t_wall_s , event_stim , event_target_x_norm ,
    event_target_y_norm , event_off_plate , <force...> ,
    <encoder...> , <CoP...> , <pose 99> , <vel 66> ,
    <angles 12> , <angle_vels 12>

Two files are produced (in the session folder by default):

  <session>_timeseries.csv    100 Hz force grid + interpolated pose
  <session>_pose_native.csv   pose at the camera's native fps (only
                              written when poses_*.npz exists)

Numeric helpers (low-pass, central-difference velocity, per-board CoP,
pose interpolation) are imported from ``excel_export`` so both export
backends compute identical numbers.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.analysis.common import ForceSession, load_force_session
from src.analysis.excel_export import (
    _per_board_cop, _pose_to_force_grid, _velocity,
)
from src.analysis.pose2d import (
    ANGLE_NAMES, Pose2DSeries, get_record_start_wall_s,
    load_session_pose2d, load_video_timestamps,
)
from src.pose.mediapipe_backend import MP33_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Event-column builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_stim_event_cols(session_dir: Path,
                            t_s: np.ndarray,
                            t0_wall: float
                            ) -> tuple[list[str], list[str], list[str]]:
    """Return three parallel string columns aligned to ``t_s``:

      event_stim          response_type / target_label, only on the
                          force sample closest to each stim's t_wall
      event_target_x_norm  cognitive_reaction target x (same row only)
      event_target_y_norm  cognitive_reaction target y (same row only)

    Strings used so empty cells are an empty string (notepad-friendly)
    rather than 0 or NaN.
    """
    n = len(t_s)
    col_label = [""] * n
    col_tx    = [""] * n
    col_ty    = [""] * n

    fp = session_dir / "stimulus_log.csv"
    if not fp.exists():
        return col_label, col_tx, col_ty

    try:
        import pandas as pd
        df = pd.read_csv(fp)
    except Exception:
        return col_label, col_tx, col_ty

    if "t_wall" not in df.columns:
        return col_label, col_tx, col_ty

    has_target = ("target_x_norm" in df.columns
                  and "target_y_norm" in df.columns)
    has_label  = "target_label"   in df.columns
    has_resp   = "response_type"  in df.columns

    for _, row in df.iterrows():
        try:
            tw = float(row.get("t_wall", float("nan")))
        except Exception:
            continue
        if not np.isfinite(tw):
            continue
        t_stim = tw - t0_wall
        # Snap to nearest force sample (~10 ms grid)
        if t_stim < float(t_s[0]) - 0.05:
            continue
        if t_stim > float(t_s[-1]) + 0.05:
            continue
        idx = int(np.argmin(np.abs(t_s - t_stim)))
        # Prefer target_label (V6) over response_type (legacy reaction)
        label = ""
        if has_label:
            v = row.get("target_label")
            if v is not None and str(v).strip() and str(v) != "nan":
                label = str(v).strip()
        if not label and has_resp:
            v = row.get("response_type")
            if v is not None and str(v).strip() and str(v) != "nan":
                label = str(v).strip()
        col_label[idx] = label
        if has_target:
            tx = row.get("target_x_norm")
            ty = row.get("target_y_norm")
            try:
                tx_f = float(tx)
                if np.isfinite(tx_f):
                    col_tx[idx] = f"{tx_f:.4f}"
            except Exception:
                pass
            try:
                ty_f = float(ty)
                if np.isfinite(ty_f):
                    col_ty[idx] = f"{ty_f:.4f}"
            except Exception:
                pass
    return col_label, col_tx, col_ty


def _build_off_plate_col(session_dir: Path,
                          t_s: np.ndarray) -> list[int]:
    """Return a 0/1 column where 1 marks samples inside any off-plate
    interval recorded in events.csv. Empty list of intervals → all 0.

    A single sample has duration 1/fs; interval inclusion is
    ``t_start_s ≤ t_s ≤ t_end_s`` so the band shading aligns with what
    the replay timeline shows.
    """
    n = len(t_s)
    col = [0] * n
    fp = session_dir / "events.csv"
    if not fp.exists():
        return col
    try:
        import pandas as pd
        df = pd.read_csv(fp)
    except Exception:
        return col
    if "t_start_s" not in df.columns or "t_end_s" not in df.columns:
        return col
    if df.empty:
        return col

    starts = df["t_start_s"].to_numpy(np.float64)
    ends   = df["t_end_s"].to_numpy(np.float64)
    # Vectorised inclusive masking with a tiny epsilon so a sample whose
    # t_s equals e exactly survives float-quantization. Without this,
    # ``np.arange(N) / fs`` produces e.g. t_s[120] = 1.2000000000000001
    # while the events.csv carries e = 1.20 — an unintended exclusion.
    eps = 1e-9
    for s, e in zip(starts, ends):
        if not (np.isfinite(s) and np.isfinite(e)):
            continue
        if e < s:
            continue
        mask = (t_s >= s - eps) & (t_s <= e + eps)
        if mask.any():
            for k in np.where(mask)[0]:
                col[int(k)] = 1
    return col


# ─────────────────────────────────────────────────────────────────────────────
# Number formatting (compact + lossless-enough for analysis)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, digits: int) -> str:
    """Format a float for the CSV. NaN/None → empty string."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _digits_for(name: str) -> int:
    """Pick a sensible decimal count per column name pattern."""
    if name == "t_s" or name == "t_wall_s":
        return 6
    if name.endswith("_N") or name.endswith("_total_N"):
        return 3
    if name.endswith("_mm"):
        return 3
    if name.endswith("_x_px") or name.endswith("_y_px"):
        return 2
    if name.endswith("_vis"):
        return 3
    if name.endswith("_vx_px_s") or name.endswith("_vy_px_s"):
        return 2
    if name.endswith("_deg"):
        return 2
    if name.endswith("_deg_s"):
        return 2
    return 4


# ─────────────────────────────────────────────────────────────────────────────
# Sheet writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_timeseries_csv(out_path: Path,
                           session_dir: Path,
                           force: ForceSession,
                           pose: Optional[Pose2DSeries],
                           cam_id: Optional[str],
                           progress_cb: Optional[Callable[[str], None]]
                           ) -> None:
    """100 Hz force grid + interpolated pose + event columns.

    Column ordering — events sit RIGHT next to the time columns so the
    operator can scan timeline anomalies without scrolling:
        time(2) | events(4) | force(11) | encoder(2) | CoP(6) |
        pose_xy_vis(99) | pose_velocity(66) | angles(12) | angle_vel(12)
    """
    t = force.t_s.astype(np.float64)
    n = len(t)

    # ── time columns ────────────────────────────────────────────────────
    t_wall = np.full(n, np.nan, dtype=np.float64)
    t0_wall = float("nan")
    try:
        import pandas as pd
        df_tw = pd.read_csv(session_dir / "forces.csv", usecols=["t_wall"])
        if len(df_tw) == n:
            t_wall = df_tw["t_wall"].to_numpy(dtype=np.float64)
            t0_wall = float(t_wall[0])
    except Exception:
        pass

    # ── event columns (built first so they sit next to time) ────────────
    if progress_cb:
        progress_cb("building event columns")
    if np.isfinite(t0_wall):
        ev_label, ev_tx, ev_ty = _build_stim_event_cols(
            session_dir, t, t0_wall)
    else:
        ev_label = [""] * n
        ev_tx    = [""] * n
        ev_ty    = [""] * n
    ev_off_plate = _build_off_plate_col(session_dir, t)

    # ── force, encoder, CoP ─────────────────────────────────────────────
    cols: dict[str, np.ndarray] = {}
    corner_names = ["b1_tl_N", "b1_tr_N", "b1_bl_N", "b1_br_N",
                    "b2_tl_N", "b2_tr_N", "b2_bl_N", "b2_br_N"]
    for i, name in enumerate(corner_names[:4]):
        cols[name] = force.b1[:, i].astype(np.float64)
    for i, name in enumerate(corner_names[4:]):
        cols[name] = force.b2[:, i].astype(np.float64)
    cols["b1_total_N"] = force.b1_total.astype(np.float64)
    cols["b2_total_N"] = force.b2_total.astype(np.float64)
    cols["total_N"]    = force.total.astype(np.float64)
    cols["enc1_mm"] = force.enc1.astype(np.float64)
    cols["enc2_mm"] = force.enc2.astype(np.float64)
    pb = _per_board_cop(force)
    for k, v in pb.items():
        cols[k] = v
    cols["cop_x_mm"] = force.cop_x.astype(np.float64)
    cols["cop_y_mm"] = force.cop_y.astype(np.float64)

    # on_plate (sample-by-sample 20 N gate), kept for parity with
    # forces.csv. Different from event_off_plate which marks sustained
    # off-plate intervals from events.csv.
    on_plate = (force.total >= 20.0).astype(np.int8)
    cols["on_plate"] = on_plate

    # ── pose interpolated to force grid ─────────────────────────────────
    pose_cols: dict[str, np.ndarray] = {}
    if pose is not None and cam_id is not None:
        if progress_cb:
            progress_cb("pose: interpolating to force timeline")
        grid = _pose_to_force_grid(pose, session_dir, cam_id, t)
        if grid is not None:
            pose_cols = grid

    if pose_cols:
        if progress_cb:
            progress_cb("pose: computing velocities")
        for name in MP33_NAMES:
            x = pose_cols.get(f"{name}_x_px")
            y = pose_cols.get(f"{name}_y_px")
            if x is None or y is None:
                continue
            pose_cols[f"{name}_vx_px_s"] = _velocity(x, t,
                                                       filter_cutoff_hz=6.0)
            pose_cols[f"{name}_vy_px_s"] = _velocity(y, t,
                                                       filter_cutoff_hz=6.0)
        for name in ANGLE_NAMES:
            a = pose_cols.get(f"angle_{name}_deg")
            if a is None:
                continue
            pose_cols[f"angle_vel_{name}_deg_s"] = _velocity(
                a, t, filter_cutoff_hz=6.0)

    # Build the final ordered header list
    header: list[str] = ["t_s", "t_wall_s",
                         "event_stim", "event_target_x_norm",
                         "event_target_y_norm", "event_off_plate"]
    header += list(cols.keys())
    if pose_cols:
        for name in MP33_NAMES:
            for suf in ("_x_px", "_y_px", "_vis"):
                k = f"{name}{suf}"
                if k in pose_cols:
                    header.append(k)
        for name in MP33_NAMES:
            for suf in ("_vx_px_s", "_vy_px_s"):
                k = f"{name}{suf}"
                if k in pose_cols:
                    header.append(k)
        for name in ANGLE_NAMES:
            k = f"angle_{name}_deg"
            if k in pose_cols:
                header.append(k)
        for name in ANGLE_NAMES:
            k = f"angle_vel_{name}_deg_s"
            if k in pose_cols:
                header.append(k)

    # Pre-format every numeric column once so per-row writes only do
    # list lookups (the hot loop below runs ~6 000 times for a 60 s
    # cognitive session — premature loops would dominate runtime).
    if progress_cb:
        progress_cb(f"writing {n} rows × {len(header)} cols")

    formatted: dict[str, list[str]] = {}
    for h in header:
        if h in ("t_s", "t_wall_s"):
            arr = t if h == "t_s" else t_wall
            formatted[h] = [_fmt(v, 6) for v in arr]
        elif h == "event_stim":
            formatted[h] = ev_label
        elif h == "event_target_x_norm":
            formatted[h] = ev_tx
        elif h == "event_target_y_norm":
            formatted[h] = ev_ty
        elif h == "event_off_plate":
            formatted[h] = [str(v) for v in ev_off_plate]
        elif h == "on_plate":
            formatted[h] = [str(int(v)) for v in cols[h]]
        else:
            arr = cols[h] if h in cols else pose_cols[h]
            d = _digits_for(h)
            formatted[h] = [_fmt(v, d) for v in arr]

    # Write UTF-8 with BOM so Excel + Notepad both render Korean and
    # all special characters cleanly.
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for i in range(n):
            w.writerow([formatted[h][i] for h in header])


def _write_pose_native_csv(out_path: Path,
                            session_dir: Path,
                            pose: Pose2DSeries,
                            cam_id: Optional[str],
                            t0_wall: float,
                            progress_cb: Optional[Callable[[str], None]]
                            ) -> None:
    """Pose at the camera's native fps with no force interpolation.

    Adds the same event columns next to the time so that, if the
    operator wants to drill into a specific stim-trial reach, they
    can scroll to a stim row and read the surrounding pose frames at
    the camera's actual sampling rate.
    """
    walls = (load_video_timestamps(session_dir, cam_id)
             if cam_id else None)
    rec_start = get_record_start_wall_s(session_dir)
    n_frames = len(pose)
    if walls is not None and rec_start is not None and len(walls) == n_frames:
        t_s = walls.astype(np.float64) - float(rec_start)
        wall_arr = walls.astype(np.float64)
    else:
        # Fallback — assume uniform fps; t_wall left empty.
        t_s = np.arange(n_frames, dtype=np.float64) / max(pose.fps, 1.0)
        wall_arr = np.full(n_frames, np.nan, dtype=np.float64)

    if not np.isfinite(t0_wall):
        t0_wall = (float(wall_arr[0])
                   if np.isfinite(wall_arr[0])
                   else float(rec_start) if rec_start is not None
                   else 0.0)

    # Event columns aligned to pose-frame times
    ev_label, ev_tx, ev_ty = _build_stim_event_cols(
        session_dir, t_s, t0_wall) \
        if np.isfinite(t0_wall) else ([""] * n_frames, [""] * n_frames,
                                        [""] * n_frames)
    ev_off = _build_off_plate_col(session_dir, t_s)

    header = ["frame_idx", "t_s", "t_wall_s",
              "event_stim", "event_target_x_norm",
              "event_target_y_norm", "event_off_plate"]
    for n in MP33_NAMES:
        header += [f"{n}_x_px", f"{n}_y_px", f"{n}_vis"]
    for n in pose.angle_names:
        header += [f"angle_{n}_deg"]

    if progress_cb:
        progress_cb(f"writing pose-native CSV: "
                    f"{n_frames} frames × {len(header)} cols")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for i in range(n_frames):
            row: list = [
                i,
                _fmt(t_s[i], 6),
                _fmt(wall_arr[i], 6),
                ev_label[i],
                ev_tx[i],
                ev_ty[i],
                str(ev_off[i]),
            ]
            for li in range(33):
                row.append(_fmt(pose.kpts_mp33[i, li, 0], 2))
                row.append(_fmt(pose.kpts_mp33[i, li, 1], 2))
                row.append(_fmt(pose.vis_mp33[i, li], 3))
            for ai in range(pose.angles.shape[1]):
                row.append(_fmt(pose.angles[i, ai], 2))
            w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def export_session_csv(session_dir: str | Path,
                        out_path: Optional[str | Path] = None,
                        progress_cb: Optional[Callable[[str], None]] = None
                        ) -> dict[str, Path]:
    """Export one session to plain-text CSV files.

    Parameters
    ----------
    session_dir
        The session folder.
    out_path
        Path to the **timeseries** CSV. Defaults to
        ``<session_dir>/<name>_timeseries.csv``. The pose-native CSV
        is always written next to it as
        ``<name>_pose_native.csv`` (only when poses_*.npz exists).
    progress_cb
        Optional callable invoked with a status string.

    Returns
    -------
    dict
        ``{"timeseries": Path, "pose_native": Path | None}``.
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(session_dir)

    if out_path is None:
        out_ts = session_dir / f"{session_dir.name}_timeseries.csv"
    else:
        out_ts = Path(out_path)
    out_pose = out_ts.with_name(
        out_ts.stem.replace("_timeseries", "") + "_pose_native.csv")
    if "_timeseries" not in out_ts.stem:
        # User picked a custom name — derive pose path from it directly
        out_pose = out_ts.with_name(out_ts.stem + "_pose_native.csv")

    if progress_cb:
        progress_cb("loading force data")
    force = load_force_session(session_dir)

    pose_map = load_session_pose2d(session_dir)
    cam_id = next(iter(pose_map.keys())) if pose_map else None
    pose = pose_map.get(cam_id) if cam_id else None
    if progress_cb:
        progress_cb(
            f"force={len(force.t_s)} samples "
            f"{'pose=' + str(len(pose)) + ' frames' if pose else 'no pose'}")

    _write_timeseries_csv(out_ts, session_dir, force, pose, cam_id,
                           progress_cb)

    pose_path: Optional[Path] = None
    if pose is not None:
        # Reuse forces.csv first sample as t0 if possible
        t0_wall = float("nan")
        try:
            import pandas as pd
            df_tw = pd.read_csv(session_dir / "forces.csv",
                                 usecols=["t_wall"], nrows=1)
            t0_wall = float(df_tw["t_wall"].iloc[0])
        except Exception:
            pass
        _write_pose_native_csv(out_pose, session_dir, pose, cam_id,
                                t0_wall, progress_cb)
        pose_path = out_pose

    if progress_cb:
        progress_cb(f"saved: {out_ts.name}"
                    + (f" + {out_pose.name}" if pose_path else ""))

    return {"timeseries": out_ts, "pose_native": pose_path}
