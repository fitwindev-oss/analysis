"""
Common utilities for analysis modules.

Every analyzer consumes a "session" folder - a directory containing at minimum
forces.csv, plus optionally videos, timestamps, and 3D pose data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

import config


G = 9.80665    # m/s^2


# ─────────────────────────────────────────────────────────────────────────────
# Session I/O
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForceSession:
    """Force/encoder timeseries loaded from forces.csv."""
    t_s:     np.ndarray       # seconds from start (float64)
    b1:      np.ndarray       # (N, 4)  Board1 TL/TR/BL/BR in N
    b2:      np.ndarray       # (N, 4)  Board2 TL/TR/BL/BR in N
    b1_total: np.ndarray      # (N,) N
    b2_total: np.ndarray      # (N,) N
    total:   np.ndarray       # (N,) N
    cop_x:   np.ndarray       # (N,) mm in plate-world frame
    cop_y:   np.ndarray       # (N,)
    enc1:    np.ndarray       # (N,) mm
    enc2:    np.ndarray       # (N,) mm
    fs:      float            # sampling rate Hz
    session_dir: Path = field(default_factory=lambda: Path("."))
    # Phase U3-3: per-sample on/off-plate boolean. 1 = subject on
    # plate (force above threshold), 0 = subject off plate (jump
    # flight, idle, fell off). Computed at recording time by the
    # SessionRecorder; absent in pre-U3-3 sessions, where this is
    # back-filled by load_force_session using a 50N floor.
    on_plate: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    def __len__(self) -> int:
        return len(self.t_s)

    @property
    def duration_s(self) -> float:
        return float(self.t_s[-1] - self.t_s[0]) if len(self.t_s) else 0.0

    def time_slice(self, t0: float, t1: float) -> "ForceSession":
        mask = (self.t_s >= t0) & (self.t_s <= t1)
        sliced_on_plate = (self.on_plate[mask]
                           if len(self.on_plate) == len(self.t_s)
                           else np.zeros(int(mask.sum()), dtype=np.int8))
        return ForceSession(
            t_s=self.t_s[mask],
            b1=self.b1[mask], b2=self.b2[mask],
            b1_total=self.b1_total[mask],
            b2_total=self.b2_total[mask],
            total=self.total[mask],
            cop_x=self.cop_x[mask], cop_y=self.cop_y[mask],
            enc1=self.enc1[mask], enc2=self.enc2[mask],
            fs=self.fs, session_dir=self.session_dir,
            on_plate=sliced_on_plate,
        )

    # ── Physically-clipped helpers ──────────────────────────────────────────
    # Per-board vertical force is non-negative in reality; post-zero-cal
    # drift in corner channels can push the unclipped sum slightly negative.
    # These helpers return the physically-meaningful (clipped) signals for
    # analyses that require it (WBA, CMJ peak force, body weight estimation).
    @property
    def b1_total_clipped(self) -> np.ndarray:
        return np.maximum(self.b1_total, 0.0)

    @property
    def b2_total_clipped(self) -> np.ndarray:
        return np.maximum(self.b2_total, 0.0)

    @property
    def total_clipped(self) -> np.ndarray:
        """Sum of per-corner clipped forces from both boards."""
        c = np.concatenate([self.b1, self.b2], axis=1)     # (N, 8)
        return np.maximum(c, 0.0).sum(axis=1)


def load_force_session(session_dir: str | Path) -> ForceSession:
    """Load a session's forces.csv into a ForceSession."""
    session_dir = Path(session_dir)
    fp = session_dir / "forces.csv"
    if not fp.exists():
        raise FileNotFoundError(f"missing: {fp}")
    df = pd.read_csv(fp)

    # Time axis: use t_wall anchored to first sample
    t_wall = df["t_wall"].to_numpy(dtype=np.float64)
    t_s = t_wall - t_wall[0]
    dt = np.diff(t_s)
    fs = 1.0 / float(np.median(dt)) if len(dt) > 0 else float(config.SAMPLE_RATE_HZ)

    # Corner forces (N)
    b1 = np.stack([
        df["b1_tl_N"], df["b1_tr_N"], df["b1_bl_N"], df["b1_br_N"],
    ], axis=1).astype(np.float64)
    b2 = np.stack([
        df["b2_tl_N"], df["b2_tr_N"], df["b2_bl_N"], df["b2_br_N"],
    ], axis=1).astype(np.float64)
    b1_total = b1.sum(axis=1)
    b2_total = b2.sum(axis=1)

    total = df["total_n"].to_numpy(np.float64) if "total_n" in df else b1_total + b2_total
    cop_x = pd.to_numeric(df["cop_world_x_mm"], errors="coerce").to_numpy(np.float64)
    cop_y = pd.to_numeric(df["cop_world_y_mm"], errors="coerce").to_numpy(np.float64)

    enc1 = df["enc1_mm"].to_numpy(np.float64) if "enc1_mm" in df else np.zeros_like(t_s)
    enc2 = df["enc2_mm"].to_numpy(np.float64) if "enc2_mm" in df else np.zeros_like(t_s)

    # on_plate column (Phase U3-3). For pre-U3-3 sessions the column
    # is missing — back-fill with the same fixed 20 N threshold the
    # recorder uses (matches CMJ analyser's flight_threshold_n). 20 N
    # is the physical "feet have left" point and doesn't body-weight
    # scale; see src/capture/cop_state.py for rationale.
    if "on_plate" in df.columns:
        on_plate = df["on_plate"].to_numpy(np.int8)
    else:
        on_plate = (total >= 20.0).astype(np.int8)

    return ForceSession(
        t_s=t_s, b1=b1, b2=b2,
        b1_total=b1_total, b2_total=b2_total, total=total,
        cop_x=cop_x, cop_y=cop_y,
        enc1=enc1, enc2=enc2,
        fs=fs, session_dir=session_dir,
        on_plate=on_plate,
    )


def compute_departure_events(force: "ForceSession",
                             entry_threshold_n: Optional[float] = None,
                             exit_threshold_n: Optional[float] = None,
                             min_duration_s: float = 0.05) -> list[dict]:
    """Recompute departure events with hysteresis from a ForceSession.

    Two thresholds (matching the CMJ analyser):

      * **entry** (default 20 N): on-plate → off-plate when GRF drops
        below this. Strict, near-zero — catches actual takeoff.
      * **exit** (default ``max(50 % BW, 150 N)``): off-plate → on-plate
        when GRF rises back above this. Higher than entry to skip
        force-plate ringing — after a CMJ landing, the plate's
        zero-baseline oscillates around 20 N for ~30-50 ms before
        actual contact reasserts. A single 20 N exit threshold would
        false-trigger and split one flight into two events.

    BW is read from session.json's ``subject_kg`` when available;
    otherwise the exit threshold falls back to ``150 N`` (a value that
    safely clears the typical ringing band).

    Use this in UIs that should always reflect the *current* threshold
    semantics, independent of what was stored in events.csv at
    recording time. Old sessions recorded under a different threshold
    get their bands automatically realigned.
    """
    # Lazy import to avoid src.capture <-> src.analysis circular when
    # analysis modules are imported during recorder startup.
    from src.capture.cop_state import DEPARTURE_THRESHOLD_N

    n = len(force.t_s)
    if n == 0:
        return []

    entry = float(DEPARTURE_THRESHOLD_N
                  if entry_threshold_n is None else entry_threshold_n)
    if exit_threshold_n is None:
        # Derive from subject_kg via session.json (CMJ-style:
        # max(BW × 0.5, 150 N)).
        subject_kg = 0.0
        try:
            import json as _json
            meta_p = Path(force.session_dir) / "session.json"
            if meta_p.exists():
                meta = _json.loads(meta_p.read_text(encoding="utf-8"))
                subject_kg = float(meta.get("subject_kg") or 0.0)
        except Exception:
            subject_kg = 0.0
        exit_th = max(150.0, subject_kg * 9.80665 * 0.50)
    else:
        exit_th = float(exit_threshold_n)
    # Belt-and-suspenders: exit must be ≥ entry, otherwise hysteresis
    # makes no sense.
    exit_th = max(exit_th, entry)

    events: list[dict] = []
    on_plate = True             # assume subject starts on plate
    open_start_idx: Optional[int] = None

    for i in range(n):
        f = float(force.total[i])
        if on_plate:
            if f < entry:
                on_plate = False
                open_start_idx = i
        else:
            if f >= exit_th:
                on_plate = True
                t_start = float(force.t_s[open_start_idx])
                t_end   = float(force.t_s[i])
                duration = t_end - t_start
                if duration >= min_duration_s - 1e-9:
                    events.append({
                        "trial_idx":    len(events),
                        "t_start_s":    round(t_start, 4),
                        "t_end_s":      round(t_end, 4),
                        # ForceSession.t_s is normalised to start at 0
                        # so wall fields are informational only.
                        "t_start_wall": round(t_start, 6),
                        "t_end_wall":   round(t_end, 6),
                        "duration_s":   round(duration, 4),
                        "n_samples":    int(i - open_start_idx),
                    })
                open_start_idx = None

    # Still-open interval at end of recording — close it at the last
    # sample. (Subject was off the plate when the recording stopped.)
    if not on_plate and open_start_idx is not None:
        t_start = float(force.t_s[open_start_idx])
        t_end   = float(force.t_s[-1])
        duration = t_end - t_start
        if duration >= min_duration_s - 1e-9:
            events.append({
                "trial_idx":    len(events),
                "t_start_s":    round(t_start, 4),
                "t_end_s":      round(t_end, 4),
                "t_start_wall": round(t_start, 6),
                "t_end_wall":   round(t_end, 6),
                "duration_s":   round(duration, 4),
                "n_samples":    int(n - open_start_idx),
            })

    return events


def load_departure_events(session_dir: str | Path) -> list[dict]:
    """Load events.csv as a list of departure-interval dicts.

    Returns ``[]`` when the file is missing (pre-U3-3 sessions, or a
    session where no off-plate intervals lasted long enough to qualify).
    Numeric columns are coerced to float; ``trial_idx``/``n_samples``
    are coerced to int. Callers can rely on the keys
    ``t_start_s, t_end_s, duration_s, n_samples`` always being present.
    """
    fp = Path(session_dir) / "events.csv"
    if not fp.exists():
        return []
    try:
        df = pd.read_csv(fp)
    except Exception:
        return []
    out: list[dict] = []
    for _, row in df.iterrows():
        out.append({
            "trial_idx":    int(row.get("trial_idx", 0)),
            "t_start_s":    float(row["t_start_s"]),
            "t_end_s":      float(row["t_end_s"]),
            "t_start_wall": float(row.get("t_start_wall", 0.0) or 0.0),
            "t_end_wall":   float(row.get("t_end_wall",   0.0) or 0.0),
            "duration_s":   float(row["duration_s"]),
            "n_samples":    int(row.get("n_samples", 0)),
        })
    return out


def load_poses3d_world(session_dir: str | Path) -> dict | None:
    """
    Load world-frame 3D poses if available.
    Expects data/calibration/poses3d_world_<session_name>.npz
    or a session-local poses3d_world.npz.
    """
    session_dir = Path(session_dir)
    # Prefer session-local file if it exists
    local = session_dir / "poses3d_world.npz"
    if local.exists():
        p = np.load(local, allow_pickle=True)
        return dict(
            kpts3d=p["kpts3d_world"].astype(np.float32),
            fps=float(p["fps"]),
            joint_names=list(map(str, p["joint_names"].tolist())),
        )
    # Fallback: central calib dir
    calib_file = config.CALIB_DIR / f"poses3d_world_{session_dir.name}.npz"
    if calib_file.exists():
        p = np.load(calib_file, allow_pickle=True)
        return dict(
            kpts3d=p["kpts3d_world"].astype(np.float32),
            fps=float(p["fps"]),
            joint_names=list(map(str, p["joint_names"].tolist())),
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────

def butter_lowpass(x: np.ndarray, cutoff_hz: float, fs: float,
                   order: int = 4) -> np.ndarray:
    """Zero-lag Butterworth low-pass filter. x may be 1D or 2D (axis=0 time)."""
    if len(x) < 3 * order:
        return x.copy()
    nyq = fs / 2.0
    Wn = min(cutoff_hz / nyq, 0.999)
    b, a = butter(order, Wn, btype="low")
    return filtfilt(b, a, x, axis=0)


def moving_average(x: np.ndarray, win_samples: int) -> np.ndarray:
    """Simple centered moving average."""
    if win_samples < 2:
        return x.copy()
    kernel = np.ones(win_samples) / win_samples
    return np.convolve(x, kernel, mode="same")


def numerical_derivative(x: np.ndarray, fs: float) -> np.ndarray:
    """Central-difference derivative of a 1D signal. Returns same-length array."""
    dt = 1.0 / fs
    out = np.zeros_like(x)
    out[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    out[0] = (x[1] - x[0]) / dt
    out[-1] = (x[-1] - x[-2]) / dt
    return out


def confidence_ellipse_area_95(x: np.ndarray, y: np.ndarray) -> float:
    """
    95% confidence ellipse area for a 2D point cloud (mm^2).
    Uses the chi-squared approximation: area = pi * 5.991 * sqrt(lambda1 * lambda2)
    where lambda_i are eigenvalues of the sample covariance.
    """
    xy = np.stack([x, y], axis=1)
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = (xy.T @ xy) / max(len(xy) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    return float(np.pi * 5.991 * np.sqrt(eigvals[0] * eigvals[1]))
