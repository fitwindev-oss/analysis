"""
Barbell / dumbbell kinematic analysis from linear encoders.

The two encoders are assumed to track **position** (mm) of a tool (bar,
dumbbell, etc.). Typical use:
  - Encoder 1 tracks one end of the barbell
  - Encoder 2 tracks the other end (or a second tool)

For vertical lifts (squat, bench, deadlift), encoder displacement IS the bar
height. For swept-path lifts (clean, snatch), the encoder can track either
vertical height or a guide-rail distance depending on setup.

Metrics per detected rep:
  rom_mm           range of motion (max - min displacement)
  eccentric_time_s time from start of descent to bottom
  concentric_time_s time from bottom back up to ROM reached
  mean_con_vel_m_s mean concentric velocity (m/s)
  peak_con_vel_m_s peak concentric velocity
  mean_con_power_w mean concentric power (if bar_mass_kg given)
  peak_con_power_w peak concentric power
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

import config
from .common import (
    ForceSession, butter_lowpass, numerical_derivative, load_force_session,
)


def _encoder_available(channel: int) -> bool:
    """Returns the config availability flag for encoder 1 or 2.
    Defaults to True for 1 and False for 2 when the attribute is missing,
    matching the project default (right encoder currently broken)."""
    if channel == 1:
        return bool(getattr(config, "ENCODER1_AVAILABLE", True))
    if channel == 2:
        return bool(getattr(config, "ENCODER2_AVAILABLE", False))
    return False


@dataclass
class RepMetrics:
    idx: int
    t_start_s: float
    t_bottom_s: float
    t_end_s: float
    rom_mm: float
    eccentric_time_s: float
    concentric_time_s: float
    mean_con_vel_m_s: float
    peak_con_vel_m_s: float
    mean_con_power_w: float
    peak_con_power_w: float


@dataclass
class EncoderResult:
    channel: int                     # 1 or 2
    duration_s: float
    n_reps: int
    reps: list = field(default_factory=list)

    mean_rom_mm: float = 0.0
    mean_con_vel_m_s: float = 0.0
    peak_con_vel_m_s: float = 0.0
    mean_con_power_w: float = 0.0
    peak_con_power_w: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reps"] = [asdict(r) for r in self.reps]
        return d


def detect_reps(displacement_mm: np.ndarray, fs: float,
                min_rom_mm: float = 150.0,
                min_rest_s: float = 0.4) -> list[tuple[int, int, int]]:
    """
    Rep detection via threshold-crossing around a reference "top" level.

    Each rep is a contiguous region where displacement drops below
    (top - min_rom_mm/2). The rep's bottom is the minimum within that region;
    start/end are the entry and exit indices across the threshold.

    This is robust when the resting level between reps is flat (no local
    peak), which defeats pure find_peaks-based detection.

    Returns list of (i_start, i_bottom, i_end) indices.
    """
    x = butter_lowpass(displacement_mm, 3.0, fs)
    # "Top" level = 90th percentile of displacement (robust against outliers)
    top_level = np.percentile(x, 90)
    threshold = top_level - min_rom_mm * 0.5
    min_rest = max(int(min_rest_s * fs), 5)

    below = x < threshold
    # Find contiguous True regions
    reps: list[tuple[int, int, int]] = []
    i = 0
    n = len(below)
    while i < n:
        if not below[i]:
            i += 1
            continue
        i_start = i
        while i < n and below[i]:
            i += 1
        i_end = i - 1
        # Expand slightly to include the top-of-rep on both sides
        i_start = max(0, i_start - min_rest // 2)
        i_end   = min(n - 1, i_end + min_rest // 2)
        segment = x[i_start:i_end + 1]
        rom = float(segment.max() - segment.min())
        if rom < min_rom_mm:
            continue
        i_bottom = i_start + int(np.argmin(segment))
        reps.append((i_start, i_bottom, i_end))
    return reps


def analyze_encoder(force: ForceSession, channel: int = 1,
                    bar_mass_kg: float = 20.0,
                    min_rom_mm: float = 150.0) -> EncoderResult:
    """
    Run rep detection + per-rep metrics on one encoder channel.

    bar_mass_kg: load on the encoder (for power computation). 20 kg = empty Olympic bar.
    """
    if channel == 1:
        x = force.enc1
    elif channel == 2:
        x = force.enc2
    else:
        raise ValueError(f"channel must be 1 or 2, got {channel}")
    # Soft guard — the data in forces.csv may be noise if the channel
    # hardware was broken at record time. Keep analysis running so
    # historical data still works, but warn loudly.
    if not _encoder_available(channel):
        warnings.warn(
            f"encoder channel {channel} is flagged unavailable in "
            f"config.ENCODER{channel}_AVAILABLE - rep metrics may be "
            f"invalid if the sensor was disconnected during recording.",
            RuntimeWarning, stacklevel=2,
        )
    fs = force.fs
    t  = force.t_s
    # Convert to meters for velocity/power
    x_m = x / 1000.0

    # Smooth displacement before differentiation
    x_s = butter_lowpass(x_m, 6.0, fs)
    v   = numerical_derivative(x_s, fs)          # m/s
    a   = numerical_derivative(v,   fs)          # m/s^2
    # Power = F * v = m*(g + a) * v  ≈ m*g*v for steady lifts; use full F for accuracy
    force_on_bar_n = bar_mass_kg * (9.80665 + a)
    power = force_on_bar_n * v                   # W

    reps_idx = detect_reps(x, fs, min_rom_mm=min_rom_mm)
    rep_list: list[RepMetrics] = []
    for k, (i_start, i_bot, i_end) in enumerate(reps_idx):
        rom = float(x[i_start:i_end].max() - x[i_start:i_end].min())
        t_start = float(t[i_start])
        t_bot   = float(t[i_bot])
        t_end   = float(t[i_end])
        ecc_t   = t_bot - t_start
        con_t   = t_end - t_bot

        # Concentric velocity is positive (going up) from bottom to top
        v_con = v[i_bot:i_end + 1]
        mean_v = float(np.mean(np.maximum(v_con, 0.0)))
        peak_v = float(np.max(v_con))

        p_con = power[i_bot:i_end + 1]
        mean_p = float(np.mean(np.maximum(p_con, 0.0)))
        peak_p = float(np.max(p_con))

        rep_list.append(RepMetrics(
            idx=k, t_start_s=t_start, t_bottom_s=t_bot, t_end_s=t_end,
            rom_mm=rom,
            eccentric_time_s=ecc_t, concentric_time_s=con_t,
            mean_con_vel_m_s=mean_v, peak_con_vel_m_s=peak_v,
            mean_con_power_w=mean_p, peak_con_power_w=peak_p,
        ))

    if rep_list:
        res_mean_rom = float(np.mean([r.rom_mm for r in rep_list]))
        res_mean_v   = float(np.mean([r.mean_con_vel_m_s for r in rep_list]))
        res_peak_v   = float(np.max( [r.peak_con_vel_m_s for r in rep_list]))
        res_mean_p   = float(np.mean([r.mean_con_power_w for r in rep_list]))
        res_peak_p   = float(np.max( [r.peak_con_power_w for r in rep_list]))
    else:
        res_mean_rom = res_mean_v = res_peak_v = res_mean_p = res_peak_p = 0.0

    return EncoderResult(
        channel=channel,
        duration_s=float(t[-1] - t[0]),
        n_reps=len(rep_list),
        reps=rep_list,
        mean_rom_mm=res_mean_rom,
        mean_con_vel_m_s=res_mean_v,
        peak_con_vel_m_s=res_peak_v,
        mean_con_power_w=res_mean_p,
        peak_con_power_w=res_peak_p,
    )


def analyze_encoder_file(session_dir, **kw) -> EncoderResult:
    return analyze_encoder(load_force_session(session_dir), **kw)


class RealtimeRepCounter:
    """Streaming rep counter for live MeasureTab feedback.

    State machine:
      - "up"   : signal is near the running max; once it drops below
                 (top - min_rom/2), transition to "down".
      - "down" : signal has descended. Track running min (the bottom).
                 Once the signal rises back above (bottom + min_rom/2),
                 count 1 rep and transition back to "up".

    Hysteresis = min_rom/2 in both directions (≈ 40 mm for the default
    80 mm ROM), well above encoder noise (<1 mm after MA smoothing), so
    false counts from vibration are suppressed.

    Aligned with the offline ``detect_reps()`` algorithm: both use the
    same ROM criterion, so the live number should be within ±1 of the
    post-recording analyzer output.

    Usage:
        counter = RealtimeRepCounter(min_rom_mm=80.0)
        for sample in stream:
            n = counter.push(sample.enc1_mm)
            dashboard.set_rep_count(n)
    """

    def __init__(self, min_rom_mm: float = 80.0):
        self.min_rom: float = float(min_rom_mm)
        self.reset()

    def reset(self) -> None:
        self.state: str = "up"
        self.top_mm:    Optional[float] = None
        self.bottom_mm: Optional[float] = None
        self.rep_count: int = 0

    def push(self, mm: float) -> int:
        """Process one sample, return the current rep count."""
        v = float(mm)
        if self.top_mm is None:
            self.top_mm = v

        if self.state == "up":
            # Track the highest point so the subsequent descent threshold
            # floats upward with bar rack height / setup changes.
            if v > self.top_mm:
                self.top_mm = v
            if v < self.top_mm - self.min_rom * 0.5:
                self.state = "down"
                self.bottom_mm = v
        else:   # "down"
            assert self.bottom_mm is not None
            if v < self.bottom_mm:
                self.bottom_mm = v
            elif v > self.bottom_mm + self.min_rom * 0.5:
                self.state = "up"
                self.rep_count += 1
                # Reset top to the current position so the next rep's
                # descent is judged from here (not from a stale setup high).
                self.top_mm = v

        return self.rep_count
