"""
Smart stable-stance detector for recording scripts.

Purpose:
  Ensure the subject has stepped onto the plate and is holding a stable
  loaded stance for N seconds BEFORE the recording starts. This removes
  the historical problem of the DAQ zero-calibration (first 5 s after
  launch) happening with the subject already on the plate, which would
  cancel out body weight from all subsequent readings.

Usage pattern:
    detector = StabilityDetector(subject_kg=95.0, stability_target_s=3.0)
    # ... feed every incoming DAQ frame ...
    state = detector.update(daq_frame)
    draw_wait_overlay(preview_frame, state, subject_kg=95.0)
    if state.status == "READY":
        # transition to RECORDING
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional
import time

import cv2
import numpy as np


GRAVITY = 9.80665


@dataclass
class StabilityState:
    status: str                # 'STEP_ON' | 'STABILIZING' | 'READY' | 'TIMEOUT'
    total_n: float
    b1_n: float
    b2_n: float
    std_total_n: float
    stable_progress_s: float   # seconds stable so far
    stable_target_s: float     # required seconds
    wait_elapsed_s: float      # total seconds spent waiting


class StabilityDetector:
    """Ring-buffer stability detector for DAQ frames.

    Feed DAQ frames via `update(frame)`. Query state via the returned
    StabilityState.  When `state.status == 'READY'`, the caller should
    transition its state machine to the RECORDING phase.
    """

    def __init__(self,
                 subject_kg: float = 90.0,
                 stability_target_s: Optional[float] = None,
                 window_s: float = 1.0,
                 display_window_s: float = 2.0,
                 total_fraction: float = 0.70,
                 min_board_n: float = 100.0,
                 stability_n: Optional[float] = None,
                 timeout_s: float = 60.0,
                 stance_mode: str = "two"):
        self.subject_kg = float(subject_kg)
        # Per-stance default hold time: two-foot gets a real 3 s stability
        # hold; single-leg drops to 0.5 s so subjects with poor balance can
        # start without holding the pose for a long time. Explicit callers
        # can still override via stability_target_s.
        if stability_target_s is None:
            stability_target_s = 0.5 if stance_mode in ("left", "right") else 3.0
        self.stability_target_s = float(stability_target_s)
        self.window_s = float(window_s)
        # display_window_s >= window_s: longer averaging window just for the
        # numbers shown to the user. std-based stability check still uses
        # the shorter window so the detector stays responsive.
        self.display_window_s = max(float(display_window_s), self.window_s)
        self.total_fraction = float(total_fraction)
        self.min_board_n = float(min_board_n)
        # σ threshold raised from 30 N to 50 N to tolerate normal postural
        # sway + DAQ noise. Still tight enough to reject obvious wobble.
        self.stability_n = (
            float(stability_n) if stability_n is not None
            else max(50.0, 0.05 * self.subject_kg * GRAVITY)
        )
        self.timeout_s = float(timeout_s)
        if stance_mode not in ("two", "left", "right"):
            raise ValueError(
                f"stance_mode must be 'two', 'left', or 'right', got {stance_mode!r}")
        self.stance_mode = stance_mode
        self._target_total_n = self.total_fraction * self.subject_kg * GRAVITY
        # For single-leg stances we require at least 60% of BW on the loaded
        # board, and the other board to stay below 100 N. total_fraction is
        # still used as a sanity floor on the loaded-board force.
        self._single_leg_loaded_n = 0.60 * self.subject_kg * GRAVITY
        self._single_leg_unloaded_max_n = 100.0

        # Hysteresis: once loading_ok becomes True, re-exiting it requires a
        # stricter (lower) fail condition. Prevents boundary flicker when the
        # load oscillates near the threshold due to sway/noise.
        self._loading_ok_latched = False
        # Entry / exit margins (multiplicative)
        self._load_exit_margin   = 0.85    # must drop 15% below target to flip off
        self._board_exit_margin  = 0.60    # per-board exit at 60 N (from 100 N)
        self._single_exit_margin = 0.85    # loaded-board must stay >= 85% of target

        # Ring buffers sized for the longer (display) window
        max_samples = max(int(self.display_window_s * 200), 20)   # up to 200 Hz
        self._times:  deque = deque(maxlen=max_samples)
        self._totals: deque = deque(maxlen=max_samples)
        self._b1s:    deque = deque(maxlen=max_samples)
        self._b2s:    deque = deque(maxlen=max_samples)

        self._stable_since_ns: Optional[int] = None
        self._wait_start_ns: int = time.monotonic_ns()

    @property
    def target_total_n(self) -> float:
        return self._target_total_n

    def reset_wait_start(self) -> None:
        """Reset the wait-timeout clock (call right before starting to feed)."""
        self._wait_start_ns = time.monotonic_ns()
        self._stable_since_ns = None

    def update(self, frame) -> StabilityState:
        """Consume a DaqFrame and return the current StabilityState."""
        now_ns = time.monotonic_ns()
        # Trim buffer to the LONGER display window; stability std is then
        # computed on a sub-slice representing just the shorter window.
        cutoff_display = now_ns - int(self.display_window_s * 1e9)
        cutoff_stab    = now_ns - int(self.window_s * 1e9)

        self._times.append(now_ns)
        self._totals.append(float(frame.total_n))
        self._b1s.append(float(frame.b1_total_n))
        self._b2s.append(float(frame.b2_total_n))
        while self._times and self._times[0] < cutoff_display:
            self._times.popleft()
            self._totals.popleft()
            self._b1s.popleft()
            self._b2s.popleft()

        if len(self._totals) < 5:
            return self._state(now_ns, "STEP_ON", 0.0, 0.0, 0.0, 0.0)

        times  = np.fromiter(self._times,  dtype=np.int64)
        totals = np.fromiter(self._totals, dtype=np.float64)
        b1s    = np.fromiter(self._b1s,    dtype=np.float64)
        b2s    = np.fromiter(self._b2s,    dtype=np.float64)

        # Clip to zero for drift robustness (physical forces are non-negative)
        totals_c = np.maximum(totals, 0.0)
        b1s_c    = np.maximum(b1s,    0.0)
        b2s_c    = np.maximum(b2s,    0.0)

        # Display values: mean over the longer display window (full buffer)
        mean_total = float(totals_c.mean())
        mean_b1    = float(b1s_c.mean())
        mean_b2    = float(b2s_c.mean())

        # Stability check: std on the last window_s seconds only
        stab_mask = times >= cutoff_stab
        stab_slice = totals[stab_mask]
        std_total = float(stab_slice.std()) if len(stab_slice) > 5 \
                    else float(totals.std())

        stable = std_total < self.stability_n

        # Hysteresis: use stricter conditions to ENTER loading_ok, looser to
        # stay latched. Flicker near boundary is thereby suppressed.
        latched = self._loading_ok_latched
        if self.stance_mode == "two":
            enter = (mean_total >= self._target_total_n
                     and mean_b1 >= self.min_board_n
                     and mean_b2 >= self.min_board_n)
            stay  = (mean_total >= self._target_total_n * self._load_exit_margin
                     and mean_b1 >= self.min_board_n * self._board_exit_margin
                     and mean_b2 >= self.min_board_n * self._board_exit_margin)
            loading_ok = enter if not latched else stay
        elif self.stance_mode == "left":
            enter = (mean_b1 >= self._single_leg_loaded_n
                     and mean_b2 <= self._single_leg_unloaded_max_n)
            stay  = (mean_b1 >= self._single_leg_loaded_n * self._single_exit_margin
                     and mean_b2 <= self._single_leg_unloaded_max_n * 1.20)
            loading_ok = enter if not latched else stay
        else:   # "right"
            enter = (mean_b2 >= self._single_leg_loaded_n
                     and mean_b1 <= self._single_leg_unloaded_max_n)
            stay  = (mean_b2 >= self._single_leg_loaded_n * self._single_exit_margin
                     and mean_b1 <= self._single_leg_unloaded_max_n * 1.20)
            loading_ok = enter if not latched else stay
        self._loading_ok_latched = loading_ok

        if loading_ok and stable:
            if self._stable_since_ns is None:
                self._stable_since_ns = now_ns
            stable_s = (now_ns - self._stable_since_ns) / 1e9
            status = "READY" if stable_s >= self.stability_target_s else "STABILIZING"
        else:
            self._stable_since_ns = None
            # Distinguish "need weight" vs "weight OK but jittery"
            status = "STEP_ON" if not loading_ok else "STABILIZING"

        # Timeout
        wait_elapsed = (now_ns - self._wait_start_ns) / 1e9
        if status != "READY" and wait_elapsed > self.timeout_s:
            status = "TIMEOUT"

        return self._state(now_ns, status, mean_total, mean_b1, mean_b2, std_total)

    def _state(self, now_ns: int, status: str,
               total: float, b1: float, b2: float, std: float) -> StabilityState:
        progress = 0.0
        if self._stable_since_ns is not None:
            progress = (now_ns - self._stable_since_ns) / 1e9
        return StabilityState(
            status=status,
            total_n=total, b1_n=b1, b2_n=b2, std_total_n=std,
            stable_progress_s=progress,
            stable_target_s=self.stability_target_s,
            wait_elapsed_s=(now_ns - self._wait_start_ns) / 1e9,
        )


def draw_wait_overlay(frame: np.ndarray, state: StabilityState,
                      subject_kg: float,
                      stance_mode: str = "two") -> None:
    """Draw the stability-wait UI overlay on a preview frame (in place)."""
    h, w = frame.shape[:2]

    # Status banner varies by stance mode to prompt the user correctly.
    stance_hint = {
        "two":   "BOTH feet",
        "left":  "LEFT foot only",
        "right": "RIGHT foot only",
    }.get(stance_mode, "BOTH feet")
    status_map = {
        "STEP_ON":     (f"STEP ON PLATE - {stance_hint}",  (0,   0, 255)),
        "STABILIZING": ("STABILIZING...",                  (0, 200, 255)),
        "READY":       ("READY - GO!",                     (0, 255,   0)),
        "TIMEOUT":     ("TIMEOUT",                         (100, 100, 255)),
    }
    banner, color = status_map.get(state.status, ("...", (150, 150, 150)))

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.putText(frame, banner, (20, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3, cv2.LINE_AA)

    g = GRAVITY
    # Loading-condition text depends on stance
    if stance_mode == "two":
        target_n = 0.70 * subject_kg * g
        load_line = (f"Total: {state.total_n:6.0f} N   "
                     f"target >= {target_n:.0f} N")
        load_ok = state.total_n >= target_n
        board_line = (f"Board1: {state.b1_n:5.0f} N   "
                      f"Board2: {state.b2_n:5.0f} N   (each >= 100 N)")
        board_ok = state.b1_n >= 100 and state.b2_n >= 100
    elif stance_mode == "left":
        need_b1 = 0.60 * subject_kg * g
        load_line = (f"Board1 (LEFT): {state.b1_n:6.0f} N   "
                     f"target >= {need_b1:.0f} N")
        load_ok = state.b1_n >= need_b1
        board_line = (f"Board2 (RIGHT): {state.b2_n:5.0f} N   (< 100 N)")
        board_ok = state.b2_n < 100
    else:   # right
        need_b2 = 0.60 * subject_kg * g
        load_line = (f"Board2 (RIGHT): {state.b2_n:6.0f} N   "
                     f"target >= {need_b2:.0f} N")
        load_ok = state.b2_n >= need_b2
        board_line = (f"Board1 (LEFT): {state.b1_n:5.0f} N   (< 100 N)")
        board_ok = state.b1_n < 100

    # Bottom status box
    box_top = h - 170
    cv2.rectangle(frame, (0, box_top), (w, h), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, box_top), (w, h), (60, 60, 60), 1)

    col_load = (0, 255, 0) if load_ok else (100, 100, 255)
    cv2.putText(frame, load_line, (20, box_top + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_load, 2)

    col_b = (0, 255, 0) if board_ok else (100, 100, 255)
    cv2.putText(frame, board_line, (20, box_top + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_b, 1)

    col_s = (0, 255, 0) if state.std_total_n < 50 else (100, 100, 255)
    cv2.putText(frame,
                f"Stability: std = {state.std_total_n:4.1f} N   (< 50 N)",
                (20, box_top + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_s, 1)

    # Progress bar
    bar_x = 20
    bar_y = box_top + 110
    bar_w = w - 40
    bar_h = 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), 1)
    frac = min(state.stable_progress_s / max(state.stable_target_s, 1e-6), 1.0)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * frac), bar_y + bar_h), color, -1)
    cv2.putText(frame,
                f"Stable: {state.stable_progress_s:.1f} / "
                f"{state.stable_target_s:.1f} s   "
                f"| waited {state.wait_elapsed_s:.0f} s",
                (bar_x, bar_y + bar_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
