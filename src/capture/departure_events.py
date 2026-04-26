"""
Streaming state machine that turns per-sample on/off-plate booleans
into discrete departure events.

A "departure event" is a contiguous stretch of off-plate (``on_plate=0``)
samples that lasts at least ``min_duration_s`` (default 50 ms = 5
samples at 100 Hz). Shorter excursions are treated as noise and
ignored — this filters out single-sample spikes around the threshold
boundary that aren't physically meaningful departures.

Usage in the recorder hot path:

    tracker = DepartureEventTracker(min_duration_s=0.05)
    for fr in daq_stream:
        on_plate = classify_on_plate(fr.total_n, threshold)
        tracker.update(on_plate, t_rel_s=fr.t_wall - rec_start)
    events = tracker.finalize()      # list[dict]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _OpenInterval:
    t_start_s: float
    t_start_wall: float
    n_samples: int = 0


@dataclass
class DepartureEvent:
    trial_idx:   int
    t_start_s:   float       # seconds from record_start
    t_end_s:     float
    t_start_wall: float      # absolute wall clock
    t_end_wall:  float
    duration_s:  float
    n_samples:   int

    def to_row(self) -> dict:
        return {
            "trial_idx":     self.trial_idx,
            "t_start_s":     round(self.t_start_s, 4),
            "t_end_s":       round(self.t_end_s, 4),
            "t_start_wall":  round(self.t_start_wall, 6),
            "t_end_wall":    round(self.t_end_wall, 6),
            "duration_s":    round(self.duration_s, 4),
            "n_samples":     self.n_samples,
        }


class DepartureEventTracker:
    """Streaming state machine for on-plate / off-plate transitions.

    Calling ``update(on_plate, ...)`` per sample is O(1). Only
    ``finalize()`` allocates — it returns the recorded events as a
    list of dicts ready for CSV writing.
    """

    def __init__(self, min_duration_s: float = 0.05):
        self._min_dur = float(min_duration_s)
        self._events: list[DepartureEvent] = []
        self._open: Optional[_OpenInterval] = None
        # Last sample we've seen — needed so ``finalize()`` can close
        # an interval that's still open at end-of-recording.
        self._last_t_s: Optional[float] = None
        self._last_t_wall: Optional[float] = None

    # ── streaming ────────────────────────────────────────────────────
    def update(self, on_plate: int, t_s: float, t_wall: float) -> None:
        """Feed one sample into the machine.

        Args:
            on_plate: 0 = off plate, 1 = on plate (binary)
            t_s:      seconds since record_start
            t_wall:   absolute wall-clock seconds (for csv reference)
        """
        self._last_t_s = t_s
        self._last_t_wall = t_wall

        if on_plate:
            # Currently on plate — close any open interval that's long
            # enough to qualify as an event.
            if self._open is not None:
                self._close_interval(end_t_s=t_s, end_t_wall=t_wall)
        else:
            # Off plate — open a new interval if none active.
            if self._open is None:
                self._open = _OpenInterval(
                    t_start_s=t_s, t_start_wall=t_wall, n_samples=1)
            else:
                self._open.n_samples += 1

    def finalize(self) -> list[dict]:
        """Close any still-open interval and return all qualifying
        events as a list of dicts. Safe to call multiple times — only
        the first call has side effects."""
        if self._open is not None and self._last_t_s is not None:
            self._close_interval(
                end_t_s=self._last_t_s, end_t_wall=self._last_t_wall)
        return [ev.to_row() for ev in self._events]

    # ── helpers ──────────────────────────────────────────────────────
    def _close_interval(self, end_t_s: float, end_t_wall: float) -> None:
        op = self._open
        assert op is not None
        duration = end_t_s - op.t_start_s
        # Tolerate sub-microsecond float rounding so a stretch of exactly
        # ``min_duration_s`` worth of samples is kept (e.g. 0.15 − 0.10 in
        # IEEE-754 evaluates to 0.0499999…, which would fail a strict ≥).
        if duration >= self._min_dur - 1e-9:
            self._events.append(DepartureEvent(
                trial_idx=len(self._events),
                t_start_s=op.t_start_s,
                t_end_s=end_t_s,
                t_start_wall=op.t_start_wall,
                t_end_wall=end_t_wall,
                duration_s=duration,
                n_samples=op.n_samples,
            ))
        self._open = None

    # ── summary ──────────────────────────────────────────────────────
    def summary(self) -> dict:
        """Return aggregate departure statistics for session.json.

        Call AFTER ``finalize()``. When no events occurred returns
        ``n_events=0`` with all timing fields set to None so JSON
        readers can distinguish "no departures" from "didn't track".
        """
        if not self._events:
            return {
                "n_events":            0,
                "total_off_plate_s":   0.0,
                "longest_off_s":       0.0,
                "first_departure_t_s": None,
                "last_return_t_s":     None,
            }
        total = sum(ev.duration_s for ev in self._events)
        longest = max(ev.duration_s for ev in self._events)
        return {
            "n_events":            len(self._events),
            "total_off_plate_s":   round(total, 4),
            "longest_off_s":       round(longest, 4),
            "first_departure_t_s": round(self._events[0].t_start_s, 4),
            "last_return_t_s":     round(self._events[-1].t_end_s, 4),
        }
