"""
Real-time CoP noise filter (Phase U2-4).

The raw ``DaqFrame.cop_world_mm()`` already returns NaN when the total
vertical force is below 5 N. That handles "no person on the plate" but
NOT the trickier mid-recording case where a CMJ flight phase or sudden
unloading produces brief force spikes (10-30 N) just above the 5 N
threshold — these spikes pass through and produce **wildly varying CoP
values** (since CoP at near-zero force is mathematically undefined).

This filter applies two complementary gates:

  1. **Force-context gate** — reject the sample if the total force
     falls below ``BW_GATE_RATIO × subject_kg × g`` (default 20 % BW).
     Catches the entire jump-flight phase reliably because subject
     mass is known and the gate scales with it.

  2. **Velocity outlier gate** — reject the sample if the implied CoP
     velocity from the last accepted sample exceeds
     ``MAX_VELOCITY_MM_S`` (default 3 000 mm/s = 3 m/s). True CoP
     during landing impact moves up to ~1 m/s, so 3 m/s is a generous
     ceiling that still rejects spikes (which jump 100s of mm in a
     single 10 ms sample → effective 10-50 m/s).

When a sample is rejected, the caller can either skip plotting (so the
trail "freezes" at the last valid position) or display a faded marker
at that last position — see ``COPTrajectory.push_filtered``.
"""
from __future__ import annotations

from typing import Optional, Tuple

GRAVITY = 9.80665


class CoPFilter:
    """Stateful CoP validity filter — call ``accept()`` per DaqFrame.

    Reset between sessions via ``set_subject_kg()`` or ``reset()``.
    """

    # Defaults — tuned for adult subjects on the FITWIN twin-plate setup.
    BW_GATE_RATIO     = 0.20    # 20 % bodyweight
    MAX_VELOCITY_MM_S = 3000.0  # 3 m/s ceiling on legitimate CoP travel
    SAMPLE_RATE_HZ    = 100.0   # used for per-sample velocity threshold

    def __init__(self, subject_kg: float = 0.0):
        self._subject_kg = max(0.0, float(subject_kg))
        self._force_threshold_n = self._compute_force_threshold(
            self._subject_kg)
        # Per-sample step ceiling: velocity_max / sample_rate. e.g.
        # 3 m/s / 100 Hz = 30 mm per sample.
        self._max_step_mm = self.MAX_VELOCITY_MM_S / self.SAMPLE_RATE_HZ
        self._last_valid: Optional[Tuple[float, float]] = None

    # ── public ──────────────────────────────────────────────────────────
    def set_subject_kg(self, kg: float) -> None:
        self._subject_kg = max(0.0, float(kg))
        self._force_threshold_n = self._compute_force_threshold(
            self._subject_kg)
        # Velocity reset: don't carry an old reference into a new session.
        self._last_valid = None

    def reset(self) -> None:
        """Forget the last-valid sample. Call when the session boundary
        crosses (between tests, after Stop, on subject change)."""
        self._last_valid = None

    @property
    def force_threshold_n(self) -> float:
        return self._force_threshold_n

    @property
    def last_valid(self) -> Optional[Tuple[float, float]]:
        """The most recent (cx_mm, cy_mm) that passed both gates, or
        None if no sample has been accepted since the last reset."""
        return self._last_valid

    def accept(self, total_n: float, cx: float, cy: float) -> bool:
        """Decide whether this CoP sample should be displayed.

        Returns True if the sample is valid (and updates the
        last-valid reference); False if it should be suppressed.
        Callers may still keep ``self._last_valid`` for the faded
        marker rendering.
        """
        # Gate 1: force context
        if total_n < self._force_threshold_n:
            return False
        # NaN inputs are always rejected (cop_world_mm returns NaN for
        # the < 5 N case already).
        try:
            if cx != cx or cy != cy:        # NaN check w/o numpy import
                return False
        except Exception:
            return False

        # Gate 2: velocity outlier (only after we have a reference)
        if self._last_valid is not None:
            dx = cx - self._last_valid[0]
            dy = cy - self._last_valid[1]
            if (dx * dx + dy * dy) > (self._max_step_mm * self._max_step_mm):
                # Spike — drop the reference so the next valid sample
                # re-anchors. Without this reset, a real CoP
                # discontinuity (e.g., landing impact transferring load
                # ~50 mm in one sample) keeps the velocity gate stuck
                # comparing every subsequent good sample to the
                # pre-jump position, and the entire post-impact trail
                # disappears. See filter_offline() for the same logic.
                self._last_valid = None
                return False

        self._last_valid = (float(cx), float(cy))
        return True

    # ── helpers ─────────────────────────────────────────────────────────
    @classmethod
    def _compute_force_threshold(cls, subject_kg: float) -> float:
        """Convert subject mass (kg) to a Newton threshold using the
        configured BW ratio. Falls back to a 50 N absolute floor when
        subject_kg is unknown so a default-config (no subject loaded)
        session still gets some flight-phase rejection."""
        if subject_kg <= 0:
            return 50.0
        return float(subject_kg) * GRAVITY * cls.BW_GATE_RATIO


def filter_offline(total_n, cop_x, cop_y,
                   subject_kg: float = 0.0,
                   sample_rate_hz: float = 100.0):
    """Vectorised offline equivalent of ``CoPFilter.accept()``.

    Used by the ReplayPanel to clean up CoP arrays loaded from a
    saved ``forces.csv``. The recorder writes raw CoP (with only a
    5 N gate from ``DaqFrame.cop_world_mm``), so flight-phase spikes
    survive in the saved data and need this same gate to be applied
    on read.

    Returns the masked ``(cop_x, cop_y)`` arrays — sample positions
    that fail the force gate or velocity gate become NaN. The shape
    is preserved so callers can still index by sample.

    Iteration is sequential (not pure-vectorised) because the
    velocity gate is stateful — it only knows what to compare against
    after seeing the previous accepted sample. ~6000 samples per
    typical session run in <5 ms.
    """
    import numpy as np
    cop_x = np.asarray(cop_x, dtype=np.float64).copy()
    cop_y = np.asarray(cop_y, dtype=np.float64).copy()
    total_n = np.asarray(total_n, dtype=np.float64)
    n = len(cop_x)
    if n == 0:
        return cop_x, cop_y

    threshold_n = (subject_kg * GRAVITY * CoPFilter.BW_GATE_RATIO
                   if subject_kg > 0 else 50.0)
    max_step = CoPFilter.MAX_VELOCITY_MM_S / sample_rate_hz

    last_x: float = float("nan")
    last_y: float = float("nan")

    for i in range(n):
        cx = cop_x[i]
        cy = cop_y[i]
        # Already-NaN inputs (recorder gate < 5N) stay NaN, and reset
        # the velocity reference so the next valid sample after a gap
        # is accepted as a re-anchor (not compared to a stale last).
        if cx != cx or cy != cy:
            last_x = float("nan"); last_y = float("nan")
            continue
        # Gate 1: force context — also reset last on failure.
        if total_n[i] < threshold_n:
            cop_x[i] = float("nan")
            cop_y[i] = float("nan")
            last_x = float("nan"); last_y = float("nan")
            continue
        # Gate 2: velocity outlier — only consulted when we have a
        # recent reference. After any rejection above, last_x is NaN
        # and we accept the first sample to re-anchor.
        if last_x == last_x:    # last_x is not NaN
            dx = cx - last_x
            dy = cy - last_y
            if dx * dx + dy * dy > max_step * max_step:
                cop_x[i] = float("nan")
                cop_y[i] = float("nan")
                # Re-anchor on next valid sample. Without this reset,
                # a single discontinuity (e.g., landing impact, where
                # CoP legitimately shifts ~50mm in one sample as load
                # transfers) keeps the reference stuck at the pre-jump
                # position and NaNs every subsequent sample because the
                # cumulative distance never falls back under 30 mm.
                # Side effect: a single isolated noise spike is dropped,
                # but the *second* sample of a hypothetical multi-spike
                # cluster would be accepted as the new anchor. In real
                # data, isolated single-sample spikes dominate (driven by
                # near-zero force around flight-phase boundaries that
                # already get caught by Gate 1), so this is the right
                # trade-off.
                last_x = float("nan"); last_y = float("nan")
                continue
        last_x, last_y = cx, cy

    return cop_x, cop_y
