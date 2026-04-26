"""
Unit tests for cop_state classifier and DepartureEventTracker
(Phase U3-3).

Run from project root:
    python tests/test_departure_events.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.cop_state import (
    classify_on_plate, departure_threshold_n, DEPARTURE_THRESHOLD_N,
)
from src.capture.departure_events import DepartureEventTracker


# ────────────────────────────────────────────────────────────────────────────
# T1. Threshold value
# ────────────────────────────────────────────────────────────────────────────

def test_threshold_is_fixed_20n():
    """The departure threshold is 20 N — matches CMJ analyser's
    flight_threshold_n so replay bands and CMJ flight shading mark the
    same instants. Body weight does NOT enter the formula because
    flight is a physical event at GRF ≈ 0 regardless of subject mass."""
    assert DEPARTURE_THRESHOLD_N == 20.0
    assert departure_threshold_n() == 20.0


def test_threshold_independent_of_subject_kg():
    """20 N for everyone — a 20 kg child and a 100 kg adult both leave
    the plate when GRF crosses zero, not when GRF is some fraction of
    their weight."""
    assert departure_threshold_n(0.0)    == 20.0
    assert departure_threshold_n(None)   == 20.0
    assert departure_threshold_n(20.0)   == 20.0
    assert departure_threshold_n(80.0)   == 20.0
    assert departure_threshold_n(120.0)  == 20.0


def test_threshold_ignores_legacy_kwargs():
    """The legacy ``bw_ratio`` parameter is accepted (for backward
    compat with callers wired up before the simplification) but
    ignored — output is always 20 N."""
    assert departure_threshold_n(80.0, bw_ratio=0.10) == 20.0
    assert departure_threshold_n(80.0, bw_ratio=0.30) == 20.0


def test_threshold_matches_cmj_analyser_default():
    """The whole point of fixing this at 20 N is alignment with the
    CMJ analyser's default takeoff/landing threshold."""
    from src.analysis.cmj import analyze_cmj
    import inspect
    sig = inspect.signature(analyze_cmj)
    cmj_default = sig.parameters["flight_threshold_n"].default
    assert DEPARTURE_THRESHOLD_N == cmj_default


# ────────────────────────────────────────────────────────────────────────────
# T2. classify_on_plate
# ────────────────────────────────────────────────────────────────────────────

def test_classify_on_plate_above_returns_1():
    assert classify_on_plate(total_n=200.0, threshold_n=156.9) == 1


def test_classify_on_plate_below_returns_0():
    assert classify_on_plate(total_n=10.0, threshold_n=156.9) == 0


def test_classify_on_plate_exact_boundary_returns_1():
    """Boundary is inclusive on the on-plate side (≥)."""
    assert classify_on_plate(total_n=156.9, threshold_n=156.9) == 1


def test_classify_on_plate_returns_int():
    """Must return int (0/1), not bool, for clean CSV writing."""
    assert isinstance(classify_on_plate(0.0, 50.0), int)


# ────────────────────────────────────────────────────────────────────────────
# T3. DepartureEventTracker — basic flow
# ────────────────────────────────────────────────────────────────────────────

def test_tracker_no_events_when_always_on_plate():
    tr = DepartureEventTracker(min_duration_s=0.05)
    for i in range(100):
        tr.update(on_plate=1, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    events = tr.finalize()
    assert events == []
    s = tr.summary()
    assert s["n_events"] == 0
    assert s["total_off_plate_s"] == 0.0


def test_tracker_records_one_long_departure():
    """Subject is on plate, then leaves for 500 ms, then returns."""
    tr = DepartureEventTracker(min_duration_s=0.05)
    # 100 ms on-plate
    for i in range(10):
        tr.update(on_plate=1, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    # 500 ms off-plate (50 samples at 100 Hz)
    for i in range(10, 60):
        tr.update(on_plate=0, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    # 100 ms back on-plate
    for i in range(60, 70):
        tr.update(on_plate=1, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    events = tr.finalize()
    assert len(events) == 1
    ev = events[0]
    assert abs(ev["t_start_s"] - 0.10) < 1e-3
    assert abs(ev["t_end_s"]   - 0.60) < 1e-3
    assert abs(ev["duration_s"] - 0.50) < 1e-3
    assert ev["n_samples"] == 50
    assert ev["trial_idx"] == 0


def test_tracker_drops_short_flicker():
    """3-sample (30 ms) departure < min_duration_s (50 ms) → not an event."""
    tr = DepartureEventTracker(min_duration_s=0.05)
    for i in range(10):
        tr.update(on_plate=1, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    # 3 samples off-plate
    for i in range(10, 13):
        tr.update(on_plate=0, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    # back on
    for i in range(13, 20):
        tr.update(on_plate=1, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    assert tr.finalize() == []


def test_tracker_keeps_event_at_minimum_duration():
    """Exactly 50 ms (5 samples) still qualifies."""
    tr = DepartureEventTracker(min_duration_s=0.05)
    for i in range(10):
        tr.update(on_plate=1, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    for i in range(10, 15):
        tr.update(on_plate=0, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    for i in range(15, 20):
        tr.update(on_plate=1, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    events = tr.finalize()
    # Boundary is duration >= min_duration_s — 0.05 == 0.05 → keep
    assert len(events) == 1
    assert abs(events[0]["duration_s"] - 0.05) < 1e-9


def test_tracker_multiple_events_indexed_in_order():
    """Three separate departures get trial_idx 0, 1, 2 in time order."""
    tr = DepartureEventTracker(min_duration_s=0.05)
    schedule = [
        (0.0, 1, 10),    # 100ms on
        (0.10, 0, 10),   # 100ms off  → event 0
        (0.20, 1, 10),
        (0.30, 0, 8),    # 80ms off   → event 1
        (0.38, 1, 10),
        (0.48, 0, 6),    # 60ms off   → event 2
        (0.54, 1, 10),
    ]
    t = 0.0
    for t_start, state, n in schedule:
        for k in range(n):
            tr.update(on_plate=state,
                      t_s=t, t_wall=1000.0 + t)
            t += 0.01
    events = tr.finalize()
    assert len(events) == 3
    assert [ev["trial_idx"] for ev in events] == [0, 1, 2]
    # Durations should match the off-plate windows above
    assert abs(events[0]["duration_s"] - 0.10) < 1e-3
    assert abs(events[1]["duration_s"] - 0.08) < 1e-3
    assert abs(events[2]["duration_s"] - 0.06) < 1e-3


def test_tracker_closes_open_interval_at_finalize():
    """Recording ends while subject is still off-plate — event closes
    at the last sample time."""
    tr = DepartureEventTracker(min_duration_s=0.05)
    for i in range(5):
        tr.update(on_plate=1, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    for i in range(5, 20):
        tr.update(on_plate=0, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    events = tr.finalize()
    assert len(events) == 1
    # Last off-plate sample was at i=19, t=0.19
    assert abs(events[0]["t_end_s"] - 0.19) < 1e-9


def test_tracker_finalize_idempotent():
    """Calling finalize() multiple times returns same events without
    duplicating (no second close on the same interval)."""
    tr = DepartureEventTracker(min_duration_s=0.05)
    for i in range(5):
        tr.update(on_plate=1, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    for i in range(5, 20):
        tr.update(on_plate=0, t_s=i * 0.01, t_wall=1000.0 + i * 0.01)
    e1 = tr.finalize()
    e2 = tr.finalize()
    assert e1 == e2


# ────────────────────────────────────────────────────────────────────────────
# T4. summary
# ────────────────────────────────────────────────────────────────────────────

def test_summary_aggregates_stats():
    tr = DepartureEventTracker(min_duration_s=0.05)
    # Two events: 200ms and 600ms
    for i in range(10): tr.update(1, i * 0.01, 1000.0 + i * 0.01)
    for i in range(10, 30): tr.update(0, i * 0.01, 1000.0 + i * 0.01)  # 200ms
    for i in range(30, 40): tr.update(1, i * 0.01, 1000.0 + i * 0.01)
    for i in range(40, 100): tr.update(0, i * 0.01, 1000.0 + i * 0.01)  # 600ms
    for i in range(100, 110): tr.update(1, i * 0.01, 1000.0 + i * 0.01)
    tr.finalize()
    s = tr.summary()
    assert s["n_events"] == 2
    assert abs(s["total_off_plate_s"] - 0.80) < 1e-3
    assert abs(s["longest_off_s"] - 0.60) < 1e-3
    assert abs(s["first_departure_t_s"] - 0.10) < 1e-3
    # Last event closes at the first on-plate sample after the off-stretch,
    # which is t=1.00 (i=100), not the last off-plate sample's t=0.99.
    assert abs(s["last_return_t_s"] - 1.00) < 1e-3


def test_summary_no_events_returns_zeros():
    tr = DepartureEventTracker()
    tr.update(1, 0.0, 1000.0)
    tr.update(1, 0.01, 1000.01)
    tr.finalize()
    s = tr.summary()
    assert s["n_events"] == 0
    assert s["total_off_plate_s"] == 0.0
    assert s["first_departure_t_s"] is None


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
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"=== {failed}/{len(fns)} tests failed ===")
        sys.exit(1)
    else:
        print(f"=== All {len(fns)} tests passed ===")
