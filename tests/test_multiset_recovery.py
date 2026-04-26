"""
Unit tests for src/analysis/multiset_recovery.py (Phase V2).

Run from project root:
    python tests/test_multiset_recovery.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.multiset_recovery import (
    fatigue_index, performance_decrement_score,
    grade_fi, grade_pds, fiber_tendency,
    compute_recovery_metrics, SetPerformance, RecoveryMetrics,
)


# ────────────────────────────────────────────────────────────────────────────
# fatigue_index
# ────────────────────────────────────────────────────────────────────────────
def test_fi_basic_decline():
    """500W → 350W: FI = (500-350)/500 × 100 = 30 %."""
    assert abs(fatigue_index([500.0, 420.0, 350.0]) - 30.0) < 1e-6


def test_fi_no_decline():
    """Identical sets → FI = 0."""
    assert fatigue_index([400.0, 400.0, 400.0]) == 0.0


def test_fi_improvement_negative():
    """Last > first → negative FI (means subject got stronger)."""
    fi = fatigue_index([400.0, 410.0, 430.0])
    assert fi == -7.5         # (400-430)/400 × 100


def test_fi_only_first_and_last_used():
    """FI ignores middle sets — based on first vs last only."""
    fi_a = fatigue_index([500.0, 100.0, 350.0])
    fi_b = fatigue_index([500.0, 480.0, 350.0])
    assert fi_a == fi_b == 30.0


def test_fi_too_few_sets_returns_nan():
    assert math.isnan(fatigue_index([]))
    assert math.isnan(fatigue_index([500.0]))


def test_fi_zero_first_returns_nan():
    assert math.isnan(fatigue_index([0.0, 200.0, 200.0]))


# ────────────────────────────────────────────────────────────────────────────
# performance_decrement_score
# ────────────────────────────────────────────────────────────────────────────
def test_pds_no_decline_zero():
    """PDS = 0 when every set matches the first."""
    assert performance_decrement_score([400.0, 400.0, 400.0]) == 0.0


def test_pds_basic_decline():
    """500/400/300: ideal = 500×3 = 1500, actual = 1200,
       PDS = (1 − 1200/1500) × 100 = 20 %."""
    pds = performance_decrement_score([500.0, 400.0, 300.0])
    assert abs(pds - 20.0) < 1e-6


def test_pds_uses_all_sets():
    """Different middle values must yield different PDS, unlike FI."""
    a = performance_decrement_score([500.0, 100.0, 350.0])  # collapses early
    b = performance_decrement_score([500.0, 480.0, 350.0])  # gradual
    # Both have FI = 30 % but PDS should differ.
    assert a != b
    # Earlier collapse → larger PDS.
    assert a > b


def test_pds_too_few_sets_returns_nan():
    assert math.isnan(performance_decrement_score([]))
    assert math.isnan(performance_decrement_score([300.0]))


def test_pds_negative_when_improvement():
    """If subsequent sets exceed the first, PDS goes negative."""
    pds = performance_decrement_score([400.0, 420.0, 440.0])
    assert pds < 0


# ────────────────────────────────────────────────────────────────────────────
# Grade lookups
# ────────────────────────────────────────────────────────────────────────────
def test_grade_fi_thresholds():
    """Plan boundaries: <15=1, <25=2, <40=3, <60=4, ≥60=5."""
    assert grade_fi(0.0)[0]  == 1
    assert grade_fi(14.99)[0] == 1
    assert grade_fi(15.0)[0] == 2
    assert grade_fi(24.99)[0] == 2
    assert grade_fi(25.0)[0] == 3
    assert grade_fi(39.99)[0] == 3
    assert grade_fi(40.0)[0] == 4
    assert grade_fi(59.99)[0] == 4
    assert grade_fi(60.0)[0] == 5
    assert grade_fi(85.0)[0] == 5


def test_grade_fi_negative_clipped_to_zero():
    """Improvement (negative FI) → clipped to 0 → grade 1 (Elite)."""
    assert grade_fi(-10.0)[0] == 1


def test_grade_fi_korean_labels():
    """Each grade returns its Korean label."""
    assert grade_fi(5.0)[1]   == "엘리트"
    assert grade_fi(20.0)[1]  == "좋음"
    assert grade_fi(30.0)[1]  == "보통"
    assert grade_fi(50.0)[1]  == "나쁨"
    assert grade_fi(70.0)[1]  == "위험"


def test_grade_pds_thresholds():
    """Plan boundaries: <8=1, <15=2, <25=3, <40=4, ≥40=5."""
    assert grade_pds(0.0)[0]  == 1
    assert grade_pds(7.99)[0] == 1
    assert grade_pds(8.0)[0]  == 2
    assert grade_pds(14.99)[0] == 2
    assert grade_pds(15.0)[0] == 3
    assert grade_pds(24.99)[0] == 3
    assert grade_pds(25.0)[0] == 4
    assert grade_pds(39.99)[0] == 4
    assert grade_pds(40.0)[0] == 5


def test_grade_pds_korean_labels():
    assert grade_pds(2.0)[1]  == "엘리트"
    assert grade_pds(10.0)[1] == "좋음"
    assert grade_pds(20.0)[1] == "보통"
    assert grade_pds(30.0)[1] == "나쁨"
    assert grade_pds(50.0)[1] == "위험"


# ────────────────────────────────────────────────────────────────────────────
# fiber_tendency
# ────────────────────────────────────────────────────────────────────────────
def test_tendency_low_pds_means_endurance():
    """PDS = 5 → tendency negative (endurance side)."""
    t, lbl = fiber_tendency(5.0, set1_value=400.0)
    assert t < 0
    assert "지구력" in lbl


def test_tendency_high_pds_means_power():
    """PDS = 50, set1 high → tendency positive (power side)."""
    t, lbl = fiber_tendency(50.0, set1_value=600.0)
    assert t > 0
    assert "파워" in lbl


def test_tendency_clipped_to_unit_range():
    """Tendency must always be in [-1, +1]."""
    for pds in (-100, 0, 25, 50, 100, 200):
        t, _ = fiber_tendency(pds, set1_value=500.0)
        assert -1.0 <= t <= 1.0


def test_tendency_25pct_is_neutral_zero():
    """At PDS = 25 % the tendency is 0 (neutral mid-point)."""
    t, lbl = fiber_tendency(25.0, set1_value=400.0)
    assert abs(t) < 1e-9
    assert "균형" in lbl


def test_tendency_population_ref_dampens_high_pds_low_power():
    """High PDS but low set1 power vs population_ref → tendency
    weighted DOWN, not raised to full power-side."""
    t_unweighted, _ = fiber_tendency(50.0, set1_value=200.0)
    t_weighted, _   = fiber_tendency(50.0, set1_value=200.0,
                                      population_ref=600.0)
    # Population reference weighting should reduce the magnitude of
    # the positive tendency.
    assert t_weighted < t_unweighted


# ────────────────────────────────────────────────────────────────────────────
# compute_recovery_metrics — end-to-end pipeline
# ────────────────────────────────────────────────────────────────────────────
def _perf(idx, warmup, mean_p):
    return SetPerformance(set_idx=idx, warmup=warmup, n_reps=10,
                          mean_power_w=mean_p)


def test_compute_recovery_excludes_warmup():
    """Warmup set excluded from FI/PDS calculation."""
    set_perfs = [
        _perf(0, True,  100.0),    # warmup — excluded
        _perf(1, False, 500.0),
        _perf(2, False, 400.0),
        _perf(3, False, 350.0),
    ]
    rm = compute_recovery_metrics(set_perfs)
    assert rm.n_working_sets == 3
    assert rm.set_indices == [1, 2, 3]
    assert rm.set_values == [500.0, 400.0, 350.0]
    # FI = (500-350)/500 × 100 = 30
    assert abs(rm.fi_pct - 30.0) < 0.01
    # PDS based on the 3 working sets (rm rounds to 2 decimals)
    expected_pds = (1 - (500 + 400 + 350) / (500 * 3)) * 100
    assert abs(rm.pds_pct - expected_pds) < 0.01


def test_compute_recovery_skips_when_one_working_set():
    """With only 1 working set, FI/PDS need ≥2 → skipped."""
    set_perfs = [_perf(0, True, 100.0), _perf(1, False, 500.0)]
    rm = compute_recovery_metrics(set_perfs)
    assert rm.skipped_reason == "fewer than 2 working sets"
    assert math.isnan(rm.fi_pct)


def test_compute_recovery_skips_when_zero_powers():
    """All working sets have zero power → skipped."""
    set_perfs = [_perf(0, False, 0.0), _perf(1, False, 0.0)]
    rm = compute_recovery_metrics(set_perfs)
    assert rm.skipped_reason is not None
    assert "zero" in rm.skipped_reason.lower()


def test_compute_recovery_grades_assigned():
    """End-to-end: 5%/2.5% drop → both grades = 1 (엘리트)."""
    set_perfs = [_perf(i, False, p)
                 for i, p in enumerate([400.0, 390.0, 385.0])]
    rm = compute_recovery_metrics(set_perfs)
    assert rm.fi_grade == 1
    assert rm.pds_grade == 1
    assert rm.fi_label == "엘리트"


def test_compute_recovery_can_use_alternate_variable():
    """Pipeline accepts arbitrary attribute name as the primary variable."""
    set_perfs = [
        SetPerformance(set_idx=0, warmup=False, n_reps=10, mean_power_w=999.0),
        SetPerformance(set_idx=1, warmup=False, n_reps=8,  mean_power_w=999.0),
    ]
    rm = compute_recovery_metrics(set_perfs, variable="n_reps")
    assert rm.set_values == [10.0, 8.0]
    assert abs(rm.fi_pct - 20.0) < 1e-6


def test_compute_recovery_to_dict_jsonable():
    """to_dict produces a JSON-serialisable nested dict."""
    import json
    set_perfs = [_perf(0, False, 500.0), _perf(1, False, 400.0)]
    rm = compute_recovery_metrics(set_perfs)
    d = rm.to_dict()
    json.dumps(d, default=str)   # round-trip


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
