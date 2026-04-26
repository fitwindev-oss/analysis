"""
Unit tests for src/analysis/one_rm.py (Phase V1-C).

Run from project root:
    python tests/test_one_rm.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.one_rm import (
    epley, brzycki, lombardi,
    estimate_1rm, estimate_1rm_from_sets, reliability_band,
)


# ────────────────────────────────────────────────────────────────────────────
# Single-formula correctness
# ────────────────────────────────────────────────────────────────────────────
def test_epley_known_value():
    """100kg × 10 reps → Epley 1RM = 100 × (1 + 10/30) = 133.33"""
    assert abs(epley(100.0, 10) - 100 * (1 + 10 / 30)) < 1e-9


def test_brzycki_known_value():
    """100kg × 10 reps → Brzycki 1RM = 100 × 36 / (37 - 10) = 133.33"""
    assert abs(brzycki(100.0, 10) - 100 * 36 / (37 - 10)) < 1e-9


def test_lombardi_known_value():
    """100kg × 10 reps → Lombardi 1RM = 100 × 10^0.10 ≈ 125.89"""
    assert abs(lombardi(100.0, 10) - 100 * (10 ** 0.10)) < 1e-9


def test_brzycki_diverges_at_37_reps():
    """Brzycki has division by zero at reps=37; we return NaN."""
    assert math.isnan(brzycki(100.0, 37))
    assert math.isnan(brzycki(100.0, 50))


def test_one_rep_returns_load_for_all_formulas():
    """At reps=1, all formulas should return ~the load (it IS the 1RM)."""
    L = 80.0
    # Epley: L × (1 + 1/30) = 1.033 × L
    assert abs(epley(L, 1) - L * 1.033333) < 1e-3
    # Brzycki: L × 36/36 = L exactly
    assert abs(brzycki(L, 1) - L) < 1e-9
    # Lombardi: L × 1^0.10 = L exactly
    assert abs(lombardi(L, 1) - L) < 1e-9


def test_zero_or_negative_reps_raises():
    """All formulas should reject reps < 1."""
    for fn in (epley, brzycki, lombardi):
        try:
            fn(100.0, 0)
            assert False, f"{fn.__name__} should reject reps=0"
        except ValueError:
            pass
        try:
            fn(100.0, -3)
            assert False, f"{fn.__name__} should reject reps=-3"
        except ValueError:
            pass


# ────────────────────────────────────────────────────────────────────────────
# estimate_1rm dispatch + ensemble
# ────────────────────────────────────────────────────────────────────────────
def test_estimate_1rm_method_dispatch():
    """Each named method returns the corresponding formula's value."""
    r = estimate_1rm(100.0, 10, method="epley")
    assert r["one_rm_kg"] == round(epley(100, 10), 2)

    r = estimate_1rm(100.0, 10, method="brzycki")
    assert r["one_rm_kg"] == round(brzycki(100, 10), 2)

    r = estimate_1rm(100.0, 10, method="lombardi")
    assert r["one_rm_kg"] == round(lombardi(100, 10), 2)


def test_estimate_1rm_ensemble_is_average():
    """Ensemble = arithmetic mean of all three formulas."""
    r = estimate_1rm(100.0, 10, method="ensemble")
    expected = (epley(100, 10) + brzycki(100, 10) + lombardi(100, 10)) / 3
    assert abs(r["one_rm_kg"] - round(expected, 2)) < 1e-2


def test_estimate_1rm_ensemble_skips_nan_brzycki():
    """At reps=37, Brzycki is NaN — ensemble should average only the
    other two formulas, not propagate NaN."""
    r = estimate_1rm(100.0, 37, method="ensemble")
    assert not math.isnan(r["one_rm_kg"])
    expected = (epley(100, 37) + lombardi(100, 37)) / 2
    assert abs(r["one_rm_kg"] - round(expected, 2)) < 1e-2


def test_estimate_1rm_unknown_method_raises():
    try:
        estimate_1rm(100.0, 10, method="badname")
        assert False, "should reject unknown method"
    except ValueError as e:
        assert "unknown method" in str(e)


# ────────────────────────────────────────────────────────────────────────────
# Reliability bands
# ────────────────────────────────────────────────────────────────────────────
def test_reliability_band_boundaries():
    assert reliability_band(1) == "excellent"
    assert reliability_band(5) == "excellent"
    assert reliability_band(6) == "high"
    assert reliability_band(10) == "high"
    assert reliability_band(11) == "medium"
    assert reliability_band(12) == "medium"
    assert reliability_band(13) == "low"
    assert reliability_band(20) == "low"
    assert reliability_band(21) == "unreliable"


def test_estimate_1rm_includes_reliability():
    """Output dict carries reliability matching reliability_band()."""
    assert estimate_1rm(100, 5)["reliability"] == "excellent"
    assert estimate_1rm(100, 10)["reliability"] == "high"
    assert estimate_1rm(100, 12)["reliability"] == "medium"
    assert estimate_1rm(100, 18)["reliability"] == "low"
    assert estimate_1rm(100, 25)["reliability"] == "unreliable"


# ────────────────────────────────────────────────────────────────────────────
# Multi-set: best-of selection
# ────────────────────────────────────────────────────────────────────────────
def test_from_sets_picks_best_estimate_set():
    """With 3 sets at decreasing reps (fatigue), the best 1RM is at
    set 0 (heaviest perceived effort possible at given reps × load).

    Synthetic: 100 kg × {10, 8, 6} reps. Brzycki:
      set 0: 100 × 36/27 = 133.33
      set 1: 100 × 36/29 = 124.14
      set 2: 100 × 36/31 = 116.13
    Best = set 0.
    """
    sets = [
        {"load_kg": 100, "reps": 10},
        {"load_kg": 100, "reps": 8},
        {"load_kg": 100, "reps": 6},
    ]
    r = estimate_1rm_from_sets(sets, method="brzycki")
    assert r["chosen_set_idx"] == 0
    assert r["n_working_sets"] == 3
    assert abs(r["one_rm_kg"] - round(brzycki(100, 10), 2)) < 0.01


def test_from_sets_excludes_warmup_by_default():
    """Sets flagged warmup=True must not contribute."""
    sets = [
        {"load_kg": 60, "reps": 12, "warmup": True},   # warmup — excluded
        {"load_kg": 100, "reps": 10},
        {"load_kg": 100, "reps": 8},
    ]
    r = estimate_1rm_from_sets(sets, method="epley")
    assert r["n_working_sets"] == 2
    # chosen_set_idx is 1 (the heavier of the two working sets at higher reps)
    assert r["chosen_set_idx"] == 1


def test_from_sets_includes_warmup_when_requested():
    sets = [
        {"load_kg": 60, "reps": 12, "warmup": True},
        {"load_kg": 100, "reps": 10},
    ]
    r = estimate_1rm_from_sets(sets, method="epley", include_warmup=True)
    assert r["n_working_sets"] == 2


def test_from_sets_empty_returns_nan():
    r = estimate_1rm_from_sets([])
    assert math.isnan(r["one_rm_kg"])
    assert r["chosen_set_idx"] is None
    assert r["n_working_sets"] == 0


def test_from_sets_only_warmups_returns_nan():
    """All sets are warmups — no working sets, return NaN."""
    sets = [
        {"load_kg": 60, "reps": 12, "warmup": True},
        {"load_kg": 80, "reps": 5,  "warmup": True},
    ]
    r = estimate_1rm_from_sets(sets, include_warmup=False)
    assert math.isnan(r["one_rm_kg"])
    assert r["chosen_set_idx"] is None


# ────────────────────────────────────────────────────────────────────────────
# Sanity ranges (should not be wildly off for realistic inputs)
# ────────────────────────────────────────────────────────────────────────────
def test_realistic_bench_press_estimate():
    """80 kg × 10 reps for bench press → estimated 1RM should be
    around 100-110 kg (formulas agree to within ~10%)."""
    r = estimate_1rm(80.0, 10, method="ensemble")
    assert 100.0 <= r["one_rm_kg"] <= 115.0


def test_realistic_back_squat_heavy_low_reps():
    """140 kg × 5 reps → expected 1RM ~155-165 kg."""
    r = estimate_1rm(140.0, 5, method="ensemble")
    assert 155.0 <= r["one_rm_kg"] <= 170.0


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
    print(f"=== All {len(fns)} tests passed ===")
