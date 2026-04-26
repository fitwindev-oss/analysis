"""
Unit tests for src/analysis/strength_norms.py (Phase V1-B).

Verifies the transcribed PDF tables against several spot-check
calculations and the bin-lookup edge cases.

Run from project root:
    python tests/test_strength_norms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.strength_norms import (
    grade_1rm, composite_score,
    BENCH_PRESS_MALE, BENCH_PRESS_FEMALE,
    BACK_SQUAT_MALE, BACK_SQUAT_FEMALE,
    DEADLIFT_MALE, DEADLIFT_FEMALE,
    GRADE_LABELS, GRADE_PERCENT, REGION_WEIGHTS,
    EXERCISE_REGION, VALID_EXERCISES,
)


# ────────────────────────────────────────────────────────────────────────────
# Bin lookup edges
# ────────────────────────────────────────────────────────────────────────────
def test_bw_lookup_below_first_bin():
    """A 50 kg male should hit the -54 bin (Bench Press M, lightest)."""
    coefs = BENCH_PRESS_MALE.lookup_bw_coefs(50.0)
    assert coefs == (0.48, 0.76, 1.14, 1.58, 2.06)


def test_bw_lookup_at_bin_boundary():
    """bw = 54 (boundary of -54) → still -54 bin (inclusive upper)."""
    assert BENCH_PRESS_MALE.lookup_bw_coefs(54.0) == (0.48, 0.76, 1.14, 1.58, 2.06)
    """bw = 55 → 55-59 bin."""
    assert BENCH_PRESS_MALE.lookup_bw_coefs(55.0) == (0.53, 0.82, 1.16, 1.58, 2.05)


def test_bw_lookup_far_above_clamps_to_last():
    """200 kg should clamp to the 140- bin."""
    coefs = BENCH_PRESS_MALE.lookup_bw_coefs(200.0)
    assert coefs == (0.53, 0.72, 1.16, 1.42, 1.69)   # 140-


def test_age_lookup_at_boundary():
    """age=29 in bench M → 25-29 bin (factor 1.00)."""
    assert BENCH_PRESS_MALE.lookup_age_factor(29) == 1.00
    """age=30 → 30-34 bin (factor 0.97)."""
    assert BENCH_PRESS_MALE.lookup_age_factor(30) == 0.97


def test_age_lookup_extrapolates_high():
    """age=95 → 90-100 bin (factor 0.33)."""
    assert BENCH_PRESS_MALE.lookup_age_factor(95) == 0.33
    """age=120 (out of range) → clamps to last bin (0.33)."""
    assert BENCH_PRESS_MALE.lookup_age_factor(120) == 0.33


# ────────────────────────────────────────────────────────────────────────────
# grade_1rm — known-value spot checks against PDF
# ────────────────────────────────────────────────────────────────────────────
def test_bench_male_70kg_age30_intermediate():
    """70 kg male, age 30, bench 100 kg.
    bw_coefs (70-74): intermediate=1.21, advanced=1.60
    age 30-34: 0.97
    intermediate threshold = 70 × 1.21 × 0.97 = 82.16 kg
    advanced threshold     = 70 × 1.60 × 0.97 = 108.64 kg
    100 ∈ [82.16, 108.64) → grade 3 (보통)
    """
    r = grade_1rm("bench_press", "M", 30, 70.0, 100.0)
    assert r["grade"] == 3
    assert r["label"] == "보통"
    assert r["warning"] is None
    assert abs(r["thresholds_kg"]["intermediate"] - 82.16) < 0.05
    assert abs(r["thresholds_kg"]["advanced"] - 108.64) < 0.05


def test_squat_male_80kg_age25_advanced():
    """80 kg male, age 25, squat 200 kg.
    bw_coefs (80-84): intermediate=1.63, advanced=2.08, elite=2.56
    age 25-29: 1.00
    advanced threshold = 80 × 2.08 = 166.4
    elite threshold    = 80 × 2.56 = 204.8
    200 ∈ [166.4, 204.8) → grade 2 (좋음)
    """
    r = grade_1rm("back_squat", "M", 25, 80.0, 200.0)
    assert r["grade"] == 2
    assert r["label"] == "좋음"


def test_deadlift_female_60kg_age30_intermediate():
    """60 kg female, age 30, deadlift 100 kg.
    bw_coefs (60-64): intermediate=1.38, advanced=1.88
    age 30-34 (deadlift F): 1.00
    intermediate = 60 × 1.38 = 82.8
    advanced     = 60 × 1.88 = 112.8
    100 ∈ [82.8, 112.8) → grade 3 (보통)
    """
    r = grade_1rm("deadlift", "F", 30, 60.0, 100.0)
    assert r["grade"] == 3
    assert r["label"] == "보통"


def test_elite_grade_1():
    """Far-above-elite-threshold lift → grade 1."""
    r = grade_1rm("deadlift", "M", 25, 80.0, 999.0)
    assert r["grade"] == 1
    assert r["label"] == "엘리트"
    assert r["ratio_to_elite"] > 4.0


def test_below_beginner_caution():
    """1RM at 90% of beginner threshold → grade 6 (경고/caution)."""
    # 80 kg male, 25y, bench. Beginner threshold = 80 × 0.66 × 1.00 = 52.8.
    # 90% of that = 47.52
    r = grade_1rm("bench_press", "M", 25, 80.0, 47.52)
    assert r["grade"] == 6
    assert r["warning"] == "caution"
    assert r["label"] == "경고"


def test_below_beginner_severe():
    """1RM at 50% of beginner threshold → grade 7 (심각/severe)."""
    r = grade_1rm("bench_press", "M", 25, 80.0, 26.4)   # = 0.5 × 52.8
    assert r["grade"] == 7
    assert r["warning"] == "severe"
    assert r["label"] == "심각"


def test_grade_5_beginner():
    """1RM exactly at beginner threshold → grade 5 (위험)."""
    # bench M, 80kg, 25y → beginner threshold = 80 × 0.66 × 1.00 = 52.8
    r = grade_1rm("bench_press", "M", 25, 80.0, 52.8)
    assert r["grade"] == 5
    assert r["label"] == "위험"
    assert r["warning"] is None


def test_thresholds_monotonic():
    """Beginner < Novice < Intermediate < Advanced < Elite — always."""
    for ex in VALID_EXERCISES:
        for sex in ("M", "F"):
            for age in (20, 30, 50, 70):
                for bw in (50, 70, 90, 110):
                    r = grade_1rm(ex, sex, age, bw, 100)
                    t = r["thresholds_kg"]
                    assert t["beginner"] < t["novice"] < t["intermediate"] \
                           < t["advanced"] < t["elite"], \
                           f"non-monotonic for {ex}/{sex}/age{age}/bw{bw}: {t}"


# ────────────────────────────────────────────────────────────────────────────
# Error handling
# ────────────────────────────────────────────────────────────────────────────
def test_unknown_exercise_raises():
    try:
        grade_1rm("shoulder_press", "M", 25, 80.0, 60.0)
        assert False, "should reject unknown exercise"
    except ValueError as e:
        assert "no norm" in str(e).lower()


def test_unknown_sex_raises():
    try:
        grade_1rm("bench_press", "X", 25, 80.0, 60.0)
        assert False, "should reject unknown sex"
    except ValueError:
        pass


def test_lowercase_sex_normalised():
    """sex='m' should work the same as 'M'."""
    r1 = grade_1rm("bench_press", "M", 30, 80.0, 100.0)
    r2 = grade_1rm("bench_press", "m", 30, 80.0, 100.0)
    assert r1["grade"] == r2["grade"]


def test_uppercase_exercise_normalised():
    """Exercise name should be case-insensitive."""
    r1 = grade_1rm("BENCH_PRESS", "M", 30, 80.0, 100.0)
    r2 = grade_1rm("bench_press", "M", 30, 80.0, 100.0)
    assert r1["grade"] == r2["grade"]


# ────────────────────────────────────────────────────────────────────────────
# Composite score
# ────────────────────────────────────────────────────────────────────────────
def test_composite_pdf_example():
    """The PDF gives a worked example:
        biceps=3, triceps=4, shoulder=3, chest=2, legs=5, back=4, whole_body=5
        → score = 7×75 + 7×65 + 9×75 + 13×89 + 20×55 + 20×65 + 24×55 = 6532
        score_pct = 6532 / 100 = 65.32
        65.32 ∈ [56, 65] → grade 4 (per the cutoff table 56-65 → grade 4)
    Wait — re-check: PDF says 65.32 → "65점 (4등급)" (rounded down).
    But 65 is the upper-inclusive of grade 4 (56-65) — so grade 4.
    """
    r = composite_score({
        "biceps":     3,
        "triceps":    4,
        "shoulder":   3,
        "chest":      2,
        "legs":       5,
        "back":       4,
        "whole_body": 5,
    })
    assert abs(r["score_pct"] - 65.32) < 0.01
    assert r["composite_grade"] == 4
    assert r["weighted_total"] == 100   # all 7 regions = full weight
    assert r["n_regions"] == 7


def test_composite_partial_v1_renormalises():
    """V1 only measures chest/legs/whole_body (3 of 7). Composite still
    produces a meaningful 0–100 score normalised to those 3."""
    r = composite_score({"chest": 2, "legs": 3, "whole_body": 1})
    # weight: 13 + 20 + 24 = 57
    # score:  13×89 + 20×75 + 24×100 = 1157 + 1500 + 2400 = 5057
    # score_pct = 5057 / 57 ≈ 88.7
    assert abs(r["score_pct"] - 88.72) < 0.05
    assert r["composite_grade"] == 2   # 88.7 ∈ [76, 89] → grade 2
    assert r["weighted_total"] == 57
    assert r["n_regions"] == 3


def test_composite_empty_input():
    r = composite_score({})
    assert r["composite_grade"] == 7
    assert r["n_regions"] == 0
    assert r["weighted_total"] == 0


def test_composite_unknown_region_raises():
    try:
        composite_score({"toes": 1})
        assert False
    except ValueError:
        pass


def test_composite_invalid_grade_raises():
    try:
        composite_score({"chest": 99})
        assert False
    except ValueError:
        pass


# ────────────────────────────────────────────────────────────────────────────
# Cross-table sanity
# ────────────────────────────────────────────────────────────────────────────
def test_male_thresholds_higher_than_female():
    """At same bw / age, male thresholds should generally be higher."""
    rm = grade_1rm("bench_press", "M", 30, 70.0, 100.0)
    rf = grade_1rm("bench_press", "F", 30, 70.0, 100.0)
    assert rm["thresholds_kg"]["elite"] > rf["thresholds_kg"]["elite"]


def test_age_factor_drops_after_30():
    """age 50 should have a lower factor than age 25 for the same lift."""
    r25 = grade_1rm("bench_press", "M", 25, 80.0, 100.0)
    r50 = grade_1rm("bench_press", "M", 50, 80.0, 100.0)
    assert r50["age_factor"] < r25["age_factor"]


def test_v1_exercises_cover_three_regions():
    """V1 maps each barbell lift to exactly one body region, and the
    three regions don't overlap."""
    regions = {EXERCISE_REGION[ex] for ex in VALID_EXERCISES}
    assert regions == {"chest", "legs", "whole_body"}


def test_grade_labels_exhaustive():
    """All 7 grades must have a Korean label."""
    for g in range(1, 8):
        assert g in GRADE_LABELS
        assert isinstance(GRADE_LABELS[g], str)
        assert g in GRADE_PERCENT


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
