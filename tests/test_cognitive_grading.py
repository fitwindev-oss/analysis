"""
Unit tests for V6-G1 — gamified grading + Cognitive Reaction Index (CRI).

Covers:
  - grade_trial threshold boundaries (350/500/750 ms; miss vs hit)
  - grade_score weight table
  - cri_letter_grade band boundaries
  - compute_cri MS / AS / CS components on synthetic trial sets
  - CV calculation with single / multi trial
  - 100% / 0% edge cases

Run:
    python tests/test_cognitive_grading.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.cognitive_reaction import (
    grade_trial, grade_score, cri_letter_grade, compute_cri,
    live_cri_after,
    RT_GREAT_MS, RT_GOOD_MS, RT_NORMAL_MS,
    GRADE_WEIGHT, GRADE_LABELS_KO, CRI_GRADE_BANDS,
    CogTrial,
)


# ────────────────────────────────────────────────────────────────────────────
# grade_trial — threshold boundaries
# ────────────────────────────────────────────────────────────────────────────
def test_grade_trial_great_at_boundary():
    assert grade_trial(0.0, True) == "great"
    assert grade_trial(200.0, True) == "great"
    assert grade_trial(RT_GREAT_MS, True) == "great"
    assert grade_trial(RT_GREAT_MS + 0.01, True) == "good"


def test_grade_trial_good_band():
    assert grade_trial(RT_GREAT_MS + 1, True) == "good"
    assert grade_trial(450.0, True) == "good"
    assert grade_trial(RT_GOOD_MS, True) == "good"
    assert grade_trial(RT_GOOD_MS + 0.01, True) == "normal"


def test_grade_trial_normal_band():
    assert grade_trial(RT_GOOD_MS + 1, True) == "normal"
    assert grade_trial(650.0, True) == "normal"
    assert grade_trial(RT_NORMAL_MS, True) == "normal"
    assert grade_trial(RT_NORMAL_MS + 0.01, True) == "bad"


def test_grade_trial_bad_band():
    assert grade_trial(900.0, True) == "bad"
    assert grade_trial(2000.0, True) == "bad"


def test_grade_trial_miss_overrides_hit_quality():
    """Even with great RT, a miss is a miss."""
    assert grade_trial(200.0, False) == "miss"
    assert grade_trial(800.0, False) == "miss"


def test_grade_trial_none_or_nan_returns_miss():
    assert grade_trial(None, True) == "miss"
    assert grade_trial(None, False) == "miss"
    import math
    assert grade_trial(math.nan, True) == "miss"


# ────────────────────────────────────────────────────────────────────────────
# grade_score — weight table
# ────────────────────────────────────────────────────────────────────────────
def test_grade_score_weights():
    assert grade_score("great") == 1.00
    assert grade_score("good") == 0.75
    assert grade_score("normal") == 0.50
    assert grade_score("bad") == 0.20
    assert grade_score("miss") == 0.00
    # Unknown grade defaults to 0
    assert grade_score("unknown") == 0.00


def test_grade_weight_table_complete():
    """All 5 expected keys must be in GRADE_WEIGHT."""
    expected = {"great", "good", "normal", "bad", "miss"}
    assert set(GRADE_WEIGHT.keys()) == expected


def test_grade_labels_ko_complete():
    """Korean labels exist for every grade."""
    for k in GRADE_WEIGHT.keys():
        assert k in GRADE_LABELS_KO
        assert isinstance(GRADE_LABELS_KO[k], str)
        assert len(GRADE_LABELS_KO[k]) > 0


# ────────────────────────────────────────────────────────────────────────────
# cri_letter_grade — band boundaries
# ────────────────────────────────────────────────────────────────────────────
def test_cri_letter_grade_boundaries():
    assert cri_letter_grade(100.0)[0] == "A"
    assert cri_letter_grade(85.0)[0] == "A"
    assert cri_letter_grade(84.99)[0] == "B"
    assert cri_letter_grade(70.0)[0] == "B"
    assert cri_letter_grade(69.99)[0] == "C"
    assert cri_letter_grade(55.0)[0] == "C"
    assert cri_letter_grade(54.99)[0] == "D"
    assert cri_letter_grade(40.0)[0] == "D"
    assert cri_letter_grade(39.99)[0] == "E"
    assert cri_letter_grade(0.0)[0] == "E"


def test_cri_letter_grade_returns_korean_label():
    _, label = cri_letter_grade(90.0)
    assert label == "매우 우수"
    _, label = cri_letter_grade(75.0)
    assert label == "우수"
    _, label = cri_letter_grade(20.0)
    assert label == "부족"


# ────────────────────────────────────────────────────────────────────────────
# compute_cri — composite scoring
# ────────────────────────────────────────────────────────────────────────────
def _mk_trial(rt_ms, hit=True, grade=None):
    return CogTrial(
        trial_idx=0, target_label="pos_N",
        target_x_norm=0.5, target_y_norm=0.5,
        t_stim_s=0.0, rt_ms=rt_ms, hit=hit,
        grade=grade or grade_trial(rt_ms, hit),
    )


def test_compute_cri_empty_returns_zero():
    out = compute_cri([])
    assert out["cri"] == 0.0
    assert out["mean_score"] == 0.0
    assert out["accuracy_score"] == 0.0
    assert out["consistency_score"] == 0.0
    assert out["overall_grade"] == "E"


def test_compute_cri_all_great_perfect_accuracy():
    """10 trials all under 350 ms + all hit → MS = 100, AS = 100,
    CS depends on RT spread."""
    trials = [_mk_trial(rt) for rt in (200, 220, 240, 260, 280,
                                        300, 320, 220, 240, 260)]
    out = compute_cri(trials)
    assert out["mean_score"] == 100.0
    assert out["accuracy_score"] == 100.0
    # 10 trials, RT range ~200-320 → small CV → CS high
    assert out["consistency_score"] > 70.0
    # CRI = 0.5*100 + 0.3*100 + 0.2*CS ≥ 80 + 14 = 94
    assert out["cri"] >= 90.0
    assert out["overall_grade"] == "A"


def test_compute_cri_all_misses_zero_cri():
    trials = [_mk_trial(None, hit=False) for _ in range(10)]
    out = compute_cri(trials)
    assert out["mean_score"] == 0.0
    assert out["accuracy_score"] == 0.0
    # CS = 0 (no valid RTs)
    assert out["consistency_score"] == 0.0
    assert out["cri"] == 0.0
    assert out["overall_grade"] == "E"


def test_compute_cri_grade_counts_match_trials():
    """5 great + 3 good + 1 normal + 1 miss → counts must match."""
    trials = (
        [_mk_trial(250) for _ in range(5)] +    # great
        [_mk_trial(450) for _ in range(3)] +    # good
        [_mk_trial(650)] +                       # normal
        [_mk_trial(None, hit=False)]             # miss
    )
    out = compute_cri(trials)
    assert out["grade_counts"]["great"] == 5
    assert out["grade_counts"]["good"] == 3
    assert out["grade_counts"]["normal"] == 1
    assert out["grade_counts"]["miss"] == 1
    assert out["grade_counts"]["bad"] == 0


def test_compute_cri_weighted_mean_score():
    """5×great (1.0) + 5×bad (0.2) → MS = (5×1 + 5×0.2)/10 × 100 = 60."""
    trials = ([_mk_trial(250) for _ in range(5)] +
              [_mk_trial(900) for _ in range(5)])
    out = compute_cri(trials)
    expected_ms = (5 * 1.0 + 5 * 0.2) / 10 * 100
    assert abs(out["mean_score"] - expected_ms) < 1e-6


def test_compute_cri_accuracy_excludes_misses():
    """8 hits + 2 misses → AS = 80 %."""
    trials = ([_mk_trial(400) for _ in range(8)] +
              [_mk_trial(None, hit=False) for _ in range(2)])
    out = compute_cri(trials)
    assert abs(out["accuracy_score"] - 80.0) < 1e-6


def test_compute_cri_consistency_higher_for_uniform_rts():
    """Uniform RTs should produce CS ≈ 100; spread RTs lower CS."""
    uniform = [_mk_trial(400) for _ in range(10)]
    spread  = [_mk_trial(rt) for rt in (200, 700, 250, 650, 300,
                                          600, 350, 550, 400, 500)]
    out_u = compute_cri(uniform)
    out_s = compute_cri(spread)
    assert out_u["consistency_score"] > out_s["consistency_score"]
    assert out_u["consistency_score"] > 95.0


def test_compute_cri_with_dict_trials():
    """compute_cri should also accept plain dicts (for live HUD use
    where the recorder builds light-weight dicts)."""
    trials_dict = [
        {"rt_ms": 300, "hit": True, "grade": "great"},
        {"rt_ms": 450, "hit": True, "grade": "good"},
        {"rt_ms": None, "hit": False, "grade": "miss"},
    ]
    out = compute_cri(trials_dict)
    assert out["grade_counts"]["great"] == 1
    assert out["grade_counts"]["good"] == 1
    assert out["grade_counts"]["miss"] == 1
    # MS = (1.0 + 0.75 + 0.0)/3 × 100 ≈ 58.33
    assert 58.0 <= out["mean_score"] <= 59.0


def test_live_cri_after_matches_compute_cri():
    """Convenience helper must agree with full compute_cri."""
    trials = [_mk_trial(300), _mk_trial(500), _mk_trial(None, hit=False)]
    full = compute_cri(trials)
    live = live_cri_after(trials)
    assert abs(live - full["cri"]) < 1e-9


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
