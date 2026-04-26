"""
Unit tests for V5 — squat CoP safety + L/R asymmetry warnings.

The V5 helpers (asymmetry_level / classify_cop_safety / compute_quiet_stance
/ compute_cop_safety_per_rep) are pure functions — straightforward to
exercise with synthetic inputs.

Run from project root:
    python tests/test_squat_v5.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.squat import (
    asymmetry_level, classify_cop_safety,
    compute_quiet_stance, compute_cop_safety_per_rep,
    ASYM_CAUTION_PCT, ASYM_WARNING_PCT,
)


# ────────────────────────────────────────────────────────────────────────────
# asymmetry_level
# ────────────────────────────────────────────────────────────────────────────
def test_asym_ok_below_caution():
    assert asymmetry_level(0.0) == "ok"
    assert asymmetry_level(2.5) == "ok"
    assert asymmetry_level(4.99) == "ok"


def test_asym_caution_zone():
    """5 ≤ |asym| < 10 → caution."""
    assert asymmetry_level(5.0) == "caution"
    assert asymmetry_level(7.5) == "caution"
    assert asymmetry_level(9.99) == "caution"


def test_asym_warning_zone():
    """|asym| ≥ 10 → warning."""
    assert asymmetry_level(10.0) == "warning"
    assert asymmetry_level(20.0) == "warning"


def test_asym_handles_none_and_negative():
    """None → 'ok' default; negative values use absolute value."""
    assert asymmetry_level(None) == "ok"
    assert asymmetry_level(-12.0) == "warning"
    assert asymmetry_level(-6.0) == "caution"


def test_asym_thresholds_match_module_constants():
    """The boundary values must equal the documented constants."""
    assert ASYM_CAUTION_PCT == 5.0
    assert ASYM_WARNING_PCT == 10.0


# ────────────────────────────────────────────────────────────────────────────
# classify_cop_safety
# ────────────────────────────────────────────────────────────────────────────
def test_safety_grade_1_centered():
    """Small drifts inside the tightest band → grade 1, no warning."""
    g, w = classify_cop_safety(ap_drift_mm=-10, ml_drift_max_mm=15)
    assert g == 1
    assert w is None


def test_safety_grade_1_at_band_edge():
    """Boundary inclusion: AP=-25 (rear edge) and ML=25 → grade 1."""
    assert classify_cop_safety(-25, 25)[0] == 1
    assert classify_cop_safety(5, 25)[0] == 1


def test_safety_grade_2_wider_band():
    """Drift outside grade-1 but inside grade-2 → grade 2."""
    g, w = classify_cop_safety(ap_drift_mm=-30, ml_drift_max_mm=30)
    assert g == 2
    assert w is None


def test_safety_grade_3_loose_band():
    g, w = classify_cop_safety(ap_drift_mm=-50, ml_drift_max_mm=45)
    assert g == 3
    assert w is None


def test_safety_grade_4_forward_lean():
    """AP drift > 25 mm forward → grade 4 + 'forward_lean' warning."""
    g, w = classify_cop_safety(ap_drift_mm=30, ml_drift_max_mm=20)
    assert g == 4
    assert w == "forward_lean"


def test_safety_grade_5_severe_forward_lean():
    """Forward drift > 50 mm → grade 5."""
    g, w = classify_cop_safety(ap_drift_mm=60, ml_drift_max_mm=20)
    assert g == 5
    assert w == "forward_lean"


def test_safety_grade_4_lateral_drift():
    """Excessive ML drift but acceptable AP → 'lateral_drift' warning."""
    g, w = classify_cop_safety(ap_drift_mm=-15, ml_drift_max_mm=70)
    assert g == 4
    assert w == "lateral_drift"


def test_safety_grade_5_extreme_lateral():
    """ML > 100 mm escalates to grade 5."""
    g, w = classify_cop_safety(ap_drift_mm=-15, ml_drift_max_mm=120)
    assert g == 5


def test_safety_rearfoot_excessive():
    """AP < -55 mm → 'rearfoot_excessive' warning."""
    g, w = classify_cop_safety(ap_drift_mm=-65, ml_drift_max_mm=20)
    assert g == 4
    assert w == "rearfoot_excessive"


def test_safety_rearfoot_extreme():
    """AP < -90 mm → grade 5."""
    g, w = classify_cop_safety(ap_drift_mm=-100, ml_drift_max_mm=20)
    assert g == 5
    assert w == "rearfoot_excessive"


# ────────────────────────────────────────────────────────────────────────────
# compute_quiet_stance — uses a fake ForceSession
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class _FakeForce:
    t_s:    np.ndarray
    cop_x:  np.ndarray
    cop_y:  np.ndarray


def test_quiet_stance_basic():
    """A pre-rep window of CoP averaging gives the mean."""
    fs = 100.0
    n = 1000
    t = np.arange(n) / fs
    cx = np.full(n, 280.0)
    cy = np.full(n, 215.0)
    force = _FakeForce(t_s=t, cop_x=cx, cop_y=cy)
    # First rep starts at t = 5 s (index 500). Pre-window 1 s (100 samples).
    qx, qy = compute_quiet_stance(force, fs, [(500, 600, 700)])
    assert qx == 280.0
    assert qy == 215.0


def test_quiet_stance_handles_nan():
    """NaN samples in the window are excluded from the mean."""
    fs = 100.0
    n = 1000
    t = np.arange(n) / fs
    cx = np.full(n, 280.0)
    cy = np.full(n, 215.0)
    cx[400:450] = np.nan
    force = _FakeForce(t_s=t, cop_x=cx, cop_y=cy)
    qx, qy = compute_quiet_stance(force, fs, [(500, 600, 700)])
    assert qx == 280.0   # NaN samples ignored


def test_quiet_stance_no_reps_returns_none():
    fs = 100.0
    force = _FakeForce(
        t_s=np.array([0.0]), cop_x=np.array([280.0]), cop_y=np.array([215.0]))
    qx, qy = compute_quiet_stance(force, fs, [])
    assert qx is None and qy is None


def test_quiet_stance_no_pre_window_returns_none():
    """When the first rep starts at t = 0, no pre-window data exists."""
    fs = 100.0
    n = 100
    force = _FakeForce(
        t_s=np.arange(n) / fs,
        cop_x=np.full(n, 280.0),
        cop_y=np.full(n, 215.0),
    )
    # First rep starts at index 0 — no room for a 1-s pre-window.
    qx, qy = compute_quiet_stance(force, fs, [(0, 50, 90)])
    assert qx is None and qy is None


# ────────────────────────────────────────────────────────────────────────────
# compute_cop_safety_per_rep
# ────────────────────────────────────────────────────────────────────────────
def test_safety_per_rep_basic():
    """20 mm rear-foot drift, 15 mm lateral wobble inside grade-1 band."""
    fs = 100.0
    n = 200
    cx = np.full(n, 280.0)   # quiet x
    cy = np.full(n, 215.0)   # quiet y
    # During rep (50:150), CoP drifts back to 200 (Y) and wobbles to 295 (X)
    cy[50:150] = 200.0
    cx[50:150] = 295.0
    force = _FakeForce(t_s=np.arange(n) / fs, cop_x=cx, cop_y=cy)
    ap, ml = compute_cop_safety_per_rep(
        force, i_start=50, i_end=149, quiet_x=280.0, quiet_y=215.0)
    assert abs(ap - (200 - 215)) < 1e-9       # -15 mm rearward
    assert abs(ml - 15.0) < 1e-9               # 15 mm lateral
    g, w = classify_cop_safety(ap, ml)
    assert g == 1


def test_safety_per_rep_returns_none_without_quiet():
    fs = 100.0
    n = 200
    force = _FakeForce(
        t_s=np.arange(n) / fs,
        cop_x=np.full(n, 280.0), cop_y=np.full(n, 215.0))
    ap, ml = compute_cop_safety_per_rep(
        force, i_start=50, i_end=149, quiet_x=None, quiet_y=215.0)
    assert ap is None and ml is None


def test_safety_per_rep_handles_all_nan():
    """When all rep samples are NaN, returns (None, None)."""
    fs = 100.0
    n = 200
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    force = _FakeForce(t_s=np.arange(n) / fs, cop_x=cx, cop_y=cy)
    ap, ml = compute_cop_safety_per_rep(
        force, i_start=50, i_end=149, quiet_x=280.0, quiet_y=215.0)
    assert ap is None and ml is None


def test_safety_per_rep_forward_drift_triggers_warning():
    """Forward-toe drift produces grade 4+ + forward_lean warning when
    fed through classify_cop_safety."""
    fs = 100.0
    n = 200
    cx = np.full(n, 280.0)
    cy = np.full(n, 215.0)
    cy[50:150] = 250.0     # +35 mm forward (grade-3 band breached)
    force = _FakeForce(t_s=np.arange(n) / fs, cop_x=cx, cop_y=cy)
    ap, ml = compute_cop_safety_per_rep(
        force, i_start=50, i_end=149, quiet_x=280.0, quiet_y=215.0)
    assert ap == 35.0
    g, w = classify_cop_safety(ap, ml)
    assert g >= 4
    assert w == "forward_lean"


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
