"""
Unit tests for src/analysis/ssc.py (Phase V4).

Run from project root:
    python tests/test_ssc.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.analysis.ssc as ssc_mod
from src.analysis.ssc import (
    eccentric_utilization_ratio, ssc_contribution_pct,
    grade_ssc, interpret_ssc, compute_ssc, SSCResult,
    INTERPRETATION_STRENGTH_FOCUS, INTERPRETATION_ELASTIC_FOCUS,
)


# ────────────────────────────────────────────────────────────────────────────
# eccentric_utilization_ratio
# ────────────────────────────────────────────────────────────────────────────
def test_eur_basic():
    """40 cm CMJ vs 35 cm SJ → 1.143."""
    assert abs(eccentric_utilization_ratio(0.40, 0.35) - (0.40 / 0.35)) < 1e-9


def test_eur_no_ssc_benefit():
    """Equal heights → EUR = 1.0."""
    assert eccentric_utilization_ratio(0.30, 0.30) == 1.0


def test_eur_negative_when_cmj_lower():
    """CMJ < SJ → EUR < 1 (red flag)."""
    eur = eccentric_utilization_ratio(0.28, 0.30)
    assert eur < 1.0


def test_eur_zero_sj_returns_nan():
    """Division-by-zero guard."""
    assert math.isnan(eccentric_utilization_ratio(0.40, 0.0))
    assert math.isnan(eccentric_utilization_ratio(0.40, -0.1))


# ────────────────────────────────────────────────────────────────────────────
# ssc_contribution_pct
# ────────────────────────────────────────────────────────────────────────────
def test_ssc_basic():
    """40 cm CMJ vs 35 cm SJ → (40-35)/35 × 100 = 14.29%."""
    assert abs(ssc_contribution_pct(0.40, 0.35) - 14.2857) < 1e-3


def test_ssc_zero_when_equal():
    assert ssc_contribution_pct(0.30, 0.30) == 0.0


def test_ssc_negative_when_cmj_lower():
    assert ssc_contribution_pct(0.27, 0.30) < 0


def test_ssc_zero_sj_returns_nan():
    assert math.isnan(ssc_contribution_pct(0.40, 0.0))


# ────────────────────────────────────────────────────────────────────────────
# Grade lookup
# ────────────────────────────────────────────────────────────────────────────
def test_grade_elite_at_eur_1_15():
    """EUR ≥ 1.15 → grade 1."""
    assert grade_ssc(1.15)[0] == 1
    assert grade_ssc(1.20)[0] == 1
    assert grade_ssc(2.00)[0] == 1


def test_grade_good_zone():
    """1.10 ≤ EUR < 1.15 → grade 2."""
    assert grade_ssc(1.10)[0] == 2
    assert grade_ssc(1.14)[0] == 2
    assert grade_ssc(1.149)[0] == 2


def test_grade_average_zone():
    """1.05 ≤ EUR < 1.10 → grade 3."""
    assert grade_ssc(1.05)[0] == 3
    assert grade_ssc(1.09)[0] == 3


def test_grade_poor_zone():
    """1.01 ≤ EUR < 1.05 → grade 4."""
    assert grade_ssc(1.01)[0] == 4
    assert grade_ssc(1.04)[0] == 4
    assert grade_ssc(1.049)[0] == 4


def test_grade_risk_zone():
    """EUR ≤ 1.00 → grade 5."""
    assert grade_ssc(1.00)[0] == 5
    assert grade_ssc(0.95)[0] == 5
    assert grade_ssc(0.50)[0] == 5


def test_grade_nan_returns_5():
    """NaN EUR → safe fallback to grade 5."""
    assert grade_ssc(float("nan"))[0] == 5


def test_grade_labels():
    """English labels per grade."""
    assert grade_ssc(1.20)[1] == "Elite"
    assert grade_ssc(1.10)[1] == "Good"
    assert grade_ssc(1.05)[1] == "Average"
    assert grade_ssc(1.02)[1] == "Poor"
    assert grade_ssc(0.99)[1] == "Risk"


# ────────────────────────────────────────────────────────────────────────────
# Dual interpretation
# ────────────────────────────────────────────────────────────────────────────
def test_interpret_strength_focus_for_low_1rm_grade():
    """1RM grade ≥ 3 (보통/나쁨/위험) → strength-focus interpretation."""
    for g_1rm in (3, 4, 5, 6, 7):
        focus, text = interpret_ssc(grade=2, lower_body_1rm_grade=g_1rm)
        assert focus == "strength"
        assert text == INTERPRETATION_STRENGTH_FOCUS[2]


def test_interpret_elastic_focus_for_high_1rm_grade():
    """1RM grade ≤ 2 (좋음/엘리트) → elastic-focus interpretation."""
    for g_1rm in (1, 2):
        focus, text = interpret_ssc(grade=3, lower_body_1rm_grade=g_1rm)
        assert focus == "elastic"
        assert text == INTERPRETATION_ELASTIC_FOCUS[3]


def test_interpret_defaults_to_strength_when_1rm_unknown():
    """No 1RM grade → default to strength-focus framing."""
    focus, text = interpret_ssc(grade=2, lower_body_1rm_grade=None)
    assert focus == "strength"


def test_all_grades_have_both_interpretation_texts():
    for g in (1, 2, 3, 4, 5):
        assert g in INTERPRETATION_STRENGTH_FOCUS
        assert g in INTERPRETATION_ELASTIC_FOCUS
        assert len(INTERPRETATION_STRENGTH_FOCUS[g]) > 0
        assert len(INTERPRETATION_ELASTIC_FOCUS[g]) > 0


# ────────────────────────────────────────────────────────────────────────────
# compute_ssc — end-to-end with patched DB lookup
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class FakeSession:
    id: str
    subject_id: str
    test_type: str
    session_date: str
    session_dir: str
    status: str = "analyzed"
    duration_s: float = 0.0
    options_json: Optional[str] = None
    trainer: Optional[str] = None
    notes: Optional[str] = None


def _write_jump_session(tmpdir: Path, sid: str, test_type: str,
                        height_m: float,
                        subject_id: str = "subj_v4") -> FakeSession:
    sd = tmpdir / sid
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "session.json").write_text(json.dumps({
        "test": test_type, "subject_id": subject_id,
    }), encoding="utf-8")
    (sd / "result.json").write_text(json.dumps({
        "result": {"jump_height_m_impulse": height_m}
    }), encoding="utf-8")
    return FakeSession(
        id=sid, subject_id=subject_id,
        test_type=test_type,
        session_date="2026-04-27T10:00:00",
        session_dir=str(sd),
    )


def _patch_lookups(tmpdir: Path,
                    cmj_h: Optional[float],
                    sj_h: Optional[float],
                    legs_grade: Optional[int]):
    """Stub _latest_jump_height + _lower_body_grade for compute_ssc."""
    sessions = []
    if cmj_h is not None:
        sessions.append(_write_jump_session(
            tmpdir, "cmj_s", "cmj", cmj_h))
    if sj_h is not None:
        sessions.append(_write_jump_session(
            tmpdir, "sj_s", "sj", sj_h))

    def fake_latest(subject_id, test_type):
        for s in sessions:
            if s.test_type != test_type or s.subject_id != subject_id:
                continue
            return _read_height(s)
        return None

    def fake_lower_grade(subject_id):
        return legs_grade

    ssc_mod._latest_jump_height = fake_latest
    ssc_mod._lower_body_grade = fake_lower_grade


def _read_height(sess: FakeSession):
    data = json.loads(
        (Path(sess.session_dir) / "result.json").read_text(encoding="utf-8"))
    h = data["result"]["jump_height_m_impulse"]
    return float(h), sess.id, sess.session_date


def test_compute_ssc_full_flow_grade_2():
    """40 cm CMJ vs 35 cm SJ, lower-body 1RM grade 4 →
    EUR 1.143 (grade 2) + strength-focus interpretation."""
    with tempfile.TemporaryDirectory() as td:
        _patch_lookups(Path(td), cmj_h=0.40, sj_h=0.35, legs_grade=4)
        r = compute_ssc("subj_v4")
    assert r.skipped_reason is None
    assert abs(r.eur - 0.40 / 0.35) < 1e-3
    assert r.grade == 2
    assert r.interpretation_focus == "strength"
    assert "근력 수준" in (r.interpretation or "")


def test_compute_ssc_elite_advanced_lifter_uses_elastic_focus():
    """High EUR (Elite, 1.20) + already-strong subject (legs grade 1)
    → elastic-focus interpretation framing."""
    with tempfile.TemporaryDirectory() as td:
        _patch_lookups(Path(td), cmj_h=0.42, sj_h=0.35, legs_grade=1)
        r = compute_ssc("subj_v4")
    assert r.grade == 1
    assert r.interpretation_focus == "elastic"


def test_compute_ssc_skipped_when_cmj_missing():
    with tempfile.TemporaryDirectory() as td:
        _patch_lookups(Path(td), cmj_h=None, sj_h=0.30, legs_grade=3)
        r = compute_ssc("subj_v4")
    assert r.skipped_reason is not None
    assert "CMJ" in r.skipped_reason


def test_compute_ssc_skipped_when_sj_missing():
    with tempfile.TemporaryDirectory() as td:
        _patch_lookups(Path(td), cmj_h=0.40, sj_h=None, legs_grade=3)
        r = compute_ssc("subj_v4")
    assert r.skipped_reason is not None
    assert "SJ" in r.skipped_reason


def test_compute_ssc_skipped_when_zero_height():
    with tempfile.TemporaryDirectory() as td:
        _patch_lookups(Path(td), cmj_h=0.40, sj_h=0.0, legs_grade=3)
        r = compute_ssc("subj_v4")
    assert r.skipped_reason is not None


def test_compute_ssc_default_strength_focus_when_no_1rm():
    """Subject hasn't done a back_squat yet → fall back to
    strength-focus interpretation regardless of SSC grade."""
    with tempfile.TemporaryDirectory() as td:
        _patch_lookups(Path(td), cmj_h=0.40, sj_h=0.35, legs_grade=None)
        r = compute_ssc("subj_v4")
    assert r.lower_body_1rm_grade is None
    assert r.interpretation_focus == "strength"


def test_compute_ssc_to_dict_jsonable():
    with tempfile.TemporaryDirectory() as td:
        _patch_lookups(Path(td), cmj_h=0.40, sj_h=0.35, legs_grade=2)
        r = compute_ssc("subj_v4")
    d = r.to_dict()
    json.dumps(d, default=str)
    assert d["grade"] == 2
    assert "eur" in d


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
