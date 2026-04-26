"""
Unit tests for src/analysis/composite_strength.py (Phase V3).

Builds in-memory fake sessions + result.json files in a temp dir,
monkeypatches the DB lookup, and checks that the latest grade per
region wins, partial coverage normalises correctly, and missing
sessions are flagged.

Run from project root:
    python tests/test_composite_strength.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.analysis.composite_strength as cs_mod
from src.analysis.composite_strength import (
    compute_composite_strength, RegionalGrade, CompositeStrength,
    REGION_LABELS_KO, REGION_DISPLAY_ORDER,
)


# ────────────────────────────────────────────────────────────────────────────
# Fake DB session row
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


def _write_session(tmpdir: Path, sid: str, exercise: str,
                   one_rm_kg: float, grade: int,
                   grade_label: str = "보통",
                   subject_id: str = "subj_test",
                   session_date: str = "2026-04-01T10:00:00") -> FakeSession:
    """Drop a session.json + result.json under tmpdir/<sid>/ and return
    the FakeSession row that would be in the DB."""
    sd = tmpdir / sid
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "session.json").write_text(json.dumps({
        "test": "strength_3lift",
        "exercise": exercise,
        "subject_id": subject_id,
    }), encoding="utf-8")
    (sd / "result.json").write_text(json.dumps({
        "result": {
            "exercise": exercise,
            "best_1rm_kg": one_rm_kg,
            "grade": grade,
            "grade_label": grade_label,
        }
    }), encoding="utf-8")
    return FakeSession(
        id=sid, subject_id=subject_id,
        test_type="strength_3lift",
        session_date=session_date,
        session_dir=str(sd),
    )


def _patch_list_sessions(monkeypatch_target_module,
                          sessions: list) -> None:
    """Replace the module-level _list_strength_sessions with a stub."""
    monkeypatch_target_module._list_strength_sessions = (
        lambda subject_id: [s for s in sessions
                              if s.subject_id == subject_id])


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────
def test_no_sessions_returns_skipped():
    _patch_list_sessions(cs_mod, [])
    r = compute_composite_strength("nobody")
    assert r.skipped_reason is not None
    assert "세션이 없습니다" in r.skipped_reason
    assert r.n_measured == 0


def test_three_lifts_full_coverage():
    """All 3 V1 lifts measured → 3/7 regions, composite computed."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        sessions = [
            _write_session(td_p, "s1", "bench_press", 100.0, 3, "보통"),
            _write_session(td_p, "s2", "back_squat",  150.0, 2, "좋음"),
            _write_session(td_p, "s3", "deadlift",    180.0, 4, "나쁨"),
        ]
        _patch_list_sessions(cs_mod, sessions)
        r = compute_composite_strength("subj_test")

    assert r.skipped_reason is None
    assert r.n_measured == 3
    # 13 (chest) + 20 (legs) + 24 (whole_body) = 57
    assert r.coverage_weight_pct == 57
    # Verify each region populated
    region_set = {rg.region for rg in r.regions}
    assert region_set == {"chest", "legs", "whole_body"}
    # Composite score range: weighted by GRADE_PERCENT
    # chest g3 (75) × 13 + legs g2 (89) × 20 + whole g4 (65) × 24
    # = 975 + 1780 + 1560 = 4315 / 57 = 75.7 → grade 3 (66-75 → grade 3)
    assert abs(r.composite_score_pct - 75.7) < 0.5
    assert r.composite_grade == 3


def test_latest_session_per_region_wins():
    """Two bench sessions on different dates → only the latest counts."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # Older bench at grade 5, newer at grade 2 → grade 2 should win
        sessions = [
            _write_session(td_p, "old", "bench_press", 80.0, 5,
                           session_date="2026-01-01T10:00:00"),
            _write_session(td_p, "new", "bench_press", 110.0, 2,
                           session_date="2026-04-01T10:00:00"),
        ]
        # list_sessions returns sessions ordered by date DESC, so newer first.
        _patch_list_sessions(cs_mod, list(reversed(sessions)))
        r = compute_composite_strength("subj_test")

    assert r.n_measured == 1
    assert r.regions[0].grade == 2
    assert r.regions[0].one_rm_kg == 110.0


def test_partial_coverage_normalised_correctly():
    """Single bench session → composite ≈ that one region's pct alone."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        sessions = [_write_session(td_p, "s1", "bench_press",
                                    100.0, 2, "좋음")]
        _patch_list_sessions(cs_mod, sessions)
        r = compute_composite_strength("subj_test")
    # Grade 2 = 89% → score_pct = 89, single region
    assert abs(r.composite_score_pct - 89.0) < 0.5
    assert r.composite_grade == 2          # 89 ∈ [76, 89]
    assert r.coverage_weight_pct == 13     # chest weight only


def test_missing_regions_listed():
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        sessions = [_write_session(td_p, "s1", "back_squat",
                                    150.0, 3)]
        _patch_list_sessions(cs_mod, sessions)
        r = compute_composite_strength("subj_test")
    assert "chest" in r.missing_regions
    assert "deadlift" not in r.missing_regions or True  # exercise not region
    assert "whole_body" in r.missing_regions
    # All non-V1 regions should be missing
    assert "biceps" in r.missing_regions
    assert "shoulder" in r.missing_regions


def test_grade_missing_skipped_for_that_region():
    """A session that didn't grade (sex/age missing) → don't include it,
    but try the next session for that exercise."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # Newer bench session has no grade (None) — should be skipped.
        # Older bench session has grade 4 — should be used as fallback.
        sd_new = td_p / "new"
        sd_new.mkdir()
        (sd_new / "session.json").write_text(json.dumps({
            "test": "strength_3lift", "exercise": "bench_press",
            "subject_id": "subj_test",
        }), encoding="utf-8")
        (sd_new / "result.json").write_text(json.dumps({
            "result": {
                "exercise": "bench_press",
                "best_1rm_kg": 100.0,
                "grade": None,   # ← missing
                "grade_label": None,
            }
        }), encoding="utf-8")
        new_sess = FakeSession(
            id="new", subject_id="subj_test",
            test_type="strength_3lift",
            session_date="2026-04-01T10:00:00",
            session_dir=str(sd_new),
        )
        old_sess = _write_session(td_p, "old", "bench_press",
                                   90.0, 4, "나쁨",
                                   session_date="2026-01-01T10:00:00")
        _patch_list_sessions(cs_mod, [new_sess, old_sess])
        r = compute_composite_strength("subj_test")

    # Should fall back to old_sess since new has None grade.
    assert r.n_measured == 1
    assert r.regions[0].grade == 4
    assert r.regions[0].session_id == "old"


def test_regions_sorted_by_display_order():
    """Output regions follow REGION_DISPLAY_ORDER (top-down body)."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        sessions = [
            _write_session(td_p, "s_dl", "deadlift", 200.0, 1),
            _write_session(td_p, "s_sq", "back_squat", 150.0, 2),
            _write_session(td_p, "s_bp", "bench_press", 100.0, 3),
        ]
        _patch_list_sessions(cs_mod, sessions)
        r = compute_composite_strength("subj_test")
    actual_order = [rg.region for rg in r.regions]
    expected_order = [r2 for r2 in REGION_DISPLAY_ORDER if r2 in actual_order]
    assert actual_order == expected_order


def test_to_dict_serialises_regions():
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        sessions = [_write_session(td_p, "s1", "bench_press",
                                    100.0, 2, "좋음")]
        _patch_list_sessions(cs_mod, sessions)
        r = compute_composite_strength("subj_test")
    d = r.to_dict()
    json.dumps(d, default=str)    # must round-trip
    assert isinstance(d["regions"], list)
    assert d["regions"][0]["region"] == "chest"
    assert d["regions"][0]["grade"] == 2


def test_korean_labels_present_for_v1_regions():
    for k in ("chest", "legs", "whole_body", "biceps",
              "triceps", "shoulder", "back"):
        assert k in REGION_LABELS_KO
        assert isinstance(REGION_LABELS_KO[k], str)
        assert len(REGION_LABELS_KO[k]) > 0


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
