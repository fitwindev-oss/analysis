"""
Unit tests for src/analysis/strength_3lift.py (Phase V1-F).

These build a synthetic session directory (tempfile) with a forces.csv +
session.json that mimics what the recorder would write, then run the
analyzer and assert on per-set rep detection, 1RM estimates, and the
final grade lookup.

Run from project root:
    python tests/test_strength_3lift_analysis.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.strength_3lift import (
    analyze_strength_3lift, _compute_age_years, StrengthResult,
)


# ────────────────────────────────────────────────────────────────────────────
# Synthetic session builder
# ────────────────────────────────────────────────────────────────────────────
def _synth_set_signal(n_reps: int, fs: float = 100.0,
                       rest_s: float = 0.5, rep_s: float = 2.0,
                       rom_mm: float = 400.0,
                       baseline_mm: float = 1500.0) -> np.ndarray:
    """Build a synthetic encoder displacement trace for one set.

    Each rep is a smooth down-and-up cycle (cosine) of total duration
    ``rep_s`` seconds, separated by ``rest_s`` of flat baseline.
    Returns a (N,) ndarray in mm where high = top of bar.
    """
    n_rest = int(rest_s * fs)
    n_rep  = int(rep_s * fs)
    parts: list[np.ndarray] = []
    # Start with a 1-s pre-roll at top so the detector's 90th percentile
    # picks the right "top" reference.
    parts.append(np.full(int(1.0 * fs), baseline_mm))
    for _ in range(n_reps):
        # cosine arc: starts at top, dips to top - rom, back to top
        phase = np.linspace(0, 2 * np.pi, n_rep)
        cycle = baseline_mm - 0.5 * rom_mm * (1 - np.cos(phase))
        parts.append(cycle)
        parts.append(np.full(n_rest, baseline_mm))
    # Trail with a 1 s flat segment so the detector closes the last rep.
    parts.append(np.full(int(1.0 * fs), baseline_mm))
    return np.concatenate(parts)


def _build_synth_session(tmpdir: Path,
                          sex: str = "M", age_years: int = 30,
                          bw_kg: float = 80.0,
                          exercise: str = "bench_press",
                          load_kg: float = 80.0,
                          set_reps: tuple[int, ...] = (8, 10, 8),
                          warmup: bool = False,
                          use_bodyweight_load: bool = False,
                          fs: float = 100.0) -> Path:
    """Construct a fake session directory under ``tmpdir`` and return
    the path. Includes forces.csv + session.json with set boundaries.

    Each "set" is a synthetic encoder window of ``set_reps[i]`` reps;
    sets are concatenated with a ``rest_s = 30 s`` gap of baseline.
    """
    sd = tmpdir / "strength_3lift_synth"
    sd.mkdir(parents=True, exist_ok=True)

    rest_s = 30.0
    rom = 400.0
    baseline = 1500.0
    rep_s = 2.0
    intra_rest_s = 0.5

    sets_meta: list[dict] = []
    enc_segments: list[np.ndarray] = []
    t_cursor = 0.0
    for i, n_reps in enumerate(set_reps):
        sig = _synth_set_signal(
            n_reps, fs=fs, rest_s=intra_rest_s, rep_s=rep_s, rom_mm=rom,
            baseline_mm=baseline)
        seg_dur = len(sig) / fs
        is_warmup = warmup and (i == 0)
        sets_meta.append({
            "set_idx":   i,
            "t_start_s": round(t_cursor, 4),
            "t_end_s":   round(t_cursor + seg_dur, 4),
            "warmup":    is_warmup,
            "load_kg":   load_kg,
            "exercise":  exercise,
        })
        enc_segments.append(sig)
        t_cursor += seg_dur
        if i < len(set_reps) - 1:
            # Inter-set rest of flat baseline (no reps).
            rest_seg = np.full(int(rest_s * fs), baseline)
            enc_segments.append(rest_seg)
            t_cursor += rest_s

    enc1_full = np.concatenate(enc_segments)
    n_total = len(enc1_full)
    t_full  = np.arange(n_total) / fs

    # Build forces.csv: minimum schema the loader needs.
    fcsv = sd / "forces.csv"
    rec_start_wall = 1700000000.0
    with open(fcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_ns", "t_wall",
            "b1_tl_N", "b1_tr_N", "b1_bl_N", "b1_br_N",
            "b2_tl_N", "b2_tr_N", "b2_bl_N", "b2_br_N",
            "enc1_mm", "enc2_mm",
            "total_n", "cop_world_x_mm", "cop_world_y_mm",
            "on_plate",
        ])
        for i in range(n_total):
            t_wall = rec_start_wall + t_full[i]
            # Approximate a "subject standing on plate" force pattern:
            # bw constant + small variation. Force is dummy but the
            # analyzer doesn't use it for rep detection (only encoder).
            tot = bw_kg * 9.80665
            per_corner = tot / 4
            w.writerow([
                int(t_full[i] * 1e9), f"{t_wall:.6f}",
                f"{per_corner:.3f}", f"{per_corner:.3f}",
                f"{per_corner:.3f}", f"{per_corner:.3f}",
                "0.0", "0.0", "0.0", "0.0",
                f"{enc1_full[i]:.3f}", "0.0",
                f"{tot:.3f}", "200.0", "200.0",
                "1",
            ])

    # session.json with subject context + set boundaries.
    bd = (datetime(2026, 1, 1) - timedelta(days=age_years * 365 + 30)
          ).strftime("%Y-%m-%d")
    meta = {
        "test":               "strength_3lift",
        "exercise":           exercise,
        "n_sets":             len(set_reps),
        "target_reps":        12,
        "rest_s":             rest_s,
        "warmup_set":         warmup,
        "subject_kg":         bw_kg,
        "subject_birthdate":  bd,
        "subject_sex":        sex,
        "record_start_wall_s": rec_start_wall,
        "record_start_iso":    "2026-04-27T10:00:00",
        "sets":               sets_meta,
        # V1.5 — bodyweight contribution flag.
        "strength_use_bw_load": use_bodyweight_load,
    }
    (sd / "session.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return sd


# ────────────────────────────────────────────────────────────────────────────
# _compute_age_years helper
# ────────────────────────────────────────────────────────────────────────────
def test_compute_age_years_simple():
    """Birthdate 1990-05-21, ref 2020-06-01 → 30 years."""
    assert _compute_age_years("1990-05-21", "2020-06-01") == 30


def test_compute_age_years_birthday_not_yet_reached():
    """Birthdate 1990-12-15, ref 2020-06-01 → 29 (birthday not yet)."""
    assert _compute_age_years("1990-12-15", "2020-06-01") == 29


def test_compute_age_years_birthday_today():
    """Birthdate 1990-06-01, ref 2020-06-01 → 30 (just reached)."""
    assert _compute_age_years("1990-06-01", "2020-06-01") == 30


def test_compute_age_years_none_birthdate():
    assert _compute_age_years(None, "2020-06-01") is None
    assert _compute_age_years("", "2020-06-01") is None


def test_compute_age_years_bad_format():
    """Malformed birthdate should return None, not crash."""
    assert _compute_age_years("1990/05/21", "2020-06-01") is None
    assert _compute_age_years("not-a-date", "2020-06-01") is None


def test_compute_age_years_iso_with_time():
    """ref_iso may include a 'T' time component."""
    assert _compute_age_years("1990-05-21", "2020-06-01T15:30:00") == 30


# ────────────────────────────────────────────────────────────────────────────
# End-to-end: analyze a synthetic session
# ────────────────────────────────────────────────────────────────────────────
def test_analyze_basic_3_sets_male_30y_bench():
    """80 kg male, 30 y, bench press 80 kg × (8, 10, 8). Best 1RM
    should come from the 10-rep set (highest predicted 1RM)."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(
            Path(td), sex="M", age_years=30, bw_kg=80.0,
            exercise="bench_press", load_kg=80.0,
            set_reps=(8, 10, 8), warmup=False)
        r = analyze_strength_3lift(sd)

    assert isinstance(r, StrengthResult)
    assert r.exercise == "bench_press"
    assert r.region == "chest"
    assert r.sex == "M"
    assert r.age == 30
    assert r.bw_kg == 80.0
    assert r.n_sets == 3
    assert r.n_working_sets == 3
    # All sets should detect their reps. Synthetic data is clean so
    # exact match is reasonable.
    assert r.sets[0].n_reps == 8
    assert r.sets[1].n_reps == 10
    assert r.sets[2].n_reps == 8
    # Best 1RM is from the 10-rep set.
    assert r.best_set_idx == 1
    # 80 kg × 10 reps ensemble: ~107 kg
    assert 100.0 < r.best_1rm_kg < 115.0
    # Grade should be present (subject context complete).
    assert r.grade is not None
    assert 1 <= r.grade <= 7
    assert r.skipped_grade_reason is None


def test_analyze_warmup_excluded_from_best_1rm():
    """Set 0 is warmup (40 kg × 12) — should not contribute. The 80 kg
    × 10 set should win."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(
            Path(td), sex="M", age_years=30, bw_kg=80.0,
            exercise="bench_press", load_kg=80.0,    # all sets at 80
            set_reps=(12, 10, 8), warmup=True)
        r = analyze_strength_3lift(sd)

    # Set 0 is warmup — flag preserved
    assert r.sets[0].warmup is True
    assert r.sets[1].warmup is False
    assert r.sets[2].warmup is False
    assert r.n_working_sets == 2
    # best_set_idx must NOT be 0 (warmup excluded)
    assert r.best_set_idx in (1, 2)


def test_analyze_grade_skipped_when_sex_missing():
    """No subject_sex → grade is None with a clear reason."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(Path(td))
        # Mutate the meta to remove sex
        meta_p = sd / "session.json"
        meta = json.loads(meta_p.read_text())
        meta["subject_sex"] = None
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        r = analyze_strength_3lift(sd)

    assert r.grade is None
    assert r.skipped_grade_reason is not None
    assert "sex" in r.skipped_grade_reason.lower()
    # Per-set 1RM should still be computed (only the GRADE depends on sex).
    assert not math.isnan(r.best_1rm_kg)


def test_analyze_grade_skipped_when_birthdate_missing():
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(Path(td))
        meta_p = sd / "session.json"
        meta = json.loads(meta_p.read_text())
        meta["subject_birthdate"] = None
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        r = analyze_strength_3lift(sd)

    assert r.grade is None
    assert "birthdate" in r.skipped_grade_reason.lower()


def test_analyze_grade_skipped_when_no_reps_detected():
    """Encoder data is flat (zero reps everywhere) → no working set,
    no 1RM, no grade."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(Path(td))
        # Overwrite forces.csv with flat encoder data
        fcsv = sd / "forces.csv"
        rows = list(csv.DictReader(open(fcsv, "r")))
        with open(fcsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            for row in rows:
                row["enc1_mm"] = "1500.0"
                w.writerow(row)
        r = analyze_strength_3lift(sd)

    # Per-set: zero reps each, error noted
    for s in r.sets:
        assert s.n_reps == 0
        assert s.error == "no reps detected"
    assert math.isnan(r.best_1rm_kg)
    assert r.grade is None
    assert "no usable working set" in r.skipped_grade_reason


def test_analyze_rejects_non_strength_session():
    """A CMJ session.json should be refused with a clear ValueError."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(Path(td))
        meta_p = sd / "session.json"
        meta = json.loads(meta_p.read_text())
        meta["test"] = "cmj"
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        try:
            analyze_strength_3lift(sd)
            assert False, "should reject non-strength session"
        except ValueError as e:
            assert "strength_3lift" in str(e)


def test_analyze_rejects_unknown_exercise():
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(Path(td))
        meta_p = sd / "session.json"
        meta = json.loads(meta_p.read_text())
        meta["exercise"] = "leg_press"
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        try:
            analyze_strength_3lift(sd)
            assert False, "should reject leg_press"
        except ValueError as e:
            assert "exercise" in str(e).lower()


def test_to_dict_round_trip():
    """StrengthResult.to_dict() must produce a JSON-serialisable nested
    dict so the dispatcher can persist it."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(Path(td))
        r = analyze_strength_3lift(sd)
    d = r.to_dict()
    # Sanity: keys present
    assert "exercise" in d
    assert "sets" in d
    assert isinstance(d["sets"], list)
    # Each set entry is a dict with reps as a list
    if d["sets"]:
        s0 = d["sets"][0]
        assert "n_reps" in s0
        assert "reps" in s0
        assert isinstance(s0["reps"], list)
    # JSON-serialisability — round-trip through dumps + loads
    s = json.dumps(d, default=str)
    json.loads(s)


def test_dispatcher_routes_strength_3lift():
    """The analysis dispatcher must include strength_3lift after V1-F."""
    from src.analysis.dispatcher import _ANALYZERS, _register_lazy
    _register_lazy()
    assert "strength_3lift" in _ANALYZERS


# ────────────────────────────────────────────────────────────────────────────
# V1.5 — bodyweight contribution flag end-to-end
# ────────────────────────────────────────────────────────────────────────────
def test_analyze_bw_load_disabled_does_not_modify_load():
    """Without use_bodyweight_load, effective load == bar weight."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(
            Path(td), exercise="back_squat", load_kg=80.0,
            set_reps=(10, 10), use_bodyweight_load=False, warmup=False)
        r = analyze_strength_3lift(sd)
    assert r.use_bodyweight_load is False
    assert r.bw_factor == 0.0
    for s in r.sets:
        assert s.effective_load_kg == s.load_kg, (
            f"set {s.set_idx} load mismatch with BW disabled")


def test_analyze_bw_load_squat_adds_85pct():
    """BW on squat: 60 kg bar × 0.85 × 80 kg subject → effective = 128 kg."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(
            Path(td), bw_kg=80.0, exercise="back_squat",
            load_kg=60.0, set_reps=(10, 10),
            use_bodyweight_load=True, warmup=False)
        r = analyze_strength_3lift(sd)
    assert r.use_bodyweight_load is True
    assert abs(r.bw_factor - 0.85) < 1e-9
    for s in r.sets:
        assert abs(s.effective_load_kg - 128.0) < 1e-6


def test_analyze_bw_load_bench_no_addition():
    """BW factor for bench is 0 — flag has no effect."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(
            Path(td), bw_kg=80.0, exercise="bench_press",
            load_kg=70.0, set_reps=(10, 10),
            use_bodyweight_load=True, warmup=False)
        r = analyze_strength_3lift(sd)
    assert r.use_bodyweight_load is True
    assert r.bw_factor == 0.0
    for s in r.sets:
        assert s.effective_load_kg == 70.0


def test_analyze_bw_load_bodyweight_only_squat_50kg_woman():
    """The V1.5 motivating case: 50 kg woman, empty bar, BW squat 12 reps.
    effective = 0.85 × 50 = 42.5 kg → 1RM ≈ 56-58 kg (ensemble)."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(
            Path(td), sex="F", age_years=30, bw_kg=50.0,
            exercise="back_squat",
            load_kg=0.0, set_reps=(12, 12),
            use_bodyweight_load=True, warmup=False)
        r = analyze_strength_3lift(sd)
    # All sets should have effective_load = 42.5
    for s in r.sets:
        assert abs(s.effective_load_kg - 42.5) < 1e-6
    # Best 1RM is in 50-65 kg range (12 reps × 42.5 kg ensemble)
    assert 50.0 <= r.best_1rm_kg <= 65.0
    # Grade still gets computed (subject context complete)
    assert r.grade is not None


def test_analyze_bw_load_high_reps_marked_unreliable():
    """BW squat with 25 reps → reliability = 'unreliable' even though
    1RM is still computed (per user decision (a))."""
    with tempfile.TemporaryDirectory() as td:
        sd = _build_synth_session(
            Path(td), sex="F", age_years=30, bw_kg=50.0,
            exercise="back_squat",
            load_kg=0.0, set_reps=(25, 25),
            use_bodyweight_load=True, warmup=False)
        r = analyze_strength_3lift(sd)
    for s in r.sets:
        assert s.n_reps == 25
        assert s.reliability == "unreliable"
        # 1RM is still produced — not NaN
        assert not math.isnan(s.one_rm_kg)
    # The aggregate best_reliability should also reflect this.
    assert r.best_reliability == "unreliable"


def test_analyze_bw_load_meta_field_round_trip():
    """The strength_use_bw_load meta key is consumed correctly when
    set both ways."""
    with tempfile.TemporaryDirectory() as td:
        sd_off = _build_synth_session(
            Path(td) / "off", exercise="back_squat", load_kg=60.0,
            set_reps=(10,), use_bodyweight_load=False, warmup=False)
        sd_on = _build_synth_session(
            Path(td) / "on", exercise="back_squat", load_kg=60.0,
            set_reps=(10,), use_bodyweight_load=True, warmup=False)
        r_off = analyze_strength_3lift(sd_off)
        r_on  = analyze_strength_3lift(sd_on)
    assert r_off.use_bodyweight_load is False
    assert r_on.use_bodyweight_load is True
    # 1RM with BW on must be larger (effective load is bigger).
    assert r_on.best_1rm_kg > r_off.best_1rm_kg


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
