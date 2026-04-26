"""
Strength 3-lift analysis pipeline (Phase V1-F).

Consumes a multi-set ``strength_3lift`` session (bench / squat / deadlift)
and produces a per-set + summary 1RM grade. Designed to be called by
the dispatcher in the same shape as the other ``analyze_*_file`` modules.

Pipeline:
    1. Load force/encoder timeseries from forces.csv.
    2. Load set boundaries + subject metadata from session.json.
    3. For each set, slice the encoder window and run ``detect_reps`` to
       count repetitions; compute per-rep velocity/power metrics.
    4. Use ``estimate_1rm`` (ensemble of Epley/Brzycki/Lombardi) on
       (load_kg, reps) per set.
    5. Pick the set with the highest predicted 1RM among non-warmup sets
       (``estimate_1rm_from_sets``).
    6. Look up the grade (1-7) from the population norms in
       ``strength_norms.grade_1rm`` using sex × age × bw.

The grade computation is skipped (with a clear flag) when subject
gender or birthdate is missing — this is enforced upstream by the
``tools/migrate_subjects_v1.py`` migration tool, but we degrade
gracefully here too so a partial session still produces useful per-set
output.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.analysis.common import ForceSession, load_force_session
from src.analysis.encoder import (
    RepMetrics, detect_reps, butter_lowpass, numerical_derivative,
)
from src.analysis.one_rm import estimate_1rm, estimate_1rm_from_sets
from src.analysis.strength_norms import (
    grade_1rm, EXERCISE_REGION, GRADE_LABELS, VALID_EXERCISES,
    EXERCISE_BW_FACTOR, effective_load_kg,
)
from src.analysis.multiset_recovery import (
    SetPerformance, RecoveryMetrics, compute_recovery_metrics,
)


# ────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class StrengthSetResult:
    """Per-set analysis: rep count, per-rep metrics, and 1RM estimate.

    ``load_kg`` is the RAW external bar weight as recorded.
    ``effective_load_kg`` is what 1RM estimation actually used:
        - if use_bodyweight_load=False → effective == bar
        - if True → effective = bar + EXERCISE_BW_FACTOR × subject_kg
    Reports surface both so the operator can verify the calculation
    (especially important for bodyweight-only sessions).
    """
    set_idx:           int
    warmup:            bool
    load_kg:           float                 # raw bar weight
    effective_load_kg: float                 # bar + α × BW (V1.5)
    t_start_s:         float
    t_end_s:           float
    n_reps:            int
    reps:              list = field(default_factory=list)   # list of RepMetrics
    one_rm_kg:         float = float("nan")     # estimated 1RM for this set
    one_rm_method:     str = "ensemble"
    reliability:       str = "unreliable"       # excellent / high / medium / low / unreliable
    epley_kg:          float = float("nan")
    brzycki_kg:        float = float("nan")
    lombardi_kg:       float = float("nan")
    error:             Optional[str] = None     # populated when reps detection failed

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reps"] = [asdict(r) for r in self.reps]
        return d


@dataclass
class StrengthResult:
    """Full session analysis with per-set breakdown + composite grade."""
    exercise:       str                     # bench_press / back_squat / deadlift
    region:         str                     # chest / legs / whole_body
    sex:            Optional[str]           # 'M' / 'F' / None
    age:            Optional[int]           # years; None if birthdate missing
    bw_kg:          float
    duration_s:     float                   # full session length

    # Per-set
    n_sets:         int = 0                 # total sets recorded
    n_working_sets: int = 0                 # excluding warmup
    sets:           list = field(default_factory=list)   # list of StrengthSetResult

    # ── Bodyweight contribution (V1.5) ──────────────────────────────────
    use_bodyweight_load: bool = False       # was BW addition enabled?
    bw_factor:           float = 0.0        # α applied to subject_kg
    # When use_bodyweight_load=True and bar_load_kg=0, the entire 1RM
    # input came from bodyweight; report should warn that this is a
    # back-calculation rather than a measured lift.

    # ── ATP-PCr recovery (V2) ───────────────────────────────────────────
    # Per-set performance + FI / PDS / fiber-tendency. None when fewer
    # than 2 working sets had usable encoder data — the recovery
    # subsection of the report degrades gracefully in that case.
    set_perfs: list = field(default_factory=list)   # list[SetPerformance]
    recovery:  Optional[dict] = None                # RecoveryMetrics.to_dict()

    # 1RM aggregate
    best_1rm_kg:    float = float("nan")
    best_set_idx:   Optional[int] = None    # which set produced the best estimate
    best_reliability: str = "unreliable"

    # Grade lookup (None when sex/age missing)
    grade:          Optional[int] = None    # 1-7
    grade_label:    Optional[str] = None    # Korean
    warning:        Optional[str] = None    # 'caution' / 'severe' / None
    thresholds_kg:  Optional[dict] = None   # beginner..elite kg
    ratio_to_elite: Optional[float] = None
    ratio_to_beginner: Optional[float] = None

    # Diagnostics
    skipped_grade_reason: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sets"] = [s.to_dict() if hasattr(s, "to_dict") else s for s in self.sets]
        # V2 — set_perfs is a list[SetPerformance]; asdict already
        # walks dataclass fields, but emit explicitly for clarity so
        # downstream consumers know this is a list of plain dicts.
        d["set_perfs"] = [
            asdict(p) if hasattr(p, "set_idx") else p
            for p in self.set_perfs
        ]
        return d


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _compute_age_years(birthdate_str: Optional[str],
                        ref_iso: Optional[str] = None) -> Optional[int]:
    """Compute age in completed years from a 'YYYY-MM-DD' birthdate.

    ``ref_iso`` is the session date (also 'YYYY-MM-DD' or ISO with time).
    When None, today's date is used. Birthdate or ref parsing failures
    return None — the caller should treat that as "age unknown".
    """
    if not birthdate_str:
        return None
    try:
        bd = datetime.strptime(birthdate_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    if ref_iso:
        try:
            # Accept full ISO ("2026-04-27T...") or date-only.
            ref_str = ref_iso.split("T")[0] if "T" in ref_iso else ref_iso
            ref = datetime.strptime(ref_str, "%Y-%m-%d").date()
        except ValueError:
            ref = datetime.utcnow().date()
    else:
        ref = datetime.utcnow().date()
    years = ref.year - bd.year - (
        (ref.month, ref.day) < (bd.month, bd.day))
    return max(0, int(years))


def _slice_force_window(force: ForceSession,
                         t_start_s: float, t_end_s: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_s, enc1, enc2) restricted to ``[t_start_s, t_end_s]``.

    ``t_s`` is re-zeroed to the slice start so per-set rep detection
    sees timestamps starting at 0 — this is purely cosmetic for the
    detector but matters for any downstream plot.
    """
    mask = (force.t_s >= t_start_s) & (force.t_s <= t_end_s)
    t = force.t_s[mask] - t_start_s
    return t, force.enc1[mask], force.enc2[mask]


def _detect_set_reps(enc_mm: np.ndarray, fs: float,
                      min_rom_mm: float = 100.0) -> int:
    """Count completed reps in a single set's encoder window.

    Lower default ``min_rom_mm`` than the global encoder analyzer (100
    vs 150) because some lifts (bench press, partial-ROM curls) have
    smaller travel than a full squat. The bar dropped during set
    transitions has no rep activity, so the detector won't false-fire.
    Returns 0 on detection failure (e.g., flat-line encoder data).
    """
    if len(enc_mm) < int(fs):    # less than 1 s of data
        return 0
    try:
        reps = detect_reps(enc_mm, fs, min_rom_mm=min_rom_mm)
    except Exception:
        return 0
    return len(reps)


# Gravity for power computation in V2.
_G = 9.80665


def _compute_set_performance(enc_mm: np.ndarray, fs: float,
                              effective_load_kg: float,
                              warmup: bool, set_idx: int,
                              min_rom_mm: float = 100.0) -> SetPerformance:
    """Per-set performance summary for V2 FI/PDS computation.

    Mirrors ``encoder.analyze_encoder`` but scoped to one set window:
        - detect rep boundaries on the encoder trace
        - per-rep mean / peak concentric velocity from the smoothed
          derivative
        - per-rep power = effective_load × (g + a) × v
        - aggregate across reps to per-set means + peaks

    Returns a SetPerformance with all-zeros fields when no reps are
    detected — the recovery metric pipeline will skip empty sets
    gracefully.
    """
    perf = SetPerformance(
        set_idx=int(set_idx), warmup=bool(warmup), n_reps=0)
    if len(enc_mm) < int(fs):
        return perf

    try:
        reps_idx = detect_reps(enc_mm, fs, min_rom_mm=min_rom_mm)
    except Exception:
        return perf
    if not reps_idx:
        return perf

    # Convert to metres and smooth before differentiation. Same constants
    # as analyze_encoder so the values are comparable across the codebase.
    x_m = enc_mm.astype(np.float64) / 1000.0
    x_s = butter_lowpass(x_m, 6.0, fs)
    v   = numerical_derivative(x_s, fs)             # m/s
    a   = numerical_derivative(v,   fs)             # m/s^2
    # Force on bar (N) = m × (g + a). For our purposes m = effective load
    # (bar + α × BW for V1.5). When use_bw is off, this collapses to bar.
    f_n = float(effective_load_kg) * (_G + a)
    power = f_n * v                                  # W

    mean_v_rep:   list[float] = []
    peak_v_rep:   list[float] = []
    mean_p_rep:   list[float] = []
    peak_p_rep:   list[float] = []
    rom_per_rep:  list[float] = []
    work_per_rep: list[float] = []

    for (i_start, i_bot, i_end) in reps_idx:
        # Concentric segment = bottom → top.
        v_con = v[i_bot:i_end + 1]
        if v_con.size == 0:
            continue
        v_pos = np.maximum(v_con, 0.0)
        mean_v_rep.append(float(v_pos.mean()))
        peak_v_rep.append(float(v_con.max()))
        # Power during concentric: clamp to ≥0 (eccentric phase has
        # negative velocity → negative power, which we exclude when
        # averaging "useful" power output).
        p_con = power[i_bot:i_end + 1]
        p_pos = np.maximum(p_con, 0.0)
        mean_p_rep.append(float(p_pos.mean()))
        peak_p_rep.append(float(p_con.max()))
        # ROM in mm + work approximated as load × ROM × g (J).
        rom = float(enc_mm[i_start:i_end + 1].max()
                    - enc_mm[i_start:i_end + 1].min())
        rom_per_rep.append(rom)
        work_per_rep.append(
            float(effective_load_kg) * (rom / 1000.0) * _G)

    n = len(mean_p_rep)
    if n == 0:
        return perf

    perf.n_reps             = n
    perf.mean_velocity_m_s  = float(np.mean(mean_v_rep))
    perf.peak_velocity_m_s  = float(np.mean(peak_v_rep))
    perf.mean_power_w       = float(np.mean(mean_p_rep))
    perf.peak_power_w       = float(np.mean(peak_p_rep))
    perf.rom_mm             = float(np.mean(rom_per_rep))
    perf.total_work_j       = float(np.sum(work_per_rep))
    return perf


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────
def analyze_strength_3lift(session_dir: str | Path) -> StrengthResult:
    """Analyse a multi-set strength_3lift session.

    Reads forces.csv + session.json from ``session_dir``, slices each
    recorded set, counts reps, estimates per-set 1RM, picks the best
    working set, and looks up the grade from the population norms.

    Raises:
        FileNotFoundError: forces.csv or session.json missing
        ValueError:        meta is not a strength_3lift session, or
                           ``exercise`` is not one of the 3 supported lifts
    """
    sd = Path(session_dir)
    meta_p = sd / "session.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"missing: {meta_p}")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))

    if meta.get("test") != "strength_3lift":
        raise ValueError(
            f"not a strength_3lift session: test={meta.get('test')!r}")
    exercise = meta.get("exercise")
    if exercise not in VALID_EXERCISES:
        raise ValueError(
            f"unsupported exercise for strength_3lift: {exercise!r} "
            f"(expected one of {VALID_EXERCISES})")

    # Subject context for the grade lookup. Sex / birthdate became
    # first-class fields in session.json with the V1-bugfix; older
    # sessions only have ``subject_id`` so we fall back to a DB lookup
    # for those. ``subject_kg`` has always been in the meta.
    sex = (meta.get("subject_sex") or meta.get("sex")
           or meta.get("gender"))
    bd_str = (meta.get("subject_birthdate") or meta.get("birthdate"))
    bw_kg = float(meta.get("subject_kg") or 0.0)
    if (not sex or not bd_str) and meta.get("subject_id"):
        try:
            from src.db.models import get_subject
            row = get_subject(meta["subject_id"])
            if row is not None:
                sex = sex or getattr(row, "gender", None)
                bd_str = bd_str or getattr(row, "birthdate", None)
                if not bw_kg:
                    bw_kg = float(getattr(row, "weight_kg", 0.0) or 0.0)
        except Exception:
            # DB unavailable / schema changed — proceed with whatever
            # session.json had. The grade step will degrade gracefully.
            pass
    rec_start_iso = meta.get("record_start_iso") or meta.get("created_at")
    age = _compute_age_years(bd_str, rec_start_iso)

    # Force / encoder timeseries
    force = load_force_session(sd)

    # Bodyweight contribution flag — V1.5. The flag's canonical home in
    # session.json is ``strength_use_bw_load``; older sessions written
    # before V1.5 are missing the key entirely (treated as False).
    use_bw = bool(meta.get("strength_use_bw_load", False))
    bw_factor = EXERCISE_BW_FACTOR.get(exercise, 0.0) if use_bw else 0.0

    # Per-set boundary records written by the recorder (V1-D).
    set_records: list[dict] = list(meta.get("sets") or [])
    set_results: list[StrengthSetResult] = []
    set_perfs:   list[SetPerformance] = []          # V2 per-set performance
    sets_for_1rm: list[dict] = []   # (load, reps, warmup) for estimate_1rm_from_sets

    for rec in set_records:
        t0, t1 = float(rec["t_start_s"]), float(rec["t_end_s"])
        bar_kg = float(rec.get("load_kg") or 0.0)
        warmup = bool(rec.get("warmup", False))
        # V1.5 — effective load includes BW fraction when enabled.
        # ``effective_load_kg`` from strength_norms applies the factor.
        eff_kg = effective_load_kg(
            exercise=exercise, bar_load_kg=bar_kg,
            subject_kg=bw_kg, use_bodyweight=use_bw)

        _, enc1_w, enc2_w = _slice_force_window(force, t0, t1)
        # The bar encoder is enc1 in the standard rig (left side). When
        # enc1 is flat-zero (channel disconnected) fall back to enc2.
        bar_signal = enc1_w
        if (np.ptp(bar_signal) < 5.0) and (np.ptp(enc2_w) >= 5.0):
            bar_signal = enc2_w
        # V2 — full per-set performance summary (ROM, velocity, power,
        # work). Reps count is reused from this since detect_reps
        # already runs inside _compute_set_performance.
        perf = _compute_set_performance(
            bar_signal, force.fs, eff_kg, warmup, int(rec["set_idx"]))
        set_perfs.append(perf)
        n_reps = perf.n_reps

        srec = StrengthSetResult(
            set_idx=int(rec["set_idx"]),
            warmup=warmup,
            load_kg=bar_kg,
            effective_load_kg=eff_kg,
            t_start_s=t0,
            t_end_s=t1,
            n_reps=n_reps,
        )

        # 1RM input is the EFFECTIVE load — that's what the formulas
        # actually need to be physically meaningful.
        if n_reps > 0 and eff_kg > 0:
            est = estimate_1rm(eff_kg, n_reps, method="ensemble")
            srec.one_rm_kg     = est["one_rm_kg"]
            srec.one_rm_method = est["method"]
            srec.reliability   = est["reliability"]
            srec.epley_kg      = est["epley_kg"]
            srec.brzycki_kg    = est["brzycki_kg"]
            srec.lombardi_kg   = est["lombardi_kg"]
            sets_for_1rm.append({
                "load_kg": eff_kg, "reps": n_reps, "warmup": warmup,
            })
        else:
            if n_reps == 0:
                srec.error = "no reps detected"
            elif eff_kg <= 0:
                srec.error = "load_kg not recorded"

        set_results.append(srec)

    # Aggregate 1RM across non-warmup sets — pick the set with the
    # highest predicted 1RM (fatigue makes later sets under-estimate).
    best_summary = estimate_1rm_from_sets(
        sets_for_1rm, method="ensemble", include_warmup=False)
    best_1rm = best_summary["one_rm_kg"]
    best_set_idx = best_summary.get("chosen_set_idx")
    # ``best_set_idx`` from estimate_1rm_from_sets is the index into the
    # FILTERED list (warmup excluded). Translate back to the original
    # set_idx so callers can highlight the right set. We match on the
    # EFFECTIVE load (what was fed into estimate_1rm) + reps so the
    # match is unambiguous when use_bodyweight_load is on.
    if best_set_idx is not None:
        per_set_list = best_summary["per_set"]
        chosen_load_reps = (per_set_list[best_set_idx]["load_kg"],
                            per_set_list[best_set_idx]["reps"])
        for sr in set_results:
            if (not sr.warmup
                    and abs(sr.effective_load_kg - chosen_load_reps[0]) < 1e-6
                    and sr.n_reps == chosen_load_reps[1]):
                best_set_idx = sr.set_idx
                break
        else:
            best_set_idx = None

    # Grade lookup — gracefully degrade if any required input is missing.
    grade_payload = None
    skipped_reason = None
    sex_norm = sex.upper() if isinstance(sex, str) else None
    if sex_norm not in ("M", "F"):
        skipped_reason = "subject sex missing"
    elif age is None:
        skipped_reason = "subject birthdate missing"
    elif bw_kg <= 0:
        skipped_reason = "subject weight missing"
    elif math.isnan(best_1rm):
        skipped_reason = "no usable working set (no reps detected)"
    else:
        try:
            grade_payload = grade_1rm(
                exercise=exercise, sex=sex_norm, age=age,
                bw_kg=bw_kg, one_rm_kg=best_1rm)
        except Exception as e:
            skipped_reason = f"grade lookup error: {e}"

    result = StrengthResult(
        exercise=exercise,
        region=EXERCISE_REGION[exercise],
        sex=sex_norm,
        age=age,
        bw_kg=bw_kg,
        duration_s=float(force.t_s[-1] - force.t_s[0]) if len(force.t_s) else 0.0,
        use_bodyweight_load=use_bw,
        bw_factor=bw_factor,
        n_sets=len(set_results),
        n_working_sets=sum(1 for r in set_results if not r.warmup),
        sets=set_results,
        best_1rm_kg=best_1rm,
        best_set_idx=best_set_idx,
        best_reliability=(set_results[best_set_idx].reliability
                          if best_set_idx is not None
                          and 0 <= best_set_idx < len(set_results)
                          else "unreliable"),
        skipped_grade_reason=skipped_reason,
    )
    if grade_payload is not None:
        result.grade             = grade_payload["grade"]
        result.grade_label       = grade_payload["label"]
        result.warning           = grade_payload["warning"]
        result.thresholds_kg     = grade_payload["thresholds_kg"]
        result.ratio_to_elite    = grade_payload["ratio_to_elite"]
        result.ratio_to_beginner = grade_payload["ratio_to_beginner"]

    # ── ATP-PCr recovery (V2) ───────────────────────────────────────────
    # Mean concentric power is the primary variable. The recovery
    # pipeline picks the working sets (warmup excluded) and computes
    # FI / PDS / fiber tendency. Returns gracefully with skipped_reason
    # if there are fewer than 2 working sets or no usable encoder data.
    result.set_perfs = set_perfs
    rec_metrics = compute_recovery_metrics(
        set_perfs, variable="mean_power_w")
    result.recovery = rec_metrics.to_dict()
    return result


def analyze_strength_3lift_file(session_dir, **kw) -> StrengthResult:
    """Dispatcher-compatible entry point. Mirrors analyze_*_file naming."""
    return analyze_strength_3lift(session_dir, **kw)
