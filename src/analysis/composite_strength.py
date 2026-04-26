"""
Composite strength assessment across multiple strength_3lift sessions
(Phase V3).

Aggregates the latest grade per body region for one subject — bench /
squat / deadlift each cover one of the seven regions in the planning
PDF (chest / legs / whole_body), and the remaining four (biceps,
triceps, shoulder, back) are populated when those exercises are added
in a later phase.

Pipeline:
    1. DB query: list strength_3lift sessions for subject_id, ordered
       by session_date descending.
    2. For each session, read result.json → exercise + grade.
    3. Take the FIRST result per region (latest by date order). This
       gives "current ability" rather than personal-best.
    4. Feed the per-region grades into ``strength_norms.composite_score``
       which weights per-region grade × population weight, normalised
       to the measured regions (so a partial 3-of-7 session still
       produces a sensible 0-100 score).

The function is on-demand: called at report-render time, not
persisted in result.json. This keeps the composite always reflecting
the latest data — adding a new bench session today automatically
updates the composite the next time the trainer opens the report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.analysis.strength_norms import (
    composite_score, EXERCISE_REGION, REGION_WEIGHTS,
    GRADE_LABELS, GRADE_PERCENT, VALID_EXERCISES,
)


# ────────────────────────────────────────────────────────────────────────────
# Region display labels (Korean) — mirrors the planning PDF's table of
# weights. Used in the diagram + per-region bar chart.
# ────────────────────────────────────────────────────────────────────────────
REGION_LABELS_KO: dict[str, str] = {
    "biceps":     "팔 (이두)",
    "triceps":    "팔 (삼두)",
    "shoulder":   "어깨",
    "chest":      "가슴",
    "legs":       "하체",
    "back":       "등",
    "whole_body": "전신",
}

# Display order (top-down, matches the body diagram layout).
REGION_DISPLAY_ORDER: tuple[str, ...] = (
    "shoulder", "chest", "biceps", "triceps", "back",
    "whole_body", "legs",
)


# ────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class RegionalGrade:
    """One region's contribution to the composite — sourced from a
    specific past session. ``grade`` is the 1-7 unified scale."""
    region:        str                    # 'chest' / 'legs' / 'whole_body' / etc.
    region_label:  str                    # Korean display label
    exercise:      str                    # 'bench_press' / 'back_squat' / 'deadlift'
    one_rm_kg:     float
    grade:         int                    # 1-7
    grade_label:   str                    # Korean
    weight_points: int                    # population weight (REGION_WEIGHTS)
    session_id:    str                    # source session
    session_date:  str                    # ISO
    measured:      bool = True            # always True for this dataclass


@dataclass
class CompositeStrength:
    """Multi-region aggregate for one subject."""
    subject_id:        str
    subject_name:      Optional[str] = None

    # Per-region (only measured regions populated; missing list below)
    regions:           list = field(default_factory=list)   # list[RegionalGrade]

    # Composite (V1-B composite_score output)
    composite_score_pct: float = 0.0
    composite_grade:     int   = 7        # default = lowest until measured
    composite_label:     str   = "심각"

    # Coverage
    n_measured:        int = 0
    n_total_regions:   int = 7
    measured_regions:  list = field(default_factory=list)   # list[str]
    missing_regions:   list = field(default_factory=list)   # list[str]
    coverage_weight_pct: float = 0.0       # sum of measured weights / 100

    # Diagnostics
    skipped_reason:    Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regions"] = [asdict(r) for r in self.regions]
        return d


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────
def compute_composite_strength(subject_id: str,
                                subject_name: Optional[str] = None,
                                ) -> CompositeStrength:
    """Aggregate latest strength_3lift grade per region for ``subject_id``.

    Returns an empty (skipped_reason populated) ``CompositeStrength``
    when the subject has no graded strength_3lift sessions yet — the
    report section uses ``skipped_reason`` to render a helpful
    "측정된 항목 없음" placeholder.
    """
    out = CompositeStrength(
        subject_id=subject_id, subject_name=subject_name,
        n_total_regions=len(REGION_WEIGHTS),
        missing_regions=list(REGION_WEIGHTS.keys()),
    )
    sessions = _list_strength_sessions(subject_id)
    if not sessions:
        out.skipped_reason = "이 피험자의 strength_3lift 세션이 없습니다"
        return out

    seen_regions: set[str] = set()
    regions: list[RegionalGrade] = []
    for sess in sessions:
        if not sess.session_dir:
            continue
        meta_p   = Path(sess.session_dir) / "session.json"
        result_p = Path(sess.session_dir) / "result.json"
        if not (meta_p.exists() and result_p.exists()):
            continue
        try:
            meta   = json.loads(meta_p.read_text(encoding="utf-8"))
            result = json.loads(result_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # The dispatcher writes ``{"result": {...}, "error": ...}``.
        # Failed analyses have ``result = None`` + ``error`` set — we
        # skip those (the next session for the same exercise might
        # have succeeded).
        result_inner = result.get("result")
        if result_inner is None:
            result_inner = result if "exercise" in result else {}
        if not isinstance(result_inner, dict):
            continue
        exercise = result_inner.get("exercise") or meta.get("exercise")
        if exercise not in VALID_EXERCISES:
            continue
        region = EXERCISE_REGION.get(exercise)
        if region is None or region in seen_regions:
            continue
        grade = result_inner.get("grade")
        if grade is None or not (1 <= int(grade) <= 7):
            # Session exists but didn't grade (e.g. missing sex/age).
            # Skip — let the next session for that exercise fill in.
            continue

        regions.append(RegionalGrade(
            region=region,
            region_label=REGION_LABELS_KO.get(region, region),
            exercise=exercise,
            one_rm_kg=float(result_inner.get("best_1rm_kg") or 0.0),
            grade=int(grade),
            grade_label=str(result_inner.get("grade_label")
                            or GRADE_LABELS.get(int(grade), "")),
            weight_points=REGION_WEIGHTS.get(region, 0),
            session_id=sess.id,
            session_date=sess.session_date,
        ))
        seen_regions.add(region)

    if not regions:
        out.skipped_reason = (
            "분석된 strength_3lift 세션은 있으나 등급이 산출된 결과가 "
            "없습니다 (피험자 성별/생년월일 누락 가능)"
        )
        return out

    # V1-B composite_score handles the partial-coverage normalisation
    # (sums the measured-region weights, not assumed 100).
    region_grades = {r.region: r.grade for r in regions}
    composite = composite_score(region_grades)

    measured_set    = set(region_grades.keys())
    missing_set     = set(REGION_WEIGHTS.keys()) - measured_set
    out.regions             = sorted(
        regions, key=lambda r: REGION_DISPLAY_ORDER.index(r.region)
        if r.region in REGION_DISPLAY_ORDER else 99)
    out.composite_score_pct = composite["score_pct"]
    out.composite_grade     = composite["composite_grade"]
    out.composite_label     = composite["composite_label"]
    out.n_measured          = composite["n_regions"]
    out.measured_regions    = list(measured_set)
    out.missing_regions     = list(missing_set)
    out.coverage_weight_pct = composite["weighted_total"]
    return out


# ────────────────────────────────────────────────────────────────────────────
# DB helper
# ────────────────────────────────────────────────────────────────────────────
def _list_strength_sessions(subject_id: str) -> list:
    """Latest-first list of strength_3lift sessions for a subject."""
    try:
        from src.db.models import list_sessions
        return list_sessions(subject_id=subject_id,
                              test_type="strength_3lift")
    except Exception:
        # DB unavailable — return empty so caller renders the
        # "no sessions" placeholder rather than crashing.
        return []
