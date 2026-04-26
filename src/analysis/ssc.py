"""
Stretch-Shortening Cycle (SSC) analysis — CMJ vs SJ comparison
(Phase V4).

Combines a subject's latest Counter-Movement Jump (CMJ) and Squat
Jump (SJ) sessions to assess how effectively they leverage stored
elastic energy through the SSC. Two indices, both jump-height based:

    EUR  (Eccentric Utilization Ratio)
        EUR = h_CMJ / h_SJ
        Ratio of counter-movement-aided height to static-start height.
        > 1.10 = good elastic recoil; ≤ 1.00 = no SSC benefit.

    SSC%
        SSC% = (h_CMJ - h_SJ) / h_SJ × 100
        Percent of jump height attributable to elastic energy.

Five-grade scale per the planning PDF:
    1등급 Elite   EUR ≥ 1.15  /  SSC ≥ 15 %
    2등급 Good    1.10–1.14   /  10–14 %
    3등급 Average 1.05–1.09   /  5–9 %
    4등급 Poor    1.01–1.04   /  1–4 %
    5등급 Risk    ≤ 1.00      /  ≤ 0 %

Dual interpretation (plan §4):
    Lower-body 1RM grade ≤ 2 (좋음 / 엘리트):
        — focus on plyometric / elasticity training (해석2)
        — SSC grade reads as "근력 + 탄성 조화" framing
    Lower-body 1RM grade ≥ 3 (보통 / 나쁨 / 위험):
        — focus on building strength first (해석1)
        — SSC grade reads as "근력 향상이 우선" framing

The 1RM cross-reference is sourced from the latest back_squat
strength_3lift session via composite_strength.compute_composite_strength.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Grade tables — EUR is primary, SSC% gives same grade by construction.
# Each row: (eur_lower_inclusive, ssc_pct_lower_inclusive, grade, label)
# Walked top-down with first-match wins.
# ────────────────────────────────────────────────────────────────────────────
_SSC_GRADES: list[tuple[float, float, int, str]] = [
    (1.15, 15.0, 1, "Elite"),
    (1.10, 10.0, 2, "Good"),
    (1.05,  5.0, 3, "Average"),
    (1.01,  1.0, 4, "Poor"),
    # The 5th tier is a catch-all for everything below grade-4 boundaries.
    (0.0,  -float("inf"), 5, "Risk"),
]


# Two interpretation sets per the plan (§4 page 31).
INTERPRETATION_STRENGTH_FOCUS: dict[int, str] = {
    1: "탄성 활용 능력이 우수합니다. 다음 단계로 근력 수준을 끌어올리면 "
       "기록이 추가로 향상될 여지가 큽니다.",
    2: "탄성 활용 능력이 좋은 편입니다. 근력 수준을 함께 키우면 더 큰 "
       "기록 향상을 기대할 수 있습니다.",
    3: "탄성 활용은 일반적인 수준입니다. 근력 수준을 우선 끌어올리는 "
       "것이 효율적입니다.",
    4: "반동을 거의 활용하지 못하고 있습니다. 근력이 부족한 상태이므로 "
       "근력 향상과 함께 유연성·협응 훈련이 필요합니다.",
    5: "반동·근력 모두 수준 이하입니다. 낙상·운동 부상 위험이 높으므로 "
       "맨몸 운동부터 단계적으로 근력을 끌어올려야 합니다.",
}

INTERPRETATION_ELASTIC_FOCUS: dict[int, str] = {
    1: "플라이오메트릭 훈련의 숙련자 수준입니다. 엘리트 선수 급의 "
       "탄성 활용 능력을 보여주고 있습니다.",
    2: "이상적인 수준입니다. 근력과 탄성이 잘 조화되어 있어 현재의 "
       "프로토콜을 그대로 유지하면 됩니다.",
    3: "반동을 이용하지만 효율은 보통 수준입니다. 근력이 충분하므로 "
       "탄성 훈련(플라이오메트릭)을 추가하면 향상 여지가 있습니다.",
    4: "근력은 충분하지만 반동을 거의 못 쓰고 있습니다. 근육이 뻣뻣하거나 "
       "협응력이 떨어진 상태일 수 있어 유연성/탄성 훈련이 필요합니다.",
    5: "근력은 충분한데 반동 시 오히려 기록이 하락했습니다. 신경근 제어 "
       "실패 또는 아킬레스건 손상 등 부상 위험이 의심됩니다 — 안전 점검이 "
       "필요합니다.",
}


# ────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class SSCResult:
    """Stretch-shortening cycle assessment for one subject."""
    subject_id:        str
    subject_name:      Optional[str] = None

    # Source heights (m)
    cmj_height_m:      Optional[float] = None
    sj_height_m:       Optional[float] = None
    cmj_session_id:    Optional[str] = None
    sj_session_id:     Optional[str] = None
    cmj_session_date:  Optional[str] = None
    sj_session_date:   Optional[str] = None

    # Indices
    eur:               Optional[float] = None     # CMJ / SJ
    ssc_pct:           Optional[float] = None     # (CMJ-SJ)/SJ × 100

    # Grade (1-5)
    grade:             Optional[int] = None
    grade_label:       Optional[str] = None

    # Cross-reference with lower-body 1RM
    lower_body_1rm_grade: Optional[int] = None    # 1-7 from strength_norms
    interpretation_focus: str = "strength"        # 'strength' or 'elastic'
    interpretation:    Optional[str] = None       # Korean one-liner

    # Diagnostics
    skipped_reason:    Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ────────────────────────────────────────────────────────────────────────────
# Public formulas
# ────────────────────────────────────────────────────────────────────────────
def eccentric_utilization_ratio(cmj_height_m: float,
                                 sj_height_m: float) -> float:
    """EUR = h_CMJ / h_SJ. NaN when SJ height is non-positive."""
    if sj_height_m is None or sj_height_m <= 0:
        return float("nan")
    return float(cmj_height_m) / float(sj_height_m)


def ssc_contribution_pct(cmj_height_m: float,
                          sj_height_m: float) -> float:
    """SSC% = (h_CMJ − h_SJ) / h_SJ × 100. NaN when SJ ≤ 0."""
    if sj_height_m is None or sj_height_m <= 0:
        return float("nan")
    return (float(cmj_height_m) - float(sj_height_m)) / float(sj_height_m) * 100.0


def grade_ssc(eur: float) -> tuple[int, str]:
    """5-grade lookup based on EUR. Returns (grade, English label)."""
    if eur is None or eur != eur:    # NaN check w/o numpy
        return 5, "Risk"
    e = float(eur)
    for eur_lo, _ssc_lo, grade, label in _SSC_GRADES:
        if e >= eur_lo:
            return grade, label
    return 5, "Risk"


def interpret_ssc(grade: int,
                  lower_body_1rm_grade: Optional[int]) -> tuple[str, str]:
    """Pick interpretation set based on lower-body 1RM grade.

    Per plan §4: 1RM grade ≤ 2 → "elastic focus" framing (subject is
    already strong, so reframe SSC as plyometric refinement);
    otherwise → "strength focus" framing (build base strength first).

    Returns ``(focus, korean_text)`` where focus is ``'strength'`` or
    ``'elastic'``.
    """
    if (lower_body_1rm_grade is not None
            and 1 <= int(lower_body_1rm_grade) <= 2):
        return ("elastic",
                INTERPRETATION_ELASTIC_FOCUS.get(int(grade), ""))
    return ("strength",
            INTERPRETATION_STRENGTH_FOCUS.get(int(grade), ""))


# ────────────────────────────────────────────────────────────────────────────
# Subject-level aggregator
# ────────────────────────────────────────────────────────────────────────────
def compute_ssc(subject_id: str,
                subject_name: Optional[str] = None) -> SSCResult:
    """Aggregate latest SJ + CMJ + lower-body 1RM into an SSC report.

    Skipped reasons (populated on SSCResult.skipped_reason):
      - "이 피험자의 CMJ 세션이 없습니다"
      - "이 피험자의 SJ 세션이 없습니다"
      - "최신 CMJ / SJ 결과에 jump_height_m_impulse가 없습니다"
    """
    out = SSCResult(subject_id=subject_id, subject_name=subject_name)

    cmj = _latest_jump_height(subject_id, test_type="cmj")
    if cmj is None:
        out.skipped_reason = "이 피험자의 CMJ 세션이 없습니다"
        return out
    sj = _latest_jump_height(subject_id, test_type="sj")
    if sj is None:
        out.skipped_reason = "이 피험자의 SJ 세션이 없습니다"
        return out

    out.cmj_height_m,    out.cmj_session_id,    out.cmj_session_date = cmj
    out.sj_height_m,     out.sj_session_id,     out.sj_session_date  = sj

    if out.cmj_height_m <= 0 or out.sj_height_m <= 0:
        out.skipped_reason = (
            "최신 CMJ/SJ 결과의 점프 높이가 측정되지 않았습니다 "
            "(takeoff/landing 검출 실패)")
        return out

    out.eur     = round(eccentric_utilization_ratio(
        out.cmj_height_m, out.sj_height_m), 4)
    out.ssc_pct = round(ssc_contribution_pct(
        out.cmj_height_m, out.sj_height_m), 2)
    grade, label = grade_ssc(out.eur)
    out.grade = grade
    out.grade_label = label

    # Cross-reference with lower-body (legs / back_squat) 1RM grade.
    out.lower_body_1rm_grade = _lower_body_grade(subject_id)
    focus, text = interpret_ssc(grade, out.lower_body_1rm_grade)
    out.interpretation_focus = focus
    out.interpretation = text
    return out


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers — DB + result.json reading
# ────────────────────────────────────────────────────────────────────────────
def _latest_jump_height(subject_id: str, test_type: str
                         ) -> Optional[tuple[float, str, str]]:
    """Return (height_m, session_id, session_date) for the latest
    CMJ or SJ session of ``subject_id``, or None when no analysed
    session exists."""
    try:
        from src.db.models import list_sessions
        sessions = list_sessions(subject_id=subject_id, test_type=test_type)
    except Exception:
        sessions = []
    for sess in sessions:
        if not sess.session_dir:
            continue
        result_p = Path(sess.session_dir) / "result.json"
        if not result_p.exists():
            continue
        try:
            data = json.loads(result_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        inner = data.get("result")
        if not isinstance(inner, dict):
            continue
        # CMJ analyzer canonical key — same field on both CMJ + SJ
        # because the dispatcher routes both through analyze_cmj_file.
        h = (inner.get("jump_height_m_impulse")
             or inner.get("jump_height_m_flight")
             or inner.get("jump_height_m"))
        if h is None or float(h) <= 0:
            continue
        return float(h), sess.id, sess.session_date
    return None


def _lower_body_grade(subject_id: str) -> Optional[int]:
    """Latest legs-region (back_squat) 1RM grade for the subject.

    Returns None when the subject has no graded back_squat session yet.
    """
    try:
        from src.analysis.composite_strength import (
            compute_composite_strength,
        )
    except Exception:
        return None
    composite = compute_composite_strength(subject_id=subject_id)
    for rg in composite.regions:
        if rg.region == "legs":
            return int(rg.grade)
    return None
