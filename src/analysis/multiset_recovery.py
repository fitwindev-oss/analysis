"""
ATP-PCr recovery resilience metrics for multi-set strength sessions
(Phase V2).

Quantifies how much the subject's per-set performance drops across a
3-5 set protocol with fixed inter-set rest. Lower drops indicate
better PCr re-synthesis efficiency and lactate buffering capacity.

Two complementary indices, both expressed as percentages:

  Fatigue Index (FI)
      FI(%) = (Set_1 - Set_N) / Set_1 × 100
      Sensitive to the *acute* drop between the first and last sets.
      Catches subjects who hold up well early then collapse, or
      vice-versa.

  Performance Decrement Score (PDS, Glaister 2008)
      PDS(%) = (1 - Σ_i Set_i / (Set_1 × N)) × 100
      Considers ALL sets — more statistically reliable for comparing
      sessions and tracking longitudinal change. PDS = 0 when every
      set matches the first; PDS = 100 when subsequent sets all
      flat-line.

Both indices are computed on the per-set values of one performance
variable. The default is **mean concentric power (W)** — captures
both load and velocity, requires encoder data. Fallback variables:

  - peak_concentric_power_w
  - mean_concentric_velocity_m_s
  - n_reps
  - total_work_j

Phase V2 wires this into the strength_3lift analyzer; FI/PDS are
populated on StrengthResult and rendered in the report.

Population grade tables transcribed from the planning PDF (Korean +
English-language exercise-physiology heuristics — to be validated on
domestic data per the V1 plan note "실험을 통해 변인 정리 필요").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Grade tables (1-5 scale, 1 = elite, 5 = at-risk)
# ────────────────────────────────────────────────────────────────────────────
# Each entry: (upper_bound_pct, grade, label, interpretation_ko)
# Lookup by bisecting on upper_bound_pct against the measured value.

_FI_GRADES: list[tuple[float, int, str, str]] = [
    (15.0, 1, "엘리트",
     "PCr 재합성 효율이 극도로 높음. 젖산 완충 능력 탁월."),
    (25.0, 2, "좋음",
     "우수한 회복력. 일반적인 숙련된 헬스 동호인 상위 레벨."),
    (40.0, 3, "보통",
     "일반 성인의 평균치. 30초 휴식 시 급격한 대사 산물 축적으로 "
     "수행력 저하가 뚜렷함."),
    (60.0, 4, "나쁨",
     "유산소성 베이스 부족 및 ATP-PCr 시스템 효율 저하. "
     "마지막 세트에 급격히 무너짐."),
    (float("inf"), 5, "위험",
     "운동 중단 권고. 신경근 피로도가 한계 초과. "
     "부상 위험 높음 (자세 붕괴)."),
]

_PDS_GRADES: list[tuple[float, int, str, str]] = [
    (8.0,  1, "엘리트",
     "세트 간 기복이 거의 없음. 놀라운 대사적 안정성 (Metabolic Stability)."),
    (15.0, 2, "좋음",
     "세트가 진행될수록 감소하나, 운동 강도를 끝까지 유지할 수 있는 수준."),
    (25.0, 3, "보통",
     "후반 세트에서 수행 횟수나 속도가 눈에 띄게 줄어듦."),
    (40.0, 4, "나쁨",
     "전체 운동량 (Volume) 확보 실패. 훈련 효과보다 피로 누적이 더 큼."),
    (float("inf"), 5, "위험",
     "대사적 탈진 (Exhaustion). 회복 시스템이 작동하지 않음."),
]


# Color hints for the report renderer (1-5 → colour).
GRADE_COLORS_RECOVERY: dict[int, str] = {
    1: "#26A69A",   # 엘리트
    2: "#9CCC65",   # 좋음
    3: "#FBC02D",   # 보통
    4: "#FFA726",   # 나쁨
    5: "#EF5350",   # 위험
}


# ────────────────────────────────────────────────────────────────────────────
# Per-set performance dataclass (input to FI/PDS)
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class SetPerformance:
    """Per-set summary metrics drawn from per-rep encoder analysis.

    Populated by ``strength_3lift._compute_set_performance`` (Phase V2).
    All numeric fields default to 0.0 / 0 when no usable encoder data
    was captured for that set; the FI/PDS computation will then return
    ``skipped_reason`` if every working set is empty.
    """
    set_idx:               int
    warmup:                bool
    n_reps:                int
    mean_power_w:          float = 0.0
    peak_power_w:          float = 0.0
    mean_velocity_m_s:     float = 0.0
    peak_velocity_m_s:     float = 0.0
    total_work_j:          float = 0.0
    rom_mm:                float = 0.0


# ────────────────────────────────────────────────────────────────────────────
# Recovery metrics dataclass
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class RecoveryMetrics:
    """ATP-PCr recovery resilience for a multi-set session.

    Always carries ``primary_variable`` so callers know which channel
    fed the calculation. Set indices in ``set_indices`` map back to
    StrengthResult.sets[] for cross-referencing.
    """
    primary_variable:    str                           # 'mean_power_w' / etc.

    # Per-working-set values (warmup excluded, in chronological order).
    set_values:          list[float] = field(default_factory=list)
    set_indices:         list[int]   = field(default_factory=list)

    # Fatigue Index — first-vs-last drop.
    fi_pct:              float = float("nan")
    fi_grade:            Optional[int] = None
    fi_label:            Optional[str] = None
    fi_interpretation:   Optional[str] = None

    # Performance Decrement Score — sum-vs-ideal drop.
    pds_pct:             float = float("nan")
    pds_grade:           Optional[int] = None
    pds_label:           Optional[str] = None
    pds_interpretation:  Optional[str] = None

    # Endurance ↔ Power tendency (-1 = pure endurance, +1 = pure power).
    # Computed from PDS (high decline = fast-twitch fatigue suggested)
    # combined with the magnitude of set 1 (high → power profile).
    fiber_tendency:      float = 0.0
    fiber_label:         Optional[str] = None         # Korean tendency label

    # Diagnostics
    skipped_reason:      Optional[str] = None
    n_working_sets:      int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ────────────────────────────────────────────────────────────────────────────
# Public formulas
# ────────────────────────────────────────────────────────────────────────────
def fatigue_index(set_values: list[float]) -> float:
    """FI(%) = (Set_1 - Set_N) / Set_1 × 100.

    Returns NaN when fewer than 2 sets are supplied or when Set_1 is
    non-positive (zero/negative — can't compute a relative drop).
    """
    if len(set_values) < 2:
        return float("nan")
    s1, sn = float(set_values[0]), float(set_values[-1])
    if s1 <= 0:
        return float("nan")
    return (s1 - sn) / s1 * 100.0


def performance_decrement_score(set_values: list[float]) -> float:
    """PDS(%) = (1 - Σ_i Set_i / (Set_1 × N)) × 100.

    Glaister-style decrement score: how much does the actual
    cumulative performance fall short of the "ideal" of every set
    matching the first. Returns NaN when fewer than 2 sets or Set_1
    non-positive.
    """
    if len(set_values) < 2:
        return float("nan")
    s1 = float(set_values[0])
    if s1 <= 0:
        return float("nan")
    n = len(set_values)
    actual = sum(float(v) for v in set_values)
    ideal = s1 * n
    return (1.0 - actual / ideal) * 100.0


def grade_fi(fi_pct: float) -> tuple[int, str, str]:
    """Look up the 1-5 grade for a Fatigue Index value.

    Negative FI (last > first → improvement) is clipped to 0 → grade 1.
    """
    v = max(0.0, float(fi_pct))
    for upper, grade, label, interp in _FI_GRADES:
        if v < upper:
            return grade, label, interp
    return _FI_GRADES[-1][1], _FI_GRADES[-1][2], _FI_GRADES[-1][3]


def grade_pds(pds_pct: float) -> tuple[int, str, str]:
    """Look up the 1-5 grade for a Performance Decrement Score."""
    v = max(0.0, float(pds_pct))
    for upper, grade, label, interp in _PDS_GRADES:
        if v < upper:
            return grade, label, interp
    return _PDS_GRADES[-1][1], _PDS_GRADES[-1][2], _PDS_GRADES[-1][3]


def fiber_tendency(pds_pct: float, set1_value: float,
                    population_ref: Optional[float] = None) -> tuple[float, str]:
    """Heuristic endurance↔power slider position from PDS + Set 1 magnitude.

    The plan suggests:
      - Low PDS  → Type 1 (slow-twitch / endurance) dominant
      - High PDS + high Set 1  → Type 2 (fast-twitch / power) dominant
      - High PDS + low Set 1   → just generally poor (not power)

    We map PDS to a [-1, +1] tendency centred at 25% (= grade-3 boundary):
      tendency = (pds_pct - 25) / 25, clipped to [-1, +1]

    When ``population_ref`` is provided (e.g. expected mean_power_w
    for the subject's bodyweight × age class), the tendency is
    weighted by ``set1 / population_ref`` so a high-PDS-yet-low-power
    subject doesn't get falsely labelled "power dominant".

    Returns ``(tendency, label_ko)``.
    """
    raw = (float(pds_pct) - 25.0) / 25.0
    raw = max(-1.0, min(1.0, raw))
    if raw > 0 and population_ref is not None and population_ref > 0:
        # Down-weight the "power" reading when set 1 is below the
        # population reference — high decline alone isn't power.
        weight = max(0.0, min(1.0, float(set1_value) / float(population_ref)))
        raw *= weight
    if raw <= -0.5:
        label = "지구력 우세 (Type 1)"
    elif raw <= -0.15:
        label = "지구력 약간 우세"
    elif raw < 0.15:
        label = "균형형"
    elif raw < 0.5:
        label = "파워 약간 우세"
    else:
        label = "파워 우세 (Type 2)"
    return raw, label


def compute_recovery_metrics(set_perfs: list[SetPerformance],
                              variable: str = "mean_power_w"
                              ) -> RecoveryMetrics:
    """End-to-end pipeline: SetPerformance list → RecoveryMetrics.

    Excludes warmup sets, picks the value of ``variable`` from each
    working set, runs FI / PDS / fiber-tendency, and assembles the
    full RecoveryMetrics dataclass.

    Args:
        set_perfs: per-set summaries (as produced by
                   ``strength_3lift._compute_set_performance``).
        variable:  attribute name on ``SetPerformance`` to use as the
                   per-set scalar. Default ``mean_power_w``.

    Skipped reasons (RecoveryMetrics.skipped_reason populated):
        - "fewer than 2 working sets"
        - "selected variable was zero in working sets"
        - "first working set value non-positive"
    """
    working = [s for s in set_perfs if not s.warmup]
    if len(working) < 2:
        return RecoveryMetrics(
            primary_variable=variable,
            set_values=[], set_indices=[],
            n_working_sets=len(working),
            skipped_reason="fewer than 2 working sets",
        )
    values = [float(getattr(s, variable, 0.0) or 0.0) for s in working]
    indices = [int(s.set_idx) for s in working]
    if not any(v > 0 for v in values):
        return RecoveryMetrics(
            primary_variable=variable,
            set_values=values, set_indices=indices,
            n_working_sets=len(working),
            skipped_reason=f"all working-set {variable} were zero",
        )
    if values[0] <= 0:
        return RecoveryMetrics(
            primary_variable=variable,
            set_values=values, set_indices=indices,
            n_working_sets=len(working),
            skipped_reason=f"first working-set {variable} was non-positive",
        )

    fi  = fatigue_index(values)
    pds = performance_decrement_score(values)
    fi_grade, fi_label, fi_interp   = grade_fi(fi)
    pds_grade, pds_label, pds_interp = grade_pds(pds)
    tendency, tendency_label = fiber_tendency(pds, values[0])

    return RecoveryMetrics(
        primary_variable=variable,
        set_values=values,
        set_indices=indices,
        fi_pct=round(fi, 2),
        fi_grade=fi_grade,
        fi_label=fi_label,
        fi_interpretation=fi_interp,
        pds_pct=round(pds, 2),
        pds_grade=pds_grade,
        pds_label=pds_label,
        pds_interpretation=pds_interp,
        fiber_tendency=round(tendency, 3),
        fiber_label=tendency_label,
        n_working_sets=len(working),
    )
