"""
Strength norms — bodyweight × age coefficient tables for 1RM grading
(Phase V1).

Tables transcribed from "근골격 측정 평가 리포트 개발 계획서" v1.
Source values are population norms from English-language strength
training literature (StrengthLevel-style). They have NOT been
calibrated for a Korean subject population. Once a sufficient
measurement corpus has accumulated, derive a ``*_KR`` variant module
and switch the default at the public-API level.

Phase V1 covers the three barbell movements only:
  - bench_press   → Chest
  - back_squat    → Legs
  - deadlift      → Whole Body

The remaining four pin-loaded movements (Shoulder Press, Barbell Curl,
Push Down, Lat Pulldown) will be added later when those machines are
integrated.

Grade scale (unified 1–7):
   1  엘리트 (Elite)             — at or above the elite threshold
   2  좋음   (Advanced)
   3  보통   (Intermediate)
   4  나쁨   (Novice)
   5  위험   (Beginner)           — at or above beginner threshold
   6  경고   (Below beginner ≥ 80%)
   7  심각   (Below 80% of beginner)

The 1–5 portion comes from the per-region tables (Beginner..Elite).
6 and 7 cover sub-beginner zones the plan flags as "주의/심각".
This 1–7 scale is also what the composite-score formula uses.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Grade labels + composite-score support
# ────────────────────────────────────────────────────────────────────────────
GRADE_LABELS: dict[int, str] = {
    1: "엘리트",
    2: "좋음",
    3: "보통",
    4: "나쁨",
    5: "위험",
    6: "경고",
    7: "심각",
}

# Per-grade percentage score (used by the composite formula in the plan).
GRADE_PERCENT: dict[int, int] = {
    1: 100, 2: 89, 3: 75, 4: 65, 5: 55, 6: 45, 7: 35,
}

# Composite-grade cutoffs on percent-score range (descending).
# Format: (lower_inclusive_pct, composite_grade).
COMPOSITE_GRADE_CUTOFFS: list[tuple[int, int]] = [
    (90, 1), (76, 2), (66, 3), (56, 4), (46, 5), (35, 6), (0, 7),
]

# Body region weight points used by the composite calculation.
# Plan: 7 + 7 + 9 + 13 + 20 + 20 + 24 = 100.
REGION_WEIGHTS: dict[str, int] = {
    "biceps":     7,
    "triceps":    7,
    "shoulder":   9,
    "chest":      13,
    "legs":       20,
    "back":       20,
    "whole_body": 24,
}

# Map V1 exercises to their body-region key.
EXERCISE_REGION: dict[str, str] = {
    "bench_press": "chest",
    "back_squat":  "legs",
    "deadlift":    "whole_body",
}

VALID_EXERCISES: tuple[str, ...] = tuple(EXERCISE_REGION.keys())


# ────────────────────────────────────────────────────────────────────────────
# Norm table object
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Norm:
    """One exercise × sex norm: bw bins + age bins.

    Lookup uses ``bisect_left`` so a value on a bin boundary falls
    into the lower bin (e.g. bw=54 → -54 bin). Out-of-range values
    are clamped to the first/last bin.
    """
    # Inclusive upper bound of each bw bin in kg. The first bin
    # is "≤ first upper". The last bin is "≥ last upper + 1" (open).
    bw_upper:    tuple[int, ...]
    # 5-tuple per row: (Beginner, Novice, Intermediate, Advanced, Elite).
    bw_coefs:    tuple[tuple[float, float, float, float, float], ...]
    age_upper:   tuple[int, ...]
    age_factors: tuple[float, ...]

    def lookup_bw_coefs(self, bw_kg: float) -> tuple[float, float, float, float, float]:
        idx = bisect_left(self.bw_upper, bw_kg)
        idx = min(idx, len(self.bw_coefs) - 1)
        return self.bw_coefs[idx]

    def lookup_age_factor(self, age: int) -> float:
        idx = bisect_left(self.age_upper, age)
        idx = min(idx, len(self.age_factors) - 1)
        return self.age_factors[idx]


# ────────────────────────────────────────────────────────────────────────────
# Bench Press — Male
# ────────────────────────────────────────────────────────────────────────────
BENCH_PRESS_MALE = _Norm(
    bw_upper=(54, 59, 64, 69, 74, 79, 84, 89, 94, 99,
              104, 109, 114, 119, 124, 129, 134, 139),
    bw_coefs=(
        (0.48, 0.76, 1.14, 1.58, 2.06),  # -54
        (0.53, 0.82, 1.16, 1.58, 2.05),  # 55-59
        (0.57, 0.85, 1.20, 1.60, 2.05),  # 60-64
        (0.60, 0.88, 1.22, 1.60, 2.03),  # 65-69
        (0.63, 0.89, 1.21, 1.60, 2.01),  # 70-74
        (0.65, 0.91, 1.23, 1.59, 1.99),  # 75-79
        (0.66, 0.93, 1.23, 1.59, 1.96),  # 80-84
        (0.68, 0.93, 1.24, 1.58, 1.94),  # 85-89
        (0.69, 0.93, 1.23, 1.57, 1.91),  # 90-94
        (0.68, 0.89, 1.22, 1.55, 1.89),  # 95-99
        (0.66, 0.88, 1.22, 1.53, 1.87),  # 100-104
        (0.65, 0.84, 1.22, 1.52, 1.85),  # 105-109
        (0.63, 0.81, 1.21, 1.51, 1.82),  # 110-114
        (0.61, 0.79, 1.20, 1.50, 1.80),  # 115-119
        (0.59, 0.76, 1.19, 1.48, 1.78),  # 120-124
        (0.57, 0.76, 1.18, 1.46, 1.75),  # 125-129
        (0.55, 0.76, 1.18, 1.45, 1.73),  # 130-134
        (0.54, 0.74, 1.17, 1.44, 1.71),  # 135-139
        (0.53, 0.72, 1.16, 1.42, 1.69),  # 140-
    ),
    age_upper=(19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89),
    age_factors=(
        0.85, 0.97, 1.00, 0.97, 0.94, 0.91, 0.88, 0.84,   # 15-19 .. 50-54
        0.81, 0.78, 0.73, 0.50, 0.45, 0.38, 0.35, 0.33,    # 55-59 .. 90-100
    ),
)

# ────────────────────────────────────────────────────────────────────────────
# Bench Press — Female
# ────────────────────────────────────────────────────────────────────────────
BENCH_PRESS_FEMALE = _Norm(
    bw_upper=(44, 49, 54, 59, 64, 69, 74, 79, 84, 89,
              94, 99, 104, 109, 114, 119),
    bw_coefs=(
        (0.20, 0.45, 0.80, 1.25, 1.75),  # -44
        (0.22, 0.47, 0.80, 1.22, 1.69),  # 45-49
        (0.24, 0.48, 0.80, 1.18, 1.64),  # 50-54
        (0.27, 0.49, 0.78, 1.16, 1.58),  # 55-59
        (0.28, 0.48, 0.78, 1.13, 1.53),  # 60-64
        (0.29, 0.49, 0.77, 1.11, 1.48),  # 65-69
        (0.29, 0.49, 0.76, 1.07, 1.44),  # 70-74
        (0.29, 0.49, 0.75, 1.05, 1.40),  # 75-79
        (0.30, 0.49, 0.74, 1.03, 1.36),  # 80-84
        (0.31, 0.48, 0.73, 1.01, 1.32),  # 85-89
        (0.31, 0.49, 0.71, 0.99, 1.29),  # 90-94
        (0.31, 0.48, 0.71, 0.97, 1.25),  # 95-99
        (0.31, 0.48, 0.69, 0.95, 1.23),  # 100-104
        (0.31, 0.48, 0.69, 0.93, 1.20),  # 105-109
        (0.31, 0.47, 0.67, 0.91, 1.17),  # 110-114
        (0.31, 0.47, 0.66, 0.90, 1.15),  # 115-119
        (0.31, 0.47, 0.66, 0.88, 1.13),  # 120-
    ),
    age_upper=(19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89),
    age_factors=(
        0.78, 0.97, 1.00, 0.95, 0.90, 0.85, 0.83, 0.80,
        0.75, 0.70, 0.65, 0.50, 0.40, 0.38, 0.35, 0.31,
    ),
)

# ────────────────────────────────────────────────────────────────────────────
# Back Squat — Male
# ────────────────────────────────────────────────────────────────────────────
BACK_SQUAT_MALE = _Norm(
    bw_upper=(54, 59, 64, 69, 74, 79, 84, 89, 94, 99,
              104, 109, 114, 119, 124, 129, 134, 139),
    bw_coefs=(
        (0.66, 1.04, 1.52, 2.08, 2.72),
        (0.73, 1.09, 1.56, 2.11, 2.71),
        (0.78, 1.13, 1.58, 2.12, 2.68),
        (0.82, 1.17, 1.60, 2.11, 2.66),
        (0.84, 1.19, 1.61, 2.10, 2.63),
        (0.88, 1.21, 1.63, 2.09, 2.60),
        (0.90, 1.23, 1.63, 2.08, 2.56),
        (0.92, 1.24, 1.62, 2.06, 2.53),
        (0.92, 1.24, 1.62, 2.04, 2.50),
        (0.91, 1.19, 1.61, 2.02, 2.46),
        (0.88, 1.17, 1.60, 2.01, 2.43),
        (0.87, 1.12, 1.60, 1.99, 2.40),
        (0.84, 1.08, 1.58, 1.96, 2.36),
        (0.81, 1.05, 1.57, 1.95, 2.34),
        (0.79, 1.01, 1.57, 1.93, 2.31),
        (0.76, 1.01, 1.55, 1.90, 2.27),
        (0.73, 1.01, 1.55, 1.88, 2.25),
        (0.72, 0.99, 1.53, 1.87, 2.21),
        (0.71, 0.96, 1.52, 1.85, 2.19),
    ),
    age_upper=(19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89),
    age_factors=(
        0.86, 0.97, 1.00, 1.00, 0.98, 0.96, 0.95, 0.89,
        0.83, 0.75, 0.69, 0.61, 0.55, 0.48, 0.44, 0.39,
    ),
)

# ────────────────────────────────────────────────────────────────────────────
# Back Squat — Female
# ────────────────────────────────────────────────────────────────────────────
BACK_SQUAT_FEMALE = _Norm(
    bw_upper=(44, 49, 54, 59, 64, 69, 74, 79, 84, 89,
              94, 99, 104, 109, 114, 119),
    bw_coefs=(
        (0.43, 0.78, 1.28, 1.88, 2.53),
        (0.44, 0.80, 1.24, 1.80, 2.42),
        (0.46, 0.78, 1.22, 1.74, 2.30),
        (0.47, 0.78, 1.18, 1.67, 2.22),
        (0.48, 0.78, 1.17, 1.62, 2.13),
        (0.49, 0.77, 1.14, 1.57, 2.05),
        (0.49, 0.76, 1.11, 1.51, 1.97),
        (0.49, 0.75, 1.08, 1.48, 1.91),
        (0.49, 0.74, 1.06, 1.44, 1.85),
        (0.51, 0.72, 1.04, 1.40, 1.79),
        (0.51, 0.74, 1.01, 1.37, 1.74),
        (0.47, 0.72, 1.00, 1.33, 1.69),
        (0.46, 0.72, 0.98, 1.30, 1.65),
        (0.46, 0.72, 0.96, 1.27, 1.61),
        (0.46, 0.71, 0.94, 1.24, 1.56),
        (0.46, 0.71, 0.92, 1.22, 1.53),
        (0.44, 0.71, 0.91, 1.19, 1.49),
    ),
    age_upper=(19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89),
    age_factors=(
        0.83, 0.97, 1.00, 1.00, 0.98, 0.95, 0.93, 0.87,
        0.80, 0.73, 0.67, 0.60, 0.53, 0.47, 0.43, 0.40,
    ),
)

# ────────────────────────────────────────────────────────────────────────────
# Deadlift — Male
# ────────────────────────────────────────────────────────────────────────────
DEADLIFT_MALE = _Norm(
    bw_upper=(54, 59, 64, 69, 74, 79, 84, 89, 94, 99,
              104, 109, 114, 119, 124, 129, 134, 139),
    bw_coefs=(
        (0.88, 1.30, 1.86, 2.50, 3.20),
        (0.93, 1.35, 1.87, 2.49, 3.16),
        (0.97, 1.38, 1.90, 2.48, 3.12),
        (1.02, 1.42, 1.91, 2.46, 3.08),
        (1.04, 1.43, 1.90, 2.44, 3.03),
        (1.05, 1.44, 1.89, 2.43, 2.99),
        (1.08, 1.45, 1.89, 2.40, 2.94),
        (1.09, 1.45, 1.88, 2.36, 2.88),
        (1.10, 1.46, 1.87, 2.34, 2.84),
        (1.08, 1.40, 1.85, 2.32, 2.80),
        (1.05, 1.38, 1.84, 2.28, 2.75),
        (1.04, 1.32, 1.83, 2.26, 2.70),
        (1.00, 1.27, 1.81, 2.23, 2.66),
        (0.97, 1.24, 1.79, 2.20, 2.63),
        (0.94, 1.19, 1.78, 2.18, 2.59),
        (0.91, 1.19, 1.76, 2.14, 2.55),
        (0.88, 1.19, 1.75, 2.12, 2.52),
        (0.86, 1.16, 1.73, 2.10, 2.48),
        (0.84, 1.13, 1.71, 2.07, 2.44),
    ),
    age_upper=(19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89),
    age_factors=(
        0.87, 0.98, 1.00, 1.00, 1.00, 0.98, 0.95, 0.90,
        0.83, 0.76, 0.68, 0.62, 0.55, 0.49, 0.44, 0.40,
    ),
)

# ────────────────────────────────────────────────────────────────────────────
# Deadlift — Female
# ────────────────────────────────────────────────────────────────────────────
DEADLIFT_FEMALE = _Norm(
    bw_upper=(44, 49, 54, 59, 64, 69, 74, 79, 84, 89,
              94, 99, 104, 109, 114, 119),
    bw_coefs=(
        (0.60, 1.00, 1.55, 2.23, 2.95),
        (0.60, 1.00, 1.51, 2.11, 2.80),
        (0.62, 0.98, 1.46, 2.04, 2.66),
        (0.62, 0.96, 1.42, 1.95, 2.55),
        (0.62, 0.95, 1.38, 1.88, 2.43),
        (0.62, 0.94, 1.34, 1.82, 2.34),
        (0.61, 0.91, 1.30, 1.76, 2.24),
        (0.60, 0.89, 1.27, 1.69, 2.17),
        (0.60, 0.89, 1.24, 1.65, 2.10),
        (0.60, 0.87, 1.20, 1.60, 2.02),
        (0.59, 0.86, 1.18, 1.56, 1.97),
        (0.58, 0.83, 1.15, 1.52, 1.91),
        (0.58, 0.82, 1.12, 1.47, 1.85),
        (0.57, 0.81, 1.10, 1.44, 1.80),
        (0.56, 0.79, 1.08, 1.40, 1.75),
        (0.56, 0.78, 1.05, 1.37, 1.71),
        (0.55, 0.77, 1.03, 1.34, 1.67),
    ),
    age_upper=(19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89),
    age_factors=(
        0.84, 0.97, 1.00, 1.00, 1.00, 0.98, 0.95, 0.89,
        0.82, 0.76, 0.68, 0.61, 0.55, 0.50, 0.45, 0.39,
    ),
)


_NORM_TABLE: dict[tuple[str, str], _Norm] = {
    ("bench_press", "M"): BENCH_PRESS_MALE,
    ("bench_press", "F"): BENCH_PRESS_FEMALE,
    ("back_squat",  "M"): BACK_SQUAT_MALE,
    ("back_squat",  "F"): BACK_SQUAT_FEMALE,
    ("deadlift",    "M"): DEADLIFT_MALE,
    ("deadlift",    "F"): DEADLIFT_FEMALE,
}


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────
def grade_1rm(exercise: str, sex: str, age: int, bw_kg: float,
              one_rm_kg: float) -> dict:
    """Compute the 1–7 grade for a 1RM lift.

    Args:
        exercise: ``'bench_press' | 'back_squat' | 'deadlift'``
        sex:      ``'M' | 'F'``
        age:      years (int; non-int rounded down via ``int()``)
        bw_kg:    bodyweight in kg
        one_rm_kg: estimated or measured 1RM in kg

    Returns dict with keys:
        exercise, sex, age, bw_kg, one_rm_kg
        grade            int 1-7
        label            str (Korean)
        thresholds_kg    dict with beginner/novice/intermediate/advanced/elite
        ratio_to_elite   one_rm_kg / elite_threshold
        ratio_to_beginner one_rm_kg / beginner_threshold
        warning          None | 'caution' (1RM in [80%, 100%) of beginner)
                              | 'severe'   (1RM < 80% of beginner)

    Raises:
        ValueError on unknown exercise or sex.
    """
    norm = _resolve_norm(exercise, sex)
    coefs = norm.lookup_bw_coefs(float(bw_kg))
    age_f = norm.lookup_age_factor(int(age))

    thresholds = {
        "beginner":     bw_kg * coefs[0] * age_f,
        "novice":       bw_kg * coefs[1] * age_f,
        "intermediate": bw_kg * coefs[2] * age_f,
        "advanced":     bw_kg * coefs[3] * age_f,
        "elite":        bw_kg * coefs[4] * age_f,
    }

    # Use a small epsilon so 1RM that is "exactly equal" to a
    # threshold (in human-meaningful terms) doesn't get pushed to the
    # lower grade by IEEE-754 representation noise. 1e-6 kg = 1 mg —
    # well below any realistic measurement precision.
    EPS = 1e-6
    if one_rm_kg >= thresholds["elite"] - EPS:
        grade, warning = 1, None
    elif one_rm_kg >= thresholds["advanced"] - EPS:
        grade, warning = 2, None
    elif one_rm_kg >= thresholds["intermediate"] - EPS:
        grade, warning = 3, None
    elif one_rm_kg >= thresholds["novice"] - EPS:
        grade, warning = 4, None
    elif one_rm_kg >= thresholds["beginner"] - EPS:
        grade, warning = 5, None
    elif one_rm_kg >= 0.80 * thresholds["beginner"] - EPS:
        grade, warning = 6, "caution"
    else:
        grade, warning = 7, "severe"

    return {
        "exercise":          exercise,
        "sex":               sex.upper(),
        "age":               int(age),
        "bw_kg":             float(bw_kg),
        "one_rm_kg":         float(one_rm_kg),
        "grade":             grade,
        "label":             GRADE_LABELS[grade],
        "thresholds_kg":     {k: round(v, 2) for k, v in thresholds.items()},
        "ratio_to_elite":    one_rm_kg / thresholds["elite"]
                              if thresholds["elite"] > 0 else 0.0,
        "ratio_to_beginner": one_rm_kg / thresholds["beginner"]
                              if thresholds["beginner"] > 0 else 0.0,
        "warning":           warning,
        "bw_coefs":          coefs,
        "age_factor":        age_f,
    }


def composite_score(per_region_grades: dict[str, int]) -> dict:
    """Compute composite (1–7) score from a dict of region → grade.

    The plan formula:
        score = Σ (region_weight × grade_percent)

    where ``grade_percent`` is the per-grade percentage from
    ``GRADE_PERCENT`` (grade 1 = 100, grade 2 = 89, ..., grade 7 = 35).
    Missing regions are skipped — total weight is renormalised to the
    measured regions so a partial assessment still produces a sensible
    score.

    Args:
        per_region_grades: e.g. ``{'chest': 2, 'legs': 3, 'whole_body': 4}``
                           keys must be in ``REGION_WEIGHTS``.
                           values must be 1-7.

    Returns dict with:
        score_pct      float 0-100
        composite_grade int 1-7
        composite_label str
        n_regions      int (how many regions contributed)
        weighted_total int (sum of weights actually used)
    """
    if not per_region_grades:
        return {
            "score_pct":       0.0,
            "composite_grade": 7,
            "composite_label": GRADE_LABELS[7],
            "n_regions":       0,
            "weighted_total":  0,
        }

    score_sum = 0.0
    weight_sum = 0
    for region, grade in per_region_grades.items():
        if region not in REGION_WEIGHTS:
            raise ValueError(f"unknown region: {region}")
        if grade not in GRADE_PERCENT:
            raise ValueError(f"invalid grade: {grade}")
        w = REGION_WEIGHTS[region]
        score_sum += w * GRADE_PERCENT[grade]
        weight_sum += w

    # Normalise to a 0–100 scale of the measured regions.
    # When all 7 regions are present, weight_sum = 100 and this is a no-op.
    score_pct = score_sum / weight_sum if weight_sum else 0.0

    composite_grade = 7
    for cutoff_pct, g in COMPOSITE_GRADE_CUTOFFS:
        if score_pct >= cutoff_pct:
            composite_grade = g
            break

    return {
        "score_pct":       round(score_pct, 2),
        "composite_grade": composite_grade,
        "composite_label": GRADE_LABELS[composite_grade],
        "n_regions":       len(per_region_grades),
        "weighted_total":  weight_sum,
    }


def _resolve_norm(exercise: str, sex: str) -> _Norm:
    key = (exercise.lower(), sex.upper())
    if key not in _NORM_TABLE:
        raise ValueError(
            f"no norm for exercise={exercise!r} sex={sex!r} "
            f"(valid: {list(_NORM_TABLE.keys())})"
        )
    return _NORM_TABLE[key]
