"""
1RM (one-rep maximum) estimation from rep-based lifts (Phase V1).

Three classical formulas are implemented; the public ``estimate_1rm``
returns either a single chosen formula's value or the ensemble
(arithmetic mean of all valid formulas), per the ``method`` argument.

Formulas:
    Epley     (1985)  : 1RM = load × (1 + reps / 30)
    Brzycki   (1993)  : 1RM = load × 36 / (37 - reps)        [reps < 37]
    Lombardi  (1989)  : 1RM = load × reps^0.10

Reliability bands (per Mayhew 1992 + LeSuer 1997):
    reps ≤ 5    excellent   (predictions within ±2-3% of true 1RM)
    reps ≤ 10   high
    reps ≤ 12   medium
    reps ≤ 20   low (only useful as a coarse screening estimate)
    reps > 20   unreliable  (formula assumptions break down)

The PDF plan calls for 10-12 reps to failure as the standard
protocol — that lands in the medium-reliability band, which is the
explicit trade-off chosen by the plan to keep the test sub-maximal
and safe for general subjects.

Velocity-based estimation (load-velocity profile fitting) is intended
for a future phase once the encoder integration captures bar velocity
per rep with sufficient precision; the function ``estimate_1rm_lv``
is stubbed here and raises ``NotImplementedError`` for now.
"""
from __future__ import annotations

import math
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Single-formula estimators
# ────────────────────────────────────────────────────────────────────────────
def epley(load_kg: float, reps: int) -> float:
    """Epley (1985)."""
    if reps < 1:
        raise ValueError(f"reps must be >= 1 (got {reps})")
    return float(load_kg) * (1.0 + reps / 30.0)


def brzycki(load_kg: float, reps: int) -> float:
    """Brzycki (1993). Returns NaN for reps >= 37 (formula divergence)."""
    if reps < 1:
        raise ValueError(f"reps must be >= 1 (got {reps})")
    if reps >= 37:
        return float("nan")
    return float(load_kg) * 36.0 / (37.0 - reps)


def lombardi(load_kg: float, reps: int) -> float:
    """Lombardi (1989). Better at high reps than the linear formulas."""
    if reps < 1:
        raise ValueError(f"reps must be >= 1 (got {reps})")
    return float(load_kg) * (reps ** 0.10)


# ────────────────────────────────────────────────────────────────────────────
# Ensemble + per-set wrappers
# ────────────────────────────────────────────────────────────────────────────
def reliability_band(reps: int) -> str:
    """Coarse reliability label for a 1RM estimate from given reps count."""
    if reps <= 5:
        return "excellent"
    if reps <= 10:
        return "high"
    if reps <= 12:
        return "medium"
    if reps <= 20:
        return "low"
    return "unreliable"


def estimate_1rm(load_kg: float, reps: int,
                 method: str = "ensemble") -> dict:
    """Estimate 1RM for a single set.

    Args:
        load_kg: weight on the bar
        reps:    completed reps to (or near) failure
        method:  ``'epley' | 'brzycki' | 'lombardi' | 'ensemble'`` (default).
                 ``ensemble`` averages all formulas that produced a finite
                 value (Brzycki returns NaN for reps >= 37, in which case
                 the average uses only the other two).

    Returns dict with:
        one_rm_kg     chosen estimate (float)
        epley_kg, brzycki_kg, lombardi_kg
        method        which method was used
        reliability   excellent/high/medium/low/unreliable
        load_kg, reps echoed for traceability
    """
    e = epley(load_kg, reps)
    b = brzycki(load_kg, reps)
    l = lombardi(load_kg, reps)

    if method == "epley":
        chosen = e
    elif method == "brzycki":
        chosen = b
    elif method == "lombardi":
        chosen = l
    elif method == "ensemble":
        valid = [v for v in (e, b, l) if not math.isnan(v)]
        chosen = sum(valid) / len(valid) if valid else float("nan")
    else:
        raise ValueError(
            f"unknown method: {method!r} "
            f"(expected: epley, brzycki, lombardi, ensemble)"
        )

    return {
        "one_rm_kg":   round(chosen, 2),
        "epley_kg":    round(e, 2),
        "brzycki_kg":  round(b, 2) if not math.isnan(b) else float("nan"),
        "lombardi_kg": round(l, 2),
        "method":      method,
        "reliability": reliability_band(reps),
        "load_kg":     float(load_kg),
        "reps":        int(reps),
    }


def estimate_1rm_from_sets(sets: list[dict],
                            method: str = "ensemble",
                            include_warmup: bool = False) -> dict:
    """Estimate 1RM from multiple sets, picking the SET with the
    highest predicted 1RM.

    Rationale: across a 3-set protocol with fatigue, the best set
    (usually set 1, sometimes 2) gives the closest estimate to the
    true 1RM. Averaging across fatigued sets systematically
    under-estimates.

    Args:
        sets: list of dicts, each with required keys ``load_kg``, ``reps``;
              optional key ``warmup`` (bool, default False).
        method: passed through to ``estimate_1rm``.
        include_warmup: if False (default), sets flagged ``warmup=True``
                        are excluded from the calculation.

    Returns dict with:
        one_rm_kg        best estimate across the working sets
        chosen_set_idx   index in the input list of the chosen set
        per_set          list of per-set estimate dicts (warmups excluded
                         when ``include_warmup`` is False)
        n_working_sets   how many sets contributed
    """
    per_set: list[dict] = []
    indices: list[int] = []
    for i, s in enumerate(sets):
        if not include_warmup and bool(s.get("warmup", False)):
            continue
        est = estimate_1rm(float(s["load_kg"]), int(s["reps"]), method=method)
        # Tag each per-set estimate with original index for downstream UI.
        est["set_idx"] = i
        est["warmup"] = bool(s.get("warmup", False))
        per_set.append(est)
        indices.append(i)

    if not per_set:
        return {
            "one_rm_kg":      float("nan"),
            "chosen_set_idx": None,
            "per_set":        [],
            "n_working_sets": 0,
            "method":         method,
        }

    # Argmax over one_rm_kg (skip NaNs).
    def _score(d: dict) -> float:
        v = d["one_rm_kg"]
        return -float("inf") if math.isnan(v) else v

    best = max(range(len(per_set)), key=lambda i: _score(per_set[i]))
    return {
        "one_rm_kg":      per_set[best]["one_rm_kg"],
        "chosen_set_idx": indices[best],
        "per_set":        per_set,
        "n_working_sets": len(per_set),
        "method":         method,
    }


# ────────────────────────────────────────────────────────────────────────────
# Velocity-based 1RM (placeholder for a later phase)
# ────────────────────────────────────────────────────────────────────────────
def estimate_1rm_lv(load_velocity_pairs: list[tuple[float, float]],
                     min_velocity_threshold: float = 0.15) -> dict:
    """Estimate 1RM from a load-velocity profile.

    Fits a linear regression to (load, mean concentric velocity)
    points and projects to the load at which velocity = the user's
    minimum velocity threshold (MVT), conventionally 0.15 m/s for
    bench press, 0.30 m/s for back squat.

    Currently NOT IMPLEMENTED — the encoder-derived velocity field
    needs additional validation before this can be trusted in
    clinical reports. Tracked as a follow-up after Phase V1.
    """
    raise NotImplementedError(
        "Load-velocity 1RM estimation will be added once encoder bar "
        "velocity has been validated against gold-standard linear "
        "transducer references (post-V1)."
    )
