"""
On-plate state classifier for raw force samples.

A subject is "on plate" when the total vertical force exceeds a fixed
threshold of **20 N** — the same value the CMJ analyser uses to detect
takeoff/landing (``flight_threshold_n`` in ``src/analysis/cmj.py``).

Why 20 N (not body-weight-derived):

  Earlier prototypes used ``max(50N, subject_kg × g × 20%)`` (e.g.
  157 N for an 80 kg subject) on the theory that "low load" should be
  treated as off-plate. That conflated two different events:

    * Subject is *unloading* the plate but feet still planted (force
      drops from 1280 N to 250 N during a CMJ counter-movement) —
      this is NOT off-plate. Subject is preparing to jump.

    * Subject's feet have *physically left* the plate (force ≈ 0) —
      this is the actual flight phase.

  A 157 N threshold catches both, producing a wider band that starts
  ~50 ms early (during takeoff push) and ends ~100 ms late (during
  landing impact recovery). Visually it disagrees with the CMJ
  analyser's flight shading by similar margins.

  The physical "feet have left" event is force-zero crossing. With
  100 Hz force sampling and noise floor ≈ 5–15 N, **20 N is the
  smallest threshold that won't false-trigger on noise**, and it
  matches the CMJ analyser exactly so all visualisations agree.

For balance-test "fell off" detection (subject stepped fully off the
plate during a balance trial), see SessionRecorder's ``_fell_off``
logic — that uses a 30% BW × 2 s threshold which is a different
event entirely (sustained presence-vs-absence over seconds, not a
takeoff transition over milliseconds).
"""
from __future__ import annotations


# 20 N — matches src/analysis/cmj.py's flight_threshold_n default so
# replay visualisations and CMJ analysis use the same takeoff/landing
# instants. Do NOT body-weight-scale this — flight is a physical event
# that happens at GRF ≈ 0 regardless of subject mass.
DEPARTURE_THRESHOLD_N = 20.0


def departure_threshold_n(subject_kg: float | None = None,
                          bw_ratio: float | None = None) -> float:
    """Force threshold below which the subject is considered off-plate.

    Returns a fixed 20 N for all subjects. The ``subject_kg`` and
    ``bw_ratio`` parameters are accepted (and ignored) for backward
    compatibility with callers wired up before the simplification.

    Use this constant whenever you need to ask "are the subject's feet
    on the plate?". For "is the load high enough that CoP is reliable?"
    use the CoPFilter's separate 20 % BW gate — that's a different
    question with a different answer.
    """
    return DEPARTURE_THRESHOLD_N


def classify_on_plate(total_n: float,
                      threshold_n: float = DEPARTURE_THRESHOLD_N) -> int:
    """Return 1 if the sample is on-plate, 0 otherwise.

    Boundary is inclusive on the on-plate side (``≥ threshold``). The
    threshold defaults to ``DEPARTURE_THRESHOLD_N`` so call sites that
    don't precompute can omit the argument; passing it explicitly is
    still cheaper than re-resolving the constant on every sample.
    """
    return int(total_n >= threshold_n)
