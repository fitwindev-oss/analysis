"""
Normative-range lookup + traffic-light classification.

Schema (see norms_data.json):
    test_type → variant → metric → key → NormRange

``variant`` differentiates stance (two/left/right), mode (eyes_open/closed),
or a placeholder ``_any`` / ``_romberg_ratio`` for cross-session derived
metrics. Variants not found in the dataset silently fall back to
``"default"`` or the closest match.

``key`` is an age/sex bucket string built from the subject profile
(``_bucket_key(subject)``). Falls back progressively
``adult_18_40_M`` → ``adult_18_40_any`` → ``default``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_DATA_PATH = Path(__file__).parent / "norms_data.json"
_CACHE: Optional[dict] = None


def _data() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = {}
    return _CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Subject bucket resolution
# ─────────────────────────────────────────────────────────────────────────────

def _age_years(subject) -> Optional[int]:
    """Compute integer age from subject.birthdate (ISO 'YYYY-MM-DD')."""
    if subject is None:
        return None
    bd = getattr(subject, "birthdate", None)
    if not bd:
        return None
    try:
        import datetime as _dt
        y, m, d = bd.split("-")
        birth = _dt.date(int(y), int(m), int(d))
        today = _dt.date.today()
        return today.year - birth.year - (
            (today.month, today.day) < (birth.month, birth.day))
    except Exception:
        return None


def _sex_tag(subject) -> str:
    """'M' / 'F' / 'any' depending on subject.gender."""
    if subject is None:
        return "any"
    g = (getattr(subject, "gender", "") or "").upper()
    return {"M": "M", "F": "F"}.get(g, "any")


def _bucket_candidates(subject) -> list[str]:
    """Ordered list of candidate keys, most-specific first."""
    age = _age_years(subject)
    sex = _sex_tag(subject)
    if age is None:
        band = "adult_18_40"
    elif age < 40:
        band = "adult_18_40"
    elif age < 60:
        band = "adult_40_60"
    else:
        band = "senior_60p"
    return [
        f"{band}_{sex}",
        f"{band}_any",
        "default",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Range + classification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NormRange:
    ok_low:       float
    ok_high:      float
    caution_low:  float
    caution_high: float
    lower_is_better:  bool = False
    higher_is_better: bool = False

    def ok_tuple(self) -> tuple[float, float]:
        return (self.ok_low, self.ok_high)

    def caution_tuple(self) -> tuple[float, float]:
        return (self.caution_low, self.caution_high)


def get_norm(test_type: str, metric: str,
             variant: Optional[str] = None,
             subject=None) -> Optional[NormRange]:
    """Return the matching NormRange or None if not found."""
    data = _data()
    node = data.get(test_type)
    if not node:
        return None
    # Variant resolution with graceful fallback
    if variant is None:
        variant = "_any" if "_any" in node else ("default" if "default" in node else next(iter(node)))
    vnode = node.get(variant) or node.get("default") or node.get("_any")
    if not vnode:
        return None
    mnode = vnode.get(metric)
    if not mnode:
        return None
    # Bucket resolution
    keys = _bucket_candidates(subject)
    chosen = None
    for k in keys:
        if k in mnode:
            chosen = mnode[k]; break
    if chosen is None:
        chosen = mnode.get("default") or next(iter(mnode.values()))
    try:
        ok_lo, ok_hi = chosen["ok_range"]
        ca_lo, ca_hi = chosen.get("caution_range", [ok_lo, ok_hi])
        return NormRange(
            ok_low=float(ok_lo), ok_high=float(ok_hi),
            caution_low=float(ca_lo), caution_high=float(ca_hi),
            lower_is_better=bool(chosen.get("lower_is_better", False)),
            higher_is_better=bool(chosen.get("higher_is_better", False)),
        )
    except Exception:
        return None


def classify(value: Optional[float], norm: Optional[NormRange]) -> str:
    """Return 'ok' / 'caution' / 'warning' / 'neutral'.

    Inside ok_range → ok. Inside caution_range but outside ok → caution.
    Outside caution_range → warning. No norm → neutral.
    """
    if value is None or norm is None:
        return "neutral"
    v = float(value)
    if norm.ok_low <= v <= norm.ok_high:
        return "ok"
    if norm.caution_low <= v <= norm.caution_high:
        return "caution"
    return "warning"


def classify_with_direction(value: Optional[float],
                            norm: Optional[NormRange]) -> str:
    """Like classify() but 'one-sided' for directional metrics.

    When ``lower_is_better=True``, values below ok_range.low are still ok
    (not warning). Same for higher_is_better with values above ok_range.high.
    """
    if value is None or norm is None:
        return "neutral"
    v = float(value)
    if norm.lower_is_better:
        if v <= norm.ok_high:
            return "ok"
        if v <= norm.caution_high:
            return "caution"
        return "warning"
    if norm.higher_is_better:
        if v >= norm.ok_low:
            return "ok"
        if v >= norm.caution_low:
            return "caution"
        return "warning"
    return classify(value, norm)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience formatters
# ─────────────────────────────────────────────────────────────────────────────

def norm_tooltip(norm: Optional[NormRange], unit: str = "") -> str:
    """Human readable string — e.g. '정상 8–20 mm/s'."""
    if norm is None:
        return ""
    return f"정상 {norm.ok_low:g}–{norm.ok_high:g}{unit}"
