"""
Report builder — assembles a list of ReportSection for a given session
based on its test type and the desired audience.

Usage:
    from src.reports.report_builder import build_trainer_report, build_subject_report
    sections = build_trainer_report(ctx)          # list[ReportSection]
    html = render_html(sections, ctx)
    render_pdf(sections, ctx, "out.pdf")
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from src.reports.base import ReportContext, ReportSection
from src.reports.norms import (
    NormRange, classify_with_direction, get_norm, norm_tooltip,
)
from src.reports.sections.balance import BalanceChartsSection
from src.reports.sections.cmj import CmjChartsSection
from src.reports.sections.common import (
    FooterSection, HeaderSection, MetricCard, NotesSection,
    SummaryCardsSection,
)
from src.reports.sections.cover import CoverPageSection
from src.reports.sections.detail import DetailSection
from src.reports.sections.encoder import EncoderChartsSection
from src.reports.sections.glossary import GlossarySection
from src.reports.sections.history import HistorySection
from src.reports.sections.pose_angles import PoseAnglesSection
from src.reports.sections.proprio import ProprioChartsSection
from src.reports.sections.reaction import ReactionChartsSection
from src.reports.sections.cognitive_reaction import CognitiveReactionSection
from src.reports.sections.squat import SquatChartsSection
from src.reports.sections.squat_precision import SquatPrecisionSection
from src.reports.sections.strength_3lift import Strength3LiftSection
from src.reports.sections.strength_composite import StrengthCompositeSection
from src.reports.sections.ssc import SSCSection
from src.reports.sections.verdict import ExecutiveSummarySection


_CHARTS_FOR_TEST: dict[str, Callable[[], ReportSection]] = {
    "balance_eo":          BalanceChartsSection,
    "balance_ec":          BalanceChartsSection,
    "cmj":                 CmjChartsSection,
    "sj":                  CmjChartsSection,    # V4 — same charts as CMJ
    "squat":               SquatChartsSection,
    "overhead_squat":      SquatChartsSection,
    "encoder":             EncoderChartsSection,
    "reaction":            ReactionChartsSection,
    # V6 — cognitive reaction has its own dedicated section that owns
    # both summary table and the three diagnostic charts, so the
    # primary "charts section" router points to it directly.
    "cognitive_reaction":  CognitiveReactionSection,
    "proprio":             ProprioChartsSection,
}


def _charts_section_for(ctx: ReportContext) -> Optional[ReportSection]:
    cls = _CHARTS_FOR_TEST.get(ctx.test_type)
    return cls() if cls is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Card helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"
    return str(v)


def _card(label: str, value: Optional[float], unit: str,
          test_type: str, metric_key: str, variant: str,
          subject) -> MetricCard:
    """Build a metric card with auto classification + norm tooltip."""
    norm = get_norm(test_type, metric_key, variant=variant, subject=subject)
    status = classify_with_direction(value, norm)
    tip = norm_tooltip(norm, unit) if norm else ""
    label_full = (f"{label}  <small style='color:#888;'>{tip}</small>"
                  if tip else label)
    return MetricCard(
        label=label_full,
        value=_fmt_value(value),
        unit=unit,
        status=status,
    )


def _plain_card(label: str, value: Optional[float], unit: str) -> MetricCard:
    return MetricCard(
        label=label,
        value=_fmt_value(value),
        unit=unit,
        status="neutral",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-test summary card builders
# ─────────────────────────────────────────────────────────────────────────────

def _balance_cards(ctx: ReportContext) -> list[MetricCard]:
    r = ctx.result or {}
    stance = ctx.session_meta.get("stance", "two")
    subj = ctx.subject
    # Use test_type to switch EO/EC norms
    test = ctx.test_type
    return [
        _card("평균 이동 속도", r.get("mean_velocity_mm_s"), " mm/s",
              test, "mean_velocity_mm_s", stance, subj),
        _card("95% 타원 면적",  r.get("ellipse95_area_mm2"), " mm²",
              test, "ellipse95_area_mm2", stance, subj),
        _card("경로 길이",       r.get("path_length_mm"),     " mm",
              test, "path_length_mm", stance, subj),
        _card("RMS ML",         r.get("rms_ml_mm"),           " mm",
              test, "rms_ml_mm", stance, subj),
        _card("RMS AP",         r.get("rms_ap_mm"),           " mm",
              test, "rms_ap_mm", stance, subj),
    ]


def _cmj_cards(ctx: ReportContext) -> list[MetricCard]:
    r = ctx.result or {}
    subj = ctx.subject
    height_m = r.get("jump_height_m_impulse", 0) or 0
    height_cm = height_m * 100
    # Build status from norm on meter value
    norm_h = get_norm("cmj", "jump_height_m_impulse",
                      variant="_any", subject=subj)
    status_h = classify_with_direction(height_m, norm_h)
    # Tooltip in cm for readability (convert from m)
    tip_h = ""
    if norm_h is not None:
        tip_h = f"정상 {norm_h.ok_low * 100:.0f}–{norm_h.ok_high * 100:.0f} cm"
    label_h = f"점프 높이  <small style='color:#888;'>{tip_h}</small>"

    return [
        MetricCard(label=label_h,
                   value=_fmt_value(height_cm), unit=" cm", status=status_h),
        _card("최고 Force (BW)", r.get("peak_force_bw"), " ×BW",
              "cmj", "peak_force_bw", "_any", subj),
        _card("Peak Power",     r.get("peak_power_w"), " W",
              "cmj", "peak_power_w", "_any", subj),
        _card("Peak RFD",       r.get("peak_rfd_n_s"), " N/s",
              "cmj", "peak_rfd_n_s", "_any", subj),
        _card("체공 시간",       r.get("flight_time_s"), " s",
              "cmj", "flight_time_s", "_any", subj),
    ]


def _squat_cards(ctx: ReportContext) -> list[MetricCard]:
    r = ctx.result or {}
    reps = r.get("reps") or []
    return [
        _plain_card("반복 횟수", len(reps), ""),
        _plain_card("평균 peak vGRF",
                    r.get("mean_peak_vgrf_bw"), " ×BW"),
        _plain_card("평균 WBA",
                    r.get("mean_wba_pct"), " %"),
    ]


def _encoder_cards(ctx: ReportContext) -> list[MetricCard]:
    r = ctx.result or {}
    reps = r.get("reps") or []
    mean_mcv = None
    mean_rom = None
    if reps:
        mcvs = [rep.get("mean_con_vel_m_s") for rep in reps
                if rep.get("mean_con_vel_m_s") is not None]
        roms = [rep.get("rom_mm") for rep in reps
                if rep.get("rom_mm") is not None]
        if mcvs: mean_mcv = sum(mcvs) / len(mcvs)
        if roms: mean_rom = sum(roms) / len(roms)
    return [
        _plain_card("rep 수", len(reps), ""),
        _plain_card("평균 MCV", mean_mcv, " m/s"),
        _plain_card("평균 ROM", mean_rom, " mm"),
    ]


def _reaction_cards(ctx: ReportContext) -> list[MetricCard]:
    r = ctx.result or {}
    trials = r.get("trials") or []
    return [
        _plain_card("trial 수", len(trials), ""),
        _plain_card("평균 RT", r.get("mean_rt_ms"), " ms"),
        _plain_card("평균 peak 변위",
                    r.get("mean_peak_displacement_mm"), " mm"),
        _plain_card("평균 회복 시간",
                    r.get("mean_recovery_time_s"), " s"),
    ]


def _cognitive_reaction_cards(ctx: ReportContext) -> list[MetricCard]:
    """Top-of-report cards for V6 cognitive reaction. V6-G6 surfaces
    the CRI composite + letter grade as the lead cards so the trainer
    sees the headline metric first."""
    r = ctx.result or {}
    cri = r.get("cri")
    grade = r.get("overall_grade") or "—"
    label_ko = r.get("overall_label_ko") or ""
    grade_str = f"{grade}  ({label_ko})" if label_ko else grade
    cri_str = f"{float(cri):.1f}" if cri is not None else "—"
    return [
        _plain_card("CRI 종합 지수", cri_str, " / 100"),
        _plain_card("종합 등급",     grade_str, ""),
        _plain_card("적중률",        r.get("hit_rate_pct"), " %"),
        _plain_card("평균 RT",       r.get("mean_rt_ms"),   " ms"),
        _plain_card("trial 수",      r.get("n_trials", 0),  ""),
    ]


def _proprio_cards(ctx: ReportContext) -> list[MetricCard]:
    r = ctx.result or {}
    return [
        _plain_card("평균 절대 오차", r.get("mean_absolute_error_mm"), " mm"),
        _plain_card("일정 오차 (CE)", r.get("constant_error_mm"),      " mm"),
        _plain_card("가변 오차 (VE)", r.get("variable_error_mm"),      " mm"),
        _plain_card("trial 수",      len(r.get("trials") or []),        ""),
    ]


def _strength_3lift_cards(ctx: ReportContext) -> list[MetricCard]:
    """Top-of-report cards for V1 strength assessment.

    Uses _plain_card (no automatic ok/caution classification) because
    the per-grade norm comparison is already shown in the dedicated
    band chart further down the report.
    """
    r = ctx.result or {}
    sets = r.get("sets") or []
    n_total = r.get("n_sets", 0) or 0
    n_working = r.get("n_working_sets", 0) or 0
    grade = r.get("grade")
    grade_label = r.get("grade_label") or "—"
    grade_str = (f"{grade} {grade_label}" if grade is not None else "—")
    # Total reps across working sets (informational)
    total_reps = sum(int(s.get("n_reps", 0))
                     for s in sets if not s.get("warmup", False))
    return [
        _plain_card("추정 1RM", r.get("best_1rm_kg"), " kg"),
        _plain_card("등급", grade_str, ""),
        _plain_card("세트", f"{n_working} / {n_total}", ""),
        _plain_card("총 반복 (워밍업 제외)", total_reps, " 회"),
    ]


_CARD_BUILDERS: dict[str, Callable[[ReportContext], list[MetricCard]]] = {
    "balance_eo":     _balance_cards,
    "balance_ec":     _balance_cards,
    "cmj":            _cmj_cards,
    "sj":             _cmj_cards,    # V4 — SJ shares CMJ's metric shape
    "squat":          _squat_cards,
    "overhead_squat": _squat_cards,
    "encoder":        _encoder_cards,
    "reaction":            _reaction_cards,
    "cognitive_reaction":  _cognitive_reaction_cards,
    "proprio":             _proprio_cards,
    "strength_3lift":      _strength_3lift_cards,
}


def _cards_for(ctx: ReportContext) -> list[MetricCard]:
    builder = _CARD_BUILDERS.get(ctx.test_type)
    if builder is None:
        return []
    return builder(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_trainer_report(ctx: ReportContext) -> list[ReportSection]:
    """Full trainer-oriented section list:

        cover → verdict → header → cards → charts → detail →
        pose → history → notes → footer

    The cover page + executive-summary verdict + Page X/Y footer are new
    in the commercial-deliverable polish pass (Phase Q4).
    """
    ctx.audience = "trainer"
    sections: list[ReportSection] = [
        CoverPageSection(),
        ExecutiveSummarySection(_cards_for),
        HeaderSection(),
        SummaryCardsSection(_cards_for),
    ]
    charts = _charts_section_for(ctx)
    if charts is not None:
        sections.append(charts)
    # Squat precision (Phase S1d) — only triggers for squat/overhead_squat,
    # otherwise SquatPrecisionSection.to_pdf_flowables returns [].
    sections.append(SquatPrecisionSection())
    # Strength 3-lift (Phase V1-G) — only triggers for ``strength_3lift``,
    # else returns empty content. Renders a 1-7 grade badge, threshold
    # band, per-set bars + table.
    sections.append(Strength3LiftSection())
    # Composite strength (Phase V3) — multi-session aggregate across
    # the 3 lifts the subject has done, with a body diagram + per-region
    # bars. Renders only for strength_3lift sessions; the section is a
    # no-op for other test types.
    sections.append(StrengthCompositeSection())
    # SSC (Phase V4) — CMJ vs SJ comparison with EUR + SSC%. Section
    # is a no-op when only one of the two jump types has been recorded.
    # Triggers on cmj / sj / strength_3lift report contexts.
    sections.append(SSCSection())
    sections += [
        DetailSection(),
        PoseAnglesSection(),
        HistorySection(),
        NotesSection(),
        FooterSection(),
    ]
    return sections


def build_subject_report(ctx: ReportContext) -> list[ReportSection]:
    """Simplified subject view — cover + verdict + friendlier presentation:

        cover → verdict → header → cards → charts → glossary → footer

    Subject reports skip detail tables + pose tables and instead append
    a plain-language glossary so non-expert readers can interpret terms.
    """
    ctx.audience = "subject"
    sections: list[ReportSection] = [
        CoverPageSection(),
        ExecutiveSummarySection(_cards_for),
        HeaderSection(),
        SummaryCardsSection(_cards_for),
    ]
    charts = _charts_section_for(ctx)
    if charts is not None:
        sections.append(charts)
    # Strength 3-lift on the subject report too — the 1-7 grade is
    # exactly the kind of digestible output a non-expert subject can act
    # on. The section guard keeps it inert for other test types.
    sections.append(Strength3LiftSection())
    sections.append(StrengthCompositeSection())
    sections.append(SSCSection())
    sections.append(GlossarySection())
    sections.append(FooterSection())
    return sections
