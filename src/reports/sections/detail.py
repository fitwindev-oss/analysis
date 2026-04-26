"""
Per-test detail sections — comprehensive metric tables.

Ports the existing HTML from report_viewer.py into the section system so
the same content works for both HTML (rich styling) and PDF (table flowables).
Charts are added later in Phase R4; this module owns the numeric tables.
"""
from __future__ import annotations

from typing import Any, Optional

from src.reports.base import ReportContext, ReportSection
from src.reports.fonts import pdf_font_family


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.{decimals}f}"
        return f"{value:.{decimals+1}f}"
    return str(value)


def _html_kv_table(rows: list[tuple[str, Any, str]]) -> str:
    """rows: list of (label, value, unit)."""
    body = "".join(
        f'<tr><td style="color:#bbb;">{lbl}</td>'
        f'<td style="text-align:right; color:#fff; font-weight:bold;">{_fmt(v)}</td>'
        f'<td style="color:#888;">{u}</td></tr>'
        for lbl, v, u in rows
    )
    return (f'<table class="report">'
            f'<thead><tr><th>지표</th><th style="text-align:right;">값</th>'
            f'<th>단위</th></tr></thead><tbody>{body}</tbody></table>')


def _pdf_kv_table(rows: list[tuple[str, Any, str]]):
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.lib.units import mm
    family = pdf_font_family()
    cell = ParagraphStyle("c", fontName=family, fontSize=9, leading=11)
    hdr = ParagraphStyle("h", fontName=family, fontSize=9,
                          textColor=HexColor("#FFFFFF"), leading=11)
    data = [[Paragraph("지표", hdr),
             Paragraph("값",   hdr),
             Paragraph("단위", hdr)]]
    for lbl, v, u in rows:
        data.append([Paragraph(lbl, cell),
                     Paragraph(_fmt(v), cell),
                     Paragraph(u, cell)])
    tbl = Table(data, colWidths=[90 * mm, 40 * mm, 30 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1976D2")),
        ("ALIGN",      (1, 0), (1, -1), "RIGHT"),
        ("FONTNAME",   (0, 0), (-1, -1), family),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [HexColor("#FFFFFF"), HexColor("#F7F7F7")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LINEBELOW",     (0, 0), (-1, -1), 0.3, HexColor("#DDDDDD")),
    ]))
    return tbl


def _section_html_wrap(title: str, body: str) -> str:
    return f"<h2>{title}</h2>{body}"


def _section_pdf_wrap(title: str, flowables: list) -> list:
    from reportlab.platypus import Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor
    family = pdf_font_family()
    h2 = ParagraphStyle("h2", fontName=family, fontSize=12,
                         textColor=HexColor("#1976D2"),
                         spaceBefore=10, spaceAfter=6)
    return [Paragraph(title, h2), *flowables, Spacer(1, 6)]


# ─────────────────────────────────────────────────────────────────────────────
# Per-test rows builders
# ─────────────────────────────────────────────────────────────────────────────

def _balance_rows(r: dict) -> list[tuple[str, Any, str]]:
    return [
        ("분석 구간 길이",        r.get("duration_s"),              "s"),
        ("유효 샘플 수",           r.get("n_samples"),               ""),
        ("평균 이동 속도",         r.get("mean_velocity_mm_s"),      "mm/s"),
        ("  · ML (좌우)",          r.get("mean_velocity_ml_mm_s"),   "mm/s"),
        ("  · AP (전후)",          r.get("mean_velocity_ap_mm_s"),   "mm/s"),
        ("경로 길이",              r.get("path_length_mm"),          "mm"),
        ("RMS ML",                r.get("rms_ml_mm"),               "mm"),
        ("RMS AP",                r.get("rms_ap_mm"),               "mm"),
        ("Range ML (p5-p95)",     r.get("range_ml_mm"),             "mm"),
        ("Range AP (p5-p95)",     r.get("range_ap_mm"),             "mm"),
        ("95% 타원 면적",          r.get("ellipse95_area_mm2"),      "mm²"),
        ("Sway area rate",        r.get("sway_area_rate_mm2_s"),    "mm²/s"),
        ("CoP 평균 X",            r.get("mean_cop_x_mm"),           "mm"),
        ("CoP 평균 Y",            r.get("mean_cop_y_mm"),           "mm"),
        ("Board1 / Board2",
         (f"{r.get('mean_board1_pct', 0):.1f}% / "
          f"{r.get('mean_board2_pct', 0):.1f}%"), ""),
    ]


def _cmj_rows(r: dict) -> list[tuple[str, Any, str]]:
    return [
        ("체중 (BW)",              r.get("bw_kg"),                   "kg"),
        ("점프 높이 (임펄스)",      (r.get("jump_height_m_impulse", 0) or 0) * 100, "cm"),
        ("점프 높이 (체공시간)",    (r.get("jump_height_m_flight", 0) or 0) * 100,  "cm"),
        ("이륙 속도",              r.get("takeoff_velocity_m_s"),    "m/s"),
        ("최고 Force",             r.get("peak_force_n"),            "N"),
        ("최고 Force (BW 대비)",   r.get("peak_force_bw"),           "×BW"),
        ("Peak RFD",               r.get("peak_rfd_n_s"),            "N/s"),
        ("Net Impulse",            r.get("net_impulse_ns"),          "N·s"),
        ("Peak Power",             r.get("peak_power_w"),            "W"),
        ("체공 시간",              r.get("flight_time_s"),           "s"),
        ("Eccentric 시간",         r.get("eccentric_duration_s"),    "s"),
        ("Concentric 시간",        r.get("concentric_duration_s"),   "s"),
    ]


def _squat_rep_table_html(reps: list[dict]) -> str:
    if not reps:
        return "<div style='color:#888;'>rep 검출 실패</div>"
    head = (f"<table class='report'><thead><tr>"
            f"<th>#</th><th>시작 (s)</th><th>바닥 (s)</th><th>종료 (s)</th>"
            f"<th>Peak vGRF (×BW)</th><th>WBA %</th>"
            f"</tr></thead><tbody>")
    rows_html = []
    for i, rep in enumerate(reps, 1):
        rows_html.append(
            f"<tr><td>{i}</td>"
            f"<td>{_fmt(rep.get('t_start_s'))}</td>"
            f"<td>{_fmt(rep.get('t_bottom_s'))}</td>"
            f"<td>{_fmt(rep.get('t_end_s'))}</td>"
            f"<td>{_fmt(rep.get('peak_vgrf_bw'))}</td>"
            f"<td>{_fmt(rep.get('mean_wba_pct'))}</td></tr>"
        )
    return head + "".join(rows_html) + "</tbody></table>"


def _encoder_rep_table_html(reps: list[dict]) -> str:
    if not reps:
        return "<div style='color:#888;'>rep 검출 실패</div>"
    head = (f"<table class='report'><thead><tr>"
            f"<th>#</th><th>ROM (mm)</th><th>Ecc (s)</th><th>Con (s)</th>"
            f"<th>Mean v (m/s)</th><th>Peak v (m/s)</th>"
            f"</tr></thead><tbody>")
    rows_html = []
    for i, rep in enumerate(reps, 1):
        rows_html.append(
            f"<tr><td>{i}</td>"
            f"<td>{_fmt(rep.get('rom_mm'))}</td>"
            f"<td>{_fmt(rep.get('eccentric_time_s'))}</td>"
            f"<td>{_fmt(rep.get('concentric_time_s'))}</td>"
            f"<td>{_fmt(rep.get('mean_con_vel_m_s'))}</td>"
            f"<td>{_fmt(rep.get('peak_con_vel_m_s'))}</td></tr>"
        )
    return head + "".join(rows_html) + "</tbody></table>"


def _reaction_trial_table_html(trials: list[dict]) -> str:
    if not trials:
        return "<div style='color:#888;'>trial 없음</div>"
    head = (f"<table class='report'><thead><tr>"
            f"<th>#</th><th>응답</th><th>RT (ms)</th>"
            f"<th>Peak Δ (mm)</th><th>회복 (s)</th>"
            f"</tr></thead><tbody>")
    rows_html = []
    for t in trials:
        rows_html.append(
            f"<tr><td>{t.get('trial_idx', 0)}</td>"
            f"<td>{t.get('response_type', '—')}</td>"
            f"<td>{_fmt(t.get('rt_ms'), 0)}</td>"
            f"<td>{_fmt(t.get('peak_displacement_mm'))}</td>"
            f"<td>{_fmt(t.get('recovery_time_s'), 2)}</td></tr>"
        )
    return head + "".join(rows_html) + "</tbody></table>"


def _proprio_rows(r: dict) -> list[tuple[str, Any, str]]:
    return [
        ("평균 절대 오차 (AE)",       r.get("mean_absolute_error_mm"),  "mm"),
        ("일정 오차 (bias, CE)",      r.get("constant_error_mm"),       "mm"),
        ("가변 오차 (VE)",            r.get("variable_error_mm"),       "mm"),
        ("trial 수",                  len(r.get("trials") or []),        ""),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Section dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class DetailSection(ReportSection):
    """Per-test detail table. Dispatches by ctx.test_type."""

    def to_html(self, ctx: ReportContext) -> str:
        tt = ctx.test_type
        r  = ctx.result or {}
        if tt in ("balance_eo", "balance_ec"):
            return _section_html_wrap(
                "정적 밸런스 (CoP 스웨이)", _html_kv_table(_balance_rows(r)))
        if tt == "cmj":
            return _section_html_wrap(
                "CMJ (Counter Movement Jump)", _html_kv_table(_cmj_rows(r)))
        if tt in ("squat", "overhead_squat"):
            reps = r.get("reps") or []
            head = _section_html_wrap(
                f"스쿼트 · {len(reps)} 회 검출",
                f"<div style='color:#bbb; margin-bottom:6px;'>"
                f"평균 peak vGRF: {_fmt(r.get('mean_peak_vgrf_bw'))} ×BW · "
                f"평균 WBA: {_fmt(r.get('mean_wba_pct'))}%"
                f"</div>")
            return head + _squat_rep_table_html(reps)
        if tt == "encoder":
            reps = r.get("reps") or []
            return _section_html_wrap(
                f"엔코더 · {len(reps)} rep 검출", _encoder_rep_table_html(reps))
        if tt == "reaction":
            trials = r.get("trials") or []
            head = _section_html_wrap(
                f"반응 시간 · {len(trials)} trial",
                f"<div style='color:#bbb; margin-bottom:6px;'>"
                f"평균 RT: {_fmt(r.get('mean_rt_ms'), 0)} ms · "
                f"평균 peak 변위: {_fmt(r.get('mean_peak_displacement_mm'))} mm"
                f"</div>")
            return head + _reaction_trial_table_html(trials)
        if tt == "proprio":
            return _section_html_wrap(
                "고유감각 (재현 오차)", _html_kv_table(_proprio_rows(r)))
        # Unknown / generic
        keys = [k for k, v in r.items()
                if isinstance(v, (int, float, str)) or v is None]
        rows = [(k, r.get(k), "") for k in keys]
        return _section_html_wrap("상세 지표", _html_kv_table(rows))

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        tt = ctx.test_type
        r  = ctx.result or {}
        if tt in ("balance_eo", "balance_ec"):
            return _section_pdf_wrap(
                "정적 밸런스 (CoP 스웨이)", [_pdf_kv_table(_balance_rows(r))])
        if tt == "cmj":
            return _section_pdf_wrap(
                "CMJ (Counter Movement Jump)", [_pdf_kv_table(_cmj_rows(r))])
        if tt in ("squat", "overhead_squat"):
            reps = r.get("reps") or []
            summary_rows = [
                ("검출된 rep 수",     len(reps),                       ""),
                ("평균 peak vGRF",    r.get("mean_peak_vgrf_bw"),      "×BW"),
                ("평균 WBA",          r.get("mean_wba_pct"),           "%"),
            ]
            return _section_pdf_wrap(
                f"스쿼트", [_pdf_kv_table(summary_rows)])
        if tt == "encoder":
            reps = r.get("reps") or []
            summary_rows = [("rep 수", len(reps), "")]
            return _section_pdf_wrap(
                "엔코더", [_pdf_kv_table(summary_rows)])
        if tt == "reaction":
            trials = r.get("trials") or []
            summary_rows = [
                ("trial 수",          len(trials),                     ""),
                ("평균 RT",           r.get("mean_rt_ms"),             "ms"),
                ("평균 peak 변위",    r.get("mean_peak_displacement_mm"), "mm"),
            ]
            return _section_pdf_wrap(
                "반응 시간", [_pdf_kv_table(summary_rows)])
        if tt == "proprio":
            return _section_pdf_wrap(
                "고유감각", [_pdf_kv_table(_proprio_rows(r))])
        # Generic fallback
        rows = [(k, v, "") for k, v in (r or {}).items()
                if isinstance(v, (int, float, str)) or v is None]
        if not rows:
            return []
        return _section_pdf_wrap("상세 지표", [_pdf_kv_table(rows)])
