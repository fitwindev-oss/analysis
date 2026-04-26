"""
Audience-agnostic sections shared by every test:

    HeaderSection       title + subject profile + session metadata
    SummaryCardsSection T1 headline cards + traffic-light status
    FooterSection       analysis timestamp + version footer
    NotesSection        trainer-written notes (trainer audience only)
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from src.reports.base import ReportContext, ReportSection
from src.reports.fonts import pdf_font_family
from src.reports.palette import (
    CARD_BG, CARD_BORDER, HEADER_FG, SECTION_FG, REPORT_FG,
    STATUS_OK, STATUS_CAUTION, STATUS_WARNING, STATUS_NEUTRAL,
)


# Human-friendly test name mapping (also used by ReportViewer)
TEST_KO: dict[str, str] = {
    "balance_eo":     "눈 뜨고 밸런스",
    "balance_ec":     "눈 감고 밸런스",
    "cmj":            "CMJ 점프",
    "squat":          "스쿼트",
    "overhead_squat": "오버헤드 스쿼트",
    "encoder":        "엔코더",
    "reaction":       "반응 시간",
    "proprio":        "고유감각",
}
_STANCE_KO = {"two": "양발", "left": "좌측 발", "right": "우측 발"}


def _rich_test_label(test_type: str, meta: dict) -> str:
    base = TEST_KO.get(test_type, test_type)
    if test_type in ("balance_eo", "balance_ec"):
        base += f" · {_STANCE_KO.get(meta.get('stance', 'two'), '양발')}"
    elif test_type == "reaction":
        trig = meta.get("reaction_trigger")
        if trig:
            base += f" · {'수동' if trig == 'manual' else '자동'}"
    elif test_type == "encoder":
        prompt = meta.get("encoder_prompt")
        if prompt:
            base += f" · {prompt[:22]}"
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

class HeaderSection(ReportSection):
    """Top-of-report banner: test name, subject summary, session metadata."""

    def to_html(self, ctx: ReportContext) -> str:
        test_label = _rich_test_label(ctx.test_type, ctx.session_meta)
        subj = ctx.subject
        subj_line = "(피험자 정보 없음)"
        if subj is not None:
            gender = {"M": "남", "F": "여"}.get(getattr(subj, "gender", ""), "")
            subj_line = (f"{subj.name} · {subj.id} · "
                         f"{subj.weight_kg:.1f} kg · "
                         f"{subj.height_cm:.1f} cm"
                         + (f" · {gender}" if gender else ""))
        duration = ctx.session_meta.get("duration_s", 0)
        date_str = ctx.session_meta.get("record_start_wall_s")
        if date_str:
            try:
                date_str = _dt.datetime.fromtimestamp(
                    float(date_str)).strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_str = str(date_str)
        else:
            date_str = ctx.session_dir.name
        audience_badge = ("피험자용" if ctx.audience == "subject"
                          else "트레이너용")
        return f"""
        <div style="padding:6px 2px 14px 2px; border-bottom:2px solid #2E7D32;">
          <h1 style="margin:0; color:{SECTION_FG};">🎯 {test_label}</h1>
          <div style="margin-top:4px; color:{REPORT_FG};">{subj_line}</div>
          <div style="margin-top:2px; color:#888; font-size:12px;">
            {date_str}  ·  {duration:.0f} s 기록  ·
            <span style="color:{HEADER_FG}; font-weight:bold;">{audience_badge}</span>
            리포트
          </div>
        </div>
        """

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor

        family = pdf_font_family()
        title_style = ParagraphStyle(
            "hdr_title", fontName=family, fontSize=18,
            textColor=HexColor("#2E7D32"), spaceAfter=4, leading=22)
        subj_style = ParagraphStyle(
            "hdr_subj", fontName=family, fontSize=11,
            textColor=HexColor("#333333"), spaceAfter=2, leading=14)
        meta_style = ParagraphStyle(
            "hdr_meta", fontName=family, fontSize=9,
            textColor=HexColor("#757575"), spaceAfter=10, leading=12)

        test_label = _rich_test_label(ctx.test_type, ctx.session_meta)
        subj = ctx.subject
        if subj is not None:
            gender = {"M": "남", "F": "여"}.get(getattr(subj, "gender", ""), "")
            subj_line = (f"{subj.name} · ID {subj.id} · "
                         f"{subj.weight_kg:.1f} kg · {subj.height_cm:.1f} cm"
                         + (f" · {gender}" if gender else ""))
        else:
            subj_line = "(피험자 정보 없음)"
        duration = ctx.session_meta.get("duration_s", 0) or 0
        rec_ts = ctx.session_meta.get("record_start_wall_s")
        if rec_ts:
            try:
                date_str = _dt.datetime.fromtimestamp(
                    float(rec_ts)).strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_str = ctx.session_dir.name
        else:
            date_str = ctx.session_dir.name
        audience_label = ("피험자용" if ctx.audience == "subject"
                          else "트레이너용")
        return [
            Paragraph(f"🎯 {test_label}", title_style),
            Paragraph(subj_line, subj_style),
            Paragraph(
                f"{date_str} &nbsp;|&nbsp; {duration:.0f} s 기록 "
                f"&nbsp;|&nbsp; <b>{audience_label}</b> 리포트",
                meta_style),
            Spacer(1, 4),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Summary cards (T1 headline)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricCard:
    label:  str
    value:  str       # pre-formatted ("38.2", "1.71×", "—")
    unit:   str = ""
    status: str = "neutral"    # ok / caution / warning / neutral / unknown

    @classmethod
    def neutral(cls, label: str, value, unit: str = "") -> "MetricCard":
        if value is None:
            return cls(label=label, value="—", unit=unit, status="unknown")
        if isinstance(value, float):
            v = f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
        else:
            v = str(value)
        return cls(label=label, value=v, unit=unit, status="neutral")


class SummaryCardsSection(ReportSection):
    """Top KPI cards. Pass a builder that computes cards from the result."""

    def __init__(self, cards_builder):
        """cards_builder: (ctx) -> list[MetricCard]"""
        self._build = cards_builder

    def to_html(self, ctx: ReportContext) -> str:
        cards = self._build(ctx)
        if not cards:
            return ""
        cards_html = "".join(
            f'<div class="metric-card {c.status}">'
            f'  <div class="label">{c.label}</div>'
            f'  <div class="value">{c.value}'
            f'    <span class="unit">{c.unit}</span>'
            f'  </div>'
            f'</div>'
            for c in cards
        )
        return f"""
        <h2>핵심 지표</h2>
        <div class="card-row">{cards_html}</div>
        """

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        cards = self._build(ctx)
        if not cards:
            return []
        family = pdf_font_family()

        title_style = ParagraphStyle(
            "summ_title", fontName=family, fontSize=14,
            textColor=HexColor("#1976D2"),
            spaceBefore=6, spaceAfter=8)

        # Status → (left-bar color, soft background tint). Tinted
        # backgrounds make the traffic-light state recognisable at arm's
        # length in printed form (matters for clinic wall-mounting).
        status_palette = {
            "ok":       (HexColor(STATUS_OK),       HexColor("#F1F8E9")),
            "caution":  (HexColor(STATUS_CAUTION),  HexColor("#FFF8E1")),
            "warning":  (HexColor(STATUS_WARNING),  HexColor("#FFEBEE")),
            "neutral":  (HexColor(STATUS_NEUTRAL),  HexColor("#F5F5F5")),
            "unknown":  (HexColor("#9E9E9E"),       HexColor("#F5F5F5")),
        }

        label_style = ParagraphStyle(
            "card_lbl", fontName=family, fontSize=8.5,
            textColor=HexColor("#607D8B"), alignment=1, leading=11)
        value_style = ParagraphStyle(
            "card_val", fontName=family, fontSize=20,
            textColor=HexColor("#212121"), alignment=1, leading=24)
        unit_style = ParagraphStyle(
            "card_unit", fontName=family, fontSize=9,
            textColor=HexColor("#757575"), alignment=1, leading=11)

        row = []
        for c in cards:
            cell = [
                Paragraph(c.label, label_style),
                Spacer(1, 2),
                Paragraph(c.value, value_style),
                Paragraph(c.unit or "&nbsp;", unit_style),
            ]
            row.append(cell)
        # Slightly tighter total width so cards don't overflow the
        # branded page frame's new left/right margins (40pt each side).
        col_w = (175 * mm) / max(len(cards), 1)
        tbl = Table([row], colWidths=[col_w] * len(cards))
        style_cmds = [
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ]
        for col, card in enumerate(cards):
            bar_color, bg_color = status_palette.get(
                card.status, status_palette["neutral"])
            # Per-cell background tint (status lane)
            style_cmds.append(("BACKGROUND", (col, 0), (col, 0), bg_color))
            # Fat left bar with status color
            style_cmds.append(("LINEBEFORE", (col, 0), (col, 0), 4, bar_color))
            # Thin dividers between cells for visual separation
            if col < len(cards) - 1:
                style_cmds.append(
                    ("LINEAFTER", (col, 0), (col, 0), 0.3, HexColor("#DDDDDD")))
        tbl.setStyle(TableStyle(style_cmds))
        return [Paragraph("핵심 지표", title_style), tbl, Spacer(1, 12)]


# ─────────────────────────────────────────────────────────────────────────────
# Notes (trainer only, simple placeholder until Phase R8)
# ─────────────────────────────────────────────────────────────────────────────

class NotesSection(ReportSection):
    """Placeholder for trainer notes. Reads meta['notes'] if present."""

    def enabled_for(self, audience: str) -> bool:
        return audience == "trainer"

    def to_html(self, ctx: ReportContext) -> str:
        notes = (ctx.session_meta.get("notes") or "").strip()
        if not notes:
            return ""
        return f"""
        <h2>트레이너 메모</h2>
        <div class="callout">{notes}</div>
        """

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        notes = (ctx.session_meta.get("notes") or "").strip()
        if not notes:
            return []
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        family = pdf_font_family()
        title = ParagraphStyle("notes_t", fontName=family, fontSize=12,
                                textColor=HexColor("#1976D2"), spaceAfter=4)
        body = ParagraphStyle("notes_b", fontName=family, fontSize=10,
                               textColor=HexColor("#333333"), leading=14,
                               leftIndent=8, borderLeftColor=HexColor("#2E7D32"),
                               borderLeftWidth=2, borderPadding=4)
        return [Paragraph("트레이너 메모", title),
                Paragraph(notes, body), Spacer(1, 6)]


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

class FooterSection(ReportSection):
    """Analysis timestamp + generator info."""

    def to_html(self, ctx: ReportContext) -> str:
        import config as _cfg
        analyzed_at = "—"
        try:
            import json
            result_json = ctx.session_dir / "result.json"
            if result_json.exists():
                analyzed_at = json.loads(
                    result_json.read_text(encoding="utf-8")
                ).get("analyzed_at", "—")
        except Exception:
            pass
        gen = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        return f"""
        <hr style="margin-top:24px; border:none; border-top:1px solid #333;">
        <div style="color:#666; font-size:11px; padding:4px 0;">
          분석 시각: {analyzed_at} &nbsp;·&nbsp;
          리포트 생성: {gen} &nbsp;·&nbsp;
          {_cfg.APP_TITLE} v{_cfg.APP_VERSION}
        </div>
        """

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        import config as _cfg
        import json
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        analyzed_at = "—"
        try:
            result_json = ctx.session_dir / "result.json"
            if result_json.exists():
                analyzed_at = json.loads(
                    result_json.read_text(encoding="utf-8")
                ).get("analyzed_at", "—")
        except Exception:
            pass
        gen = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        family = pdf_font_family()
        style = ParagraphStyle("foot", fontName=family, fontSize=8,
                                textColor=HexColor("#888888"), spaceBefore=14)
        return [Spacer(1, 12),
                Paragraph(
                    f"분석 시각: {analyzed_at} &nbsp;|&nbsp; "
                    f"리포트 생성: {gen} &nbsp;|&nbsp; "
                    f"{_cfg.APP_TITLE} v{_cfg.APP_VERSION}",
                    style)]
