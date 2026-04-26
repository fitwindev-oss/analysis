"""Reaction section — RT histogram + per-response summary."""
from __future__ import annotations

from io import BytesIO

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import make_rt_histogram, png_data_uri
from src.reports.fonts import pdf_font_family


class ReactionChartsSection(ReportSection):

    def to_html(self, ctx: ReportContext) -> str:
        trials = (ctx.result or {}).get("trials") or []
        rts = [t.get("rt_ms") for t in trials
                if t.get("rt_ms") is not None]
        if not rts:
            return ""
        png = make_rt_histogram(rts)
        return (
            "<h2>반응 시간 분포</h2>"
            f"<img class='chart' src='{png_data_uri(png)}' "
            f"style='max-width:100%; height:auto;'>"
        )

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        trials = (ctx.result or {}).get("trials") or []
        rts = [t.get("rt_ms") for t in trials
                if t.get("rt_ms") is not None]
        if not rts:
            return []
        png = make_rt_histogram(rts)
        from reportlab.platypus import Image, Paragraph, Spacer, KeepTogether
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm
        family = pdf_font_family()
        h2 = ParagraphStyle("h2", fontName=family, fontSize=12,
                             textColor=HexColor("#1976D2"),
                             spaceBefore=10, spaceAfter=6)
        return [KeepTogether([
            Paragraph("반응 시간 분포", h2),
            Image(BytesIO(png), width=140*mm, height=56*mm),
            Spacer(1, 6),
        ])]
