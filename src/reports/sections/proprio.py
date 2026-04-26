"""Proprio section — target-vs-reproduction scatter with error vectors."""
from __future__ import annotations

from io import BytesIO

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import make_proprio_scatter, png_data_uri
from src.reports.fonts import pdf_font_family


class ProprioChartsSection(ReportSection):

    def to_html(self, ctx: ReportContext) -> str:
        trials = (ctx.result or {}).get("trials") or []
        if not trials:
            return ""
        png = make_proprio_scatter(trials)
        return (
            "<h2>목표 vs 재현 위치</h2>"
            f"<img class='chart' src='{png_data_uri(png)}' "
            f"style='max-width:520px; height:auto;'>"
        )

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        trials = (ctx.result or {}).get("trials") or []
        if not trials:
            return []
        png = make_proprio_scatter(trials)
        from reportlab.platypus import Image, Paragraph, Spacer, KeepTogether
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm
        family = pdf_font_family()
        h2 = ParagraphStyle("h2", fontName=family, fontSize=12,
                             textColor=HexColor("#1976D2"),
                             spaceBefore=10, spaceAfter=6)
        return [KeepTogether([
            Paragraph("목표 vs 재현 위치", h2),
            Image(BytesIO(png), width=110*mm, height=105*mm),
            Spacer(1, 6),
        ])]
