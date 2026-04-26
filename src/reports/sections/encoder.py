"""Encoder (VBT) charts: velocity bars colored by zone + velocity loss curve."""
from __future__ import annotations

from io import BytesIO

import numpy as np

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import (
    make_rep_metric_bars, make_vbt_velocity_bars, png_data_uri,
)
from src.reports.fonts import pdf_font_family


class EncoderChartsSection(ReportSection):

    def to_html(self, ctx: ReportContext) -> str:
        reps = (ctx.result or {}).get("reps") or []
        if not reps:
            return ""
        mcvs = [r.get("mean_con_vel_m_s") for r in reps]
        png_zone = make_vbt_velocity_bars(mcvs)
        png_rom  = make_rep_metric_bars(
            [r.get("rom_mm") for r in reps],
            metric_label="ROM", unit="mm")
        return (
            "<h2>VBT 분석</h2>"
            f"<img class='chart' src='{png_data_uri(png_zone)}' "
            f"style='max-width:100%; height:auto;'>"
            f"<img class='chart' src='{png_data_uri(png_rom)}' "
            f"style='max-width:100%; height:auto;'>"
        )

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        reps = (ctx.result or {}).get("reps") or []
        if not reps:
            return []
        mcvs = [r.get("mean_con_vel_m_s") for r in reps]
        png_zone = make_vbt_velocity_bars(mcvs)
        png_rom  = make_rep_metric_bars(
            [r.get("rom_mm") for r in reps],
            metric_label="ROM", unit="mm")
        from reportlab.platypus import Image, Paragraph, Spacer, KeepTogether
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm
        family = pdf_font_family()
        h2 = ParagraphStyle("h2", fontName=family, fontSize=12,
                             textColor=HexColor("#1976D2"),
                             spaceBefore=10, spaceAfter=6)
        return [
            Paragraph("VBT 분석", h2),
            KeepTogether([
                Image(BytesIO(png_zone), width=170*mm, height=64*mm),
                Spacer(1, 3),
                Image(BytesIO(png_rom),  width=170*mm, height=54*mm),
                Spacer(1, 6),
            ]),
        ]
