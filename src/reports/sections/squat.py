"""Squat / overhead_squat charts: rep-markered force-time + per-rep bars."""
from __future__ import annotations

from io import BytesIO

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import (
    make_force_time_with_reps, make_rep_metric_bars, png_data_uri,
)
from src.reports.fonts import pdf_font_family


class SquatChartsSection(ReportSection):

    def to_html(self, ctx: ReportContext) -> str:
        reps = (ctx.result or {}).get("reps") or []
        if not reps:
            return ""
        try:
            from src.analysis.common import load_force_session
            force = load_force_session(ctx.session_dir)
        except Exception:
            return ""
        png_ts = make_force_time_with_reps(force.t_s, force.total, reps)
        png_bars = make_rep_metric_bars(
            [r.get("peak_vgrf_bw") for r in reps],
            metric_label="Peak vGRF", unit="×BW")
        wba_bars = make_rep_metric_bars(
            [r.get("mean_wba_pct") for r in reps],
            metric_label="WBA", unit="%")
        return (
            "<h2>반복 분석</h2>"
            f"<img class='chart' src='{png_data_uri(png_ts)}' "
            f"style='max-width:100%; height:auto;'>"
            f"<img class='chart' src='{png_data_uri(png_bars)}' "
            f"style='max-width:100%; height:auto;'>"
            f"<img class='chart' src='{png_data_uri(wba_bars)}' "
            f"style='max-width:100%; height:auto;'>"
        )

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        reps = (ctx.result or {}).get("reps") or []
        if not reps:
            return []
        try:
            from src.analysis.common import load_force_session
            force = load_force_session(ctx.session_dir)
        except Exception:
            return []
        png_ts = make_force_time_with_reps(force.t_s, force.total, reps)
        png_bars = make_rep_metric_bars(
            [r.get("peak_vgrf_bw") for r in reps],
            metric_label="Peak vGRF", unit="×BW")
        wba_bars = make_rep_metric_bars(
            [r.get("mean_wba_pct") for r in reps],
            metric_label="WBA", unit="%")
        from reportlab.platypus import Image, Paragraph, Spacer, KeepTogether
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm
        family = pdf_font_family()
        h2 = ParagraphStyle("h2", fontName=family, fontSize=12,
                             textColor=HexColor("#1976D2"),
                             spaceBefore=10, spaceAfter=6)
        return [
            Paragraph("반복 분석", h2),
            KeepTogether([
                Image(BytesIO(png_ts),   width=170*mm, height=64*mm),
                Spacer(1, 3),
                Image(BytesIO(png_bars), width=170*mm, height=54*mm),
                Spacer(1, 3),
                Image(BytesIO(wba_bars), width=170*mm, height=54*mm),
                Spacer(1, 6),
            ]),
        ]
