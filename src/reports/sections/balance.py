"""Balance charts section — stabilogram + CoP time-series."""
from __future__ import annotations

from io import BytesIO

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import (
    make_cop_timeseries, make_stabilogram, png_data_uri,
)
from src.reports.fonts import pdf_font_family


class BalanceChartsSection(ReportSection):
    """Two charts: 2D stabilogram + CoP ML/AP time-series."""

    def enabled_for(self, audience: str) -> bool:
        # Show in both audiences — subject gets just the stabilogram (see html/pdf)
        return True

    def _load_force(self, ctx: ReportContext):
        try:
            from src.analysis.common import load_force_session
            return load_force_session(ctx.session_dir)
        except Exception:
            return None

    # ── HTML ────────────────────────────────────────────────────────────────
    def to_html(self, ctx: ReportContext) -> str:
        force = self._load_force(ctx)
        if force is None:
            return ""
        import config as _cfg
        stab = make_stabilogram(
            force.cop_x, force.cop_y,
            plate_w_mm=_cfg.PLATE_TOTAL_WIDTH_MM,
            plate_h_mm=_cfg.PLATE_TOTAL_HEIGHT_MM,
            board_w_mm=_cfg.BOARD_WIDTH_MM, board_h_mm=_cfg.BOARD_HEIGHT_MM,
            board1_origin=_cfg.BOARD1_ORIGIN_MM,
            board2_origin=_cfg.BOARD2_ORIGIN_MM,
        )
        parts = [
            "<h2>CoP 시각화</h2>",
            "<div style='display:flex; gap:10px; flex-wrap:wrap;'>",
            f"<img class='chart' src='{png_data_uri(stab)}' "
            f"style='flex:1; min-width:320px; max-width:520px;'>",
        ]
        if ctx.audience != "subject":
            ts = make_cop_timeseries(force.t_s, force.cop_x, force.cop_y)
            parts.append(
                f"<img class='chart' src='{png_data_uri(ts)}' "
                f"style='flex:2; min-width:380px;'>"
            )
        parts.append("</div>")
        return "".join(parts)

    # ── PDF ─────────────────────────────────────────────────────────────────
    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        force = self._load_force(ctx)
        if force is None:
            return []
        import config as _cfg
        from reportlab.platypus import Image, Paragraph, Spacer, KeepTogether
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        family = pdf_font_family()
        h2 = ParagraphStyle("h2", fontName=family, fontSize=12,
                             textColor=HexColor("#1976D2"),
                             spaceBefore=10, spaceAfter=6)

        stab = make_stabilogram(
            force.cop_x, force.cop_y,
            plate_w_mm=_cfg.PLATE_TOTAL_WIDTH_MM,
            plate_h_mm=_cfg.PLATE_TOTAL_HEIGHT_MM,
            board_w_mm=_cfg.BOARD_WIDTH_MM, board_h_mm=_cfg.BOARD_HEIGHT_MM,
            board1_origin=_cfg.BOARD1_ORIGIN_MM,
            board2_origin=_cfg.BOARD2_ORIGIN_MM,
        )
        flowables = [
            Paragraph("CoP 시각화", h2),
            Image(BytesIO(stab), width=90*mm, height=72*mm),
        ]
        if ctx.audience != "subject":
            ts = make_cop_timeseries(force.t_s, force.cop_x, force.cop_y)
            flowables += [
                Spacer(1, 4),
                Image(BytesIO(ts), width=170*mm, height=72*mm),
            ]
        flowables.append(Spacer(1, 6))
        return [KeepTogether(flowables)]
