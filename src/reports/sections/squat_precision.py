"""
Squat precision-metrics section (patent 2 §4, Phase S1d).

Supplements the existing SquatChartsSection with:

  1. CoP overlay (all reps + mean trajectory) — visual proxy for CMC
  2. RFD-intervals bar chart (0-20 / 0-40 / … / 0-100 ms)
  3. Summary table: CMC (AP/ML), mean RMSE, Tempo, Impulse asymmetry,
     peak RFD, VRT

Runs only for ``test_type`` in {squat, overhead_squat}. If the session
has no reps (cancelled or detection failed), the whole section skips
rendering without raising.
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import (
    make_rfd_intervals_bar, make_squat_cop_overlay, png_data_uri,
)
from src.reports.fonts import pdf_font_family


def _fmt(v: Optional[float], digits: int = 2, unit: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if digits == 0:
            return f"{v:.0f}{unit}"
        return f"{v:.{digits}f}{unit}"
    return str(v)


def _fmt_cmc(v: Optional[float]) -> str:
    if v is None:
        return "—"
    # CMC > 0.86 = 우수, 0.66-0.85 = 양호, 0.51-0.65 = 보통, else 나쁨
    if v >= 0.86:
        grade = "우수"
    elif v >= 0.66:
        grade = "양호"
    elif v >= 0.51:
        grade = "보통"
    else:
        grade = "부족"
    return f"{v:.2f}  ({grade})"


class SquatPrecisionSection(ReportSection):
    """Patent 2 §4 precision metrics, rendered only for squat variants."""

    def _applicable(self, ctx: ReportContext) -> bool:
        return ctx.test_type in ("squat", "overhead_squat")

    # ── HTML ────────────────────────────────────────────────────────────
    def to_html(self, ctx: ReportContext) -> str:
        if not self._applicable(ctx):
            return ""
        result = ctx.result or {}
        reps = result.get("reps") or []
        if not reps:
            return ""
        try:
            from src.analysis.common import load_force_session
            force = load_force_session(ctx.session_dir)
        except Exception:
            return ""

        cop_png = make_squat_cop_overlay(
            force.t_s, force.cop_x, force.cop_y, reps)
        rfd_png = make_rfd_intervals_bar(
            [r.get("rfd_n_s") or {} for r in reps])

        # Summary table
        rows = [
            ("CMC (AP, 앞뒤 일관성)", _fmt_cmc(result.get("cmc_ap"))),
            ("CMC (ML, 좌우 일관성)", _fmt_cmc(result.get("cmc_ml"))),
            ("평균 RMSE (AP)",        _fmt(result.get("mean_rmse_ap_mm"),  1, " mm")),
            ("평균 RMSE (ML)",        _fmt(result.get("mean_rmse_ml_mm"),  1, " mm")),
            ("평균 Tempo (E:C)",      _fmt(result.get("mean_tempo_ratio"), 2)),
            ("하강 좌우 비대칭",      _fmt(result.get("mean_impulse_asym_ecc_pct"), 1, " %")),
            ("상승 좌우 비대칭",      _fmt(result.get("mean_impulse_asym_con_pct"), 1, " %")),
            ("평균 peak RFD",         _fmt(result.get("mean_peak_rfd_n_s"), 0, " N/s")),
            ("평균 VRT",              _fmt(result.get("mean_vrt_ms"), 0, " ms")),
        ]
        rows_html = "".join(
            f"<tr><td style='padding:4px 10px; color:#555;'>{k}</td>"
            f"<td style='padding:4px 10px; font-weight:600;'>{v}</td></tr>"
            for k, v in rows
        )
        return f"""
        <h2 style='margin-top:20px;'>정밀 동작 분석</h2>
        <table style='width:100%; border-collapse:collapse;
                      font-size:12px; background:#FAFAFA;
                      border:1px solid #E0E0E0; margin-bottom:12px;'>
          {rows_html}
        </table>
        <div style='display:flex; gap:12px; align-items:flex-start;
                    flex-wrap:wrap;'>
          <img class='chart' src='{png_data_uri(cop_png)}'
               style='max-width:48%; height:auto;'>
          <img class='chart' src='{png_data_uri(rfd_png)}'
               style='max-width:48%; height:auto;'>
        </div>
        """

    # ── PDF ─────────────────────────────────────────────────────────────
    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        if not self._applicable(ctx):
            return []
        result = ctx.result or {}
        reps = result.get("reps") or []
        if not reps:
            return []
        try:
            from src.analysis.common import load_force_session
            force = load_force_session(ctx.session_dir)
        except Exception:
            return []

        from reportlab.platypus import (
            Image, Paragraph, Spacer, Table, TableStyle, KeepTogether,
        )
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        family = pdf_font_family()
        h2 = ParagraphStyle(
            "sp_h2", fontName=family, fontSize=13,
            textColor=HexColor("#1976D2"), spaceBefore=14, spaceAfter=8)
        label_style = ParagraphStyle(
            "sp_lbl", fontName=family, fontSize=9,
            textColor=HexColor("#555555"))
        val_style = ParagraphStyle(
            "sp_val", fontName=family, fontSize=10,
            textColor=HexColor("#212121"))

        # Table rows
        tbl_rows: list[list] = []
        for label, value in [
            ("CMC (AP, 앞뒤 일관성)", _fmt_cmc(result.get("cmc_ap"))),
            ("CMC (ML, 좌우 일관성)", _fmt_cmc(result.get("cmc_ml"))),
            ("평균 RMSE (AP)",        _fmt(result.get("mean_rmse_ap_mm"),  1, " mm")),
            ("평균 RMSE (ML)",        _fmt(result.get("mean_rmse_ml_mm"),  1, " mm")),
            ("평균 Tempo (E:C)",      _fmt(result.get("mean_tempo_ratio"), 2)),
            ("하강 좌우 비대칭",      _fmt(result.get("mean_impulse_asym_ecc_pct"), 1, " %")),
            ("상승 좌우 비대칭",      _fmt(result.get("mean_impulse_asym_con_pct"), 1, " %")),
            ("평균 peak RFD",         _fmt(result.get("mean_peak_rfd_n_s"), 0, " N/s")),
            ("평균 VRT",              _fmt(result.get("mean_vrt_ms"), 0, " ms")),
        ]:
            tbl_rows.append([
                Paragraph(label, label_style),
                Paragraph(value, val_style),
            ])
        table = Table(tbl_rows, colWidths=[85 * mm, 85 * mm])
        table.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#FAFAFA")),
            ("BOX",           (0, 0), (-1, -1), 0.3, HexColor("#E0E0E0")),
            ("LINEBELOW",     (0, 0), (-1, -2), 0.3, HexColor("#EEEEEE")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))

        cop_png = make_squat_cop_overlay(
            force.t_s, force.cop_x, force.cop_y, reps)
        rfd_png = make_rfd_intervals_bar(
            [r.get("rfd_n_s") or {} for r in reps])

        return [
            Paragraph("정밀 동작 분석", h2),
            table,
            Spacer(1, 8),
            KeepTogether([
                Image(BytesIO(cop_png), width=85 * mm, height=68 * mm),
                Spacer(1, 4),
                Image(BytesIO(rfd_png), width=170 * mm, height=50 * mm),
                Spacer(1, 6),
            ]),
        ]
