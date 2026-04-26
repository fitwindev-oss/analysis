"""
Composite strength report section (Phase V3).

Aggregates the latest grade per body region across all
``strength_3lift`` sessions for the subject and renders:

  1. Header — subject context + measurement coverage (e.g. "3/7 부위 측정")
  2. Big composite grade badge (1-7) with Korean label and weighted
     percent score (V1-B composite_score formula).
  3. Frontal body diagram with per-region grade fill (measured = colour,
     missing = grey).
  4. Horizontal bar chart of measured regions ranked by 1RM.
  5. Source-session table — which session contributed which region's
     grade, with date.

Applies whenever the active session is ``strength_3lift`` (so it
appears alongside the per-session 1RM section). When the subject has
only one strength_3lift session ever, the composite still renders
(showing 1/7 coverage) — incentive to measure the other regions
later.
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import (
    make_body_strength_diagram, make_strength_per_region_bars,
    png_data_uri,
)
from src.reports.fonts import pdf_font_family


_GRADE_COLORS: dict[int, str] = {
    1: "#26A69A",
    2: "#9CCC65",
    3: "#FBC02D",
    4: "#FFA726",
    5: "#EF5350",
    6: "#C62828",
    7: "#7F0000",
}


def _fmt_date(iso: str) -> str:
    """'2026-04-27T02:30:55+09:00' → '2026-04-27'."""
    if not iso:
        return "—"
    try:
        return iso[:10]
    except Exception:
        return str(iso)


class StrengthCompositeSection(ReportSection):
    """Multi-session composite strength assessment (V3)."""

    def _applicable(self, ctx: ReportContext) -> bool:
        return ctx.test_type == "strength_3lift"

    # ── HTML ────────────────────────────────────────────────────────────
    def to_html(self, ctx: ReportContext) -> str:
        if not self._applicable(ctx):
            return ""
        comp = self._compute(ctx)
        if comp is None:
            return ""
        if comp.skipped_reason:
            return self._render_skipped_html(comp.skipped_reason)

        n_meas = comp.n_measured
        n_total = comp.n_total_regions
        coverage_pct = comp.coverage_weight_pct
        score = comp.composite_score_pct
        grade = comp.composite_grade
        label = comp.composite_label
        badge_color = _GRADE_COLORS.get(grade, "#9E9E9E")

        # Region grade dict for the body diagram.
        region_grades = {r.region: r.grade for r in comp.regions}
        region_one_rm = {r.region: r.one_rm_kg for r in comp.regions}
        diagram_png = make_body_strength_diagram(
            region_grades=region_grades, region_one_rm=region_one_rm)
        bars_png = make_strength_per_region_bars(
            [{"region_label": r.region_label,
              "one_rm_kg":    r.one_rm_kg,
              "grade":        r.grade}
             for r in comp.regions])

        # Source-session rows
        src_rows = ""
        for r in comp.regions:
            src_rows += (
                f"<tr>"
                f"<td>{r.region_label}</td>"
                f"<td>{r.exercise.replace('_', ' ').title()}</td>"
                f"<td style='text-align:right;'>{r.one_rm_kg:.1f} kg</td>"
                f"<td style='text-align:center;'>"
                f"<span style='background:{_GRADE_COLORS.get(r.grade, '#9E9E9E')};"
                f"color:white; padding:2px 8px; border-radius:4px; "
                f"font-weight:bold;'>{r.grade}등급 {r.grade_label}</span></td>"
                f"<td>{_fmt_date(r.session_date)}</td>"
                f"</tr>")
        # Missing regions (informational footer)
        missing_html = ""
        if comp.missing_regions:
            from src.analysis.composite_strength import REGION_LABELS_KO
            missing_labels = [REGION_LABELS_KO.get(m, m)
                              for m in comp.missing_regions]
            missing_html = (
                f"<div style='font-size:11px; color:#888; margin-top:8px;'>"
                f"미측정 부위: {', '.join(missing_labels)}"
                f"</div>"
            )

        return f"""
        <h2 style='margin-top:24px;'>종합 근력 평가</h2>
        <div style='font-size:11px; color:#666; margin-bottom:8px;'>
          이 피험자의 모든 strength_3lift 세션 중 <b>운동별 가장 최근</b>
          결과를 모아 7부위 가중치 식으로 종합한 등급입니다.
          측정 커버리지: <b>{n_meas} / {n_total}</b> 부위
          (가중치 합 {coverage_pct:.0f}/100).
        </div>
        <div style='display:flex; gap:14px; align-items:flex-start;
                     flex-wrap:wrap; margin-bottom:8px;'>
          <div>
            <div style='display:inline-block; background:{badge_color};
                         color:white; padding:14px 22px; border-radius:8px;
                         min-width:140px; text-align:center;'>
              <div style='font-size:11px;'>종합 등급</div>
              <div style='font-size:32px; font-weight:bold;'>{grade}</div>
              <div style='font-size:13px;'>{label}</div>
            </div>
          </div>
          <div style='flex:1; min-width:280px;'>
            <div style='font-size:13px; color:#424242;'>
              종합 점수:
              <b style='font-size:18px; color:#1976D2;'>{score:.1f} %</b>
            </div>
            <div style='font-size:11px; color:#666; margin-top:4px;'>
              부위별 가중치 × 등급 백분율 (1등급 = 100, 7등급 = 35)을
              측정된 부위에 한해 정규화. 7부위 모두 측정 시 가중치 합이 100,
              현재는 {coverage_pct:.0f}로 정규화됩니다.
            </div>
          </div>
        </div>
        <div style='display:flex; gap:12px; align-items:flex-start;
                     flex-wrap:wrap;'>
          <img src='{png_data_uri(diagram_png)}'
               style='max-width:42%; height:auto;' />
          <img src='{png_data_uri(bars_png)}'
               style='max-width:55%; height:auto;' />
        </div>
        <table style='width:100%; border-collapse:collapse;
                       font-size:12px; background:#FAFAFA;
                       border:1px solid #E0E0E0; margin-top:8px;'>
          <thead>
            <tr style='background:#EEEEEE;'>
              <th style='padding:6px 10px; text-align:left;'>부위</th>
              <th style='padding:6px 10px; text-align:left;'>운동</th>
              <th style='padding:6px 10px; text-align:right;'>1RM</th>
              <th style='padding:6px 10px; text-align:center;'>등급</th>
              <th style='padding:6px 10px; text-align:left;'>측정 일자</th>
            </tr>
          </thead>
          <tbody>{src_rows}</tbody>
        </table>
        {missing_html}
        """

    @staticmethod
    def _render_skipped_html(reason: str) -> str:
        return (
            "<h2 style='margin-top:24px;'>종합 근력 평가</h2>"
            f"<div style='background:#F5F5F5; border:1px solid #BDBDBD; "
            f"padding:12px 16px; border-radius:4px; color:#616161; "
            f"font-size:12px;'>"
            f"종합 평가 산출 불가: {reason}"
            f"</div>"
        )

    # ── PDF ─────────────────────────────────────────────────────────────
    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        if not self._applicable(ctx):
            return []
        comp = self._compute(ctx)
        if comp is None:
            return []

        from reportlab.platypus import (
            Image, Paragraph, Spacer, Table, TableStyle,
        )
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        family = pdf_font_family()
        h2 = ParagraphStyle(
            "sc_h2", fontName=family, fontSize=13,
            textColor=HexColor("#1976D2"), spaceBefore=14, spaceAfter=8)
        ctx_style = ParagraphStyle(
            "sc_ctx", fontName=family, fontSize=9,
            textColor=HexColor("#666666"), leading=12)
        body_style = ParagraphStyle(
            "sc_body", fontName=family, fontSize=10,
            textColor=HexColor("#212121"), leading=13)
        skip_style = ParagraphStyle(
            "sc_skip", fontName=family, fontSize=10,
            textColor=HexColor("#616161"),
            backColor=HexColor("#F5F5F5"),
            borderColor=HexColor("#BDBDBD"), borderWidth=0.5,
            borderPadding=6, leading=13)

        flow: list = [Paragraph("종합 근력 평가", h2)]

        if comp.skipped_reason:
            flow.append(Paragraph(
                f"종합 평가 산출 불가: {comp.skipped_reason}", skip_style))
            return flow

        n_meas = comp.n_measured
        n_total = comp.n_total_regions
        coverage_pct = comp.coverage_weight_pct
        score = comp.composite_score_pct
        grade = comp.composite_grade
        label = comp.composite_label
        badge_color = _GRADE_COLORS.get(grade, "#9E9E9E")

        flow.append(Paragraph(
            f"운동별 최근 strength_3lift 결과를 7부위 가중치 식으로 "
            f"종합한 등급입니다. 측정 커버리지: <b>{n_meas} / {n_total}</b> "
            f"부위 (가중치 합 {coverage_pct:.0f}/100).",
            ctx_style))
        flow.append(Spacer(1, 6))

        # Badge + score row
        badge_para = Paragraph(
            f"<para alignment='center'>"
            f"<font size='9'>종합 등급</font><br/>"
            f"<font size='22'><b>{grade}</b></font><br/>"
            f"<font size='10'>{label}</font></para>", body_style)
        score_para = Paragraph(
            f"종합 점수<br/>"
            f"<font size='16' color='#1976D2'><b>{score:.1f} %</b></font><br/>"
            f"<font size='8' color='#666'>부위별 가중치 × 등급 백분율을 "
            f"측정 부위에 한해 정규화 (현재 {coverage_pct:.0f}/100)</font>",
            body_style)
        badge_table = Table([[badge_para, score_para]],
                             colWidths=[40 * mm, 130 * mm])
        badge_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), HexColor(badge_color)),
            ("TEXTCOLOR",     (0, 0), (0, 0), HexColor("#FFFFFF")),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        flow.append(badge_table)
        flow.append(Spacer(1, 6))

        region_grades = {r.region: r.grade for r in comp.regions}
        region_one_rm = {r.region: r.one_rm_kg for r in comp.regions}
        try:
            diagram_png = make_body_strength_diagram(
                region_grades=region_grades, region_one_rm=region_one_rm)
            bars_png = make_strength_per_region_bars(
                [{"region_label": r.region_label,
                  "one_rm_kg":    r.one_rm_kg,
                  "grade":        r.grade}
                 for r in comp.regions])
        except Exception:
            diagram_png = None
            bars_png = None
        if diagram_png and bars_png:
            two_up = Table(
                [[Image(BytesIO(diagram_png), width=70 * mm, height=88 * mm),
                  Image(BytesIO(bars_png),    width=100 * mm, height=50 * mm)]],
                colWidths=[72 * mm, 100 * mm])
            two_up.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",  (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            flow.append(two_up)
            flow.append(Spacer(1, 6))

        # Source-session table
        tbl_rows: list = [["부위", "운동", "1RM", "등급", "측정일"]]
        for r in comp.regions:
            grade_cell = (f"{r.grade}등급 {r.grade_label}")
            tbl_rows.append([
                r.region_label,
                r.exercise.replace("_", " ").title(),
                f"{r.one_rm_kg:.1f} kg",
                grade_cell,
                _fmt_date(r.session_date),
            ])
        tbl = Table(tbl_rows,
                    colWidths=[24 * mm, 38 * mm, 25 * mm, 50 * mm, 28 * mm])
        tbl.setStyle(TableStyle([
            ("FONTNAME",      (0, 0), (-1, -1), family),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("BACKGROUND",    (0, 0), (-1, 0), HexColor("#EEEEEE")),
            ("BACKGROUND",    (0, 1), (-1, -1), HexColor("#FAFAFA")),
            ("BOX",           (0, 0), (-1, -1), 0.4, HexColor("#BDBDBD")),
            ("ALIGN",         (2, 1), (2, -1), "RIGHT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        flow.append(tbl)

        if comp.missing_regions:
            from src.analysis.composite_strength import REGION_LABELS_KO
            missing_labels = [REGION_LABELS_KO.get(m, m)
                              for m in comp.missing_regions]
            flow.append(Spacer(1, 4))
            flow.append(Paragraph(
                f"<font color='#888' size='9'>미측정 부위: "
                f"{', '.join(missing_labels)}</font>",
                ctx_style))

        return flow

    # ── helper ──────────────────────────────────────────────────────────
    def _compute(self, ctx: ReportContext):
        """Run the multi-session aggregator. Returns CompositeStrength
        or None when the context lacks subject_id."""
        from src.analysis.composite_strength import compute_composite_strength
        subject = ctx.subject
        if subject is None:
            # Fall back to the meta's subject_id; some legacy code
            # paths don't pass the Subject row through.
            sid = (ctx.session_meta or {}).get("subject_id")
            if not sid:
                return None
            return compute_composite_strength(
                subject_id=sid,
                subject_name=(ctx.session_meta or {}).get("subject_name"))
        return compute_composite_strength(
            subject_id=subject.id, subject_name=subject.name)
