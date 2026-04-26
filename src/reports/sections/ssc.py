"""
SSC (Stretch-Shortening Cycle) report section — CMJ vs SJ comparison
(Phase V4).

Renders only when both CMJ and SJ sessions exist for the subject.
Layout:

  1. Header — explanation of EUR / SSC% definitions
  2. Grade badge (1-5) with Korean label and dual-interpretation text
  3. CMJ vs SJ height comparison bar chart
  4. EUR grade-band horizontal visualisation
  5. Source-session table (which CMJ + SJ contributed)

The dual interpretation (해석1/해석2) is selected by the subject's
lower-body 1RM grade — see ``ssc.interpret_ssc``.

Applies for both ``cmj`` and ``sj`` test types so opening either jump's
report shows the SSC composite. Also enabled for ``strength_3lift``
sessions so the comprehensive lower-body assessment surfaces it.
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import (
    make_ssc_jump_comparison, make_ssc_grade_band, png_data_uri,
)
from src.reports.fonts import pdf_font_family


_SSC_GRADE_COLORS: dict[int, str] = {
    1: "#26A69A",
    2: "#9CCC65",
    3: "#FBC02D",
    4: "#FFA726",
    5: "#EF5350",
}

_INTERPRETATION_FOCUS_LABELS: dict[str, str] = {
    "strength": "근력 향상 우선 (해석 1)",
    "elastic":  "탄성 훈련 (해석 2)",
}


def _fmt_date(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        return iso[:10]
    except Exception:
        return str(iso)


class SSCSection(ReportSection):
    """V4 stretch-shortening cycle assessment."""

    def _applicable(self, ctx: ReportContext) -> bool:
        return ctx.test_type in ("cmj", "sj", "strength_3lift")

    # ── HTML ────────────────────────────────────────────────────────────
    def to_html(self, ctx: ReportContext) -> str:
        if not self._applicable(ctx):
            return ""
        ssc = self._compute(ctx)
        if ssc is None:
            return ""
        if ssc.skipped_reason:
            return self._render_skipped_html(ssc.skipped_reason)

        grade = int(ssc.grade or 5)
        badge_color = _SSC_GRADE_COLORS.get(grade, "#9E9E9E")
        focus_label = _INTERPRETATION_FOCUS_LABELS.get(
            ssc.interpretation_focus, ssc.interpretation_focus)

        # Charts
        try:
            comp_png = make_ssc_jump_comparison(
                cmj_height_m=ssc.cmj_height_m or 0.0,
                sj_height_m=ssc.sj_height_m or 0.0,
                eur=ssc.eur or 0.0,
                ssc_pct=ssc.ssc_pct or 0.0,
                grade=grade)
            comp_html = (f"<img src='{png_data_uri(comp_png)}' "
                         f"style='max-width:100%; height:auto;' />")
        except Exception:
            comp_html = ""
        try:
            band_png = make_ssc_grade_band(eur=ssc.eur or 0.0, grade=grade)
            band_html = (f"<img src='{png_data_uri(band_png)}' "
                         f"style='max-width:100%; height:auto; "
                         f"margin-top:6px;' />")
        except Exception:
            band_html = ""

        # 1RM cross-reference note
        if ssc.lower_body_1rm_grade is None:
            xref_note = (
                "하체(백스쿼트) 1RM 등급이 측정되지 않아 기본 해석 "
                "(근력 향상 우선)을 적용했습니다."
            )
        else:
            xref_note = (
                f"하체(백스쿼트) 1RM 등급: <b>{ssc.lower_body_1rm_grade}등급</b> "
                f"→ 해석 모드: <b>{focus_label}</b>"
            )

        return f"""
        <h2 style='margin-top:24px;'>SSC 근 탄성도 평가</h2>
        <div style='font-size:11px; color:#666; margin-bottom:8px;'>
          반동 점프(CMJ)와 정적 출발 점프(SJ)의 높이 차이로 신장-단축
          사이클(SSC) 활용 능력을 평가합니다.<br>
          EUR (Eccentric Utilization Ratio) = CMJ 높이 / SJ 높이.
          SSC% = (CMJ - SJ) / SJ × 100.
        </div>
        <div style='display:flex; gap:14px; align-items:flex-start;
                     flex-wrap:wrap; margin-bottom:8px;'>
          <div style='display:inline-block; background:{badge_color};
                       color:white; padding:14px 22px; border-radius:8px;
                       min-width:140px; text-align:center;'>
            <div style='font-size:11px;'>SSC 등급</div>
            <div style='font-size:32px; font-weight:bold;'>{grade}</div>
            <div style='font-size:13px;'>{ssc.grade_label}</div>
          </div>
          <div style='flex:1; min-width:280px;'>
            <table style='font-size:13px; color:#424242;'>
              <tr><td style='color:#666; padding:2px 8px;'>CMJ 높이</td>
                  <td style='font-weight:bold;'>
                    {(ssc.cmj_height_m or 0) * 100:.1f} cm
                  </td></tr>
              <tr><td style='color:#666; padding:2px 8px;'>SJ 높이</td>
                  <td style='font-weight:bold;'>
                    {(ssc.sj_height_m or 0) * 100:.1f} cm
                  </td></tr>
              <tr><td style='color:#666; padding:2px 8px;'>EUR</td>
                  <td style='font-weight:bold; color:#1976D2;'>
                    {ssc.eur:.2f}
                  </td></tr>
              <tr><td style='color:#666; padding:2px 8px;'>SSC%</td>
                  <td style='font-weight:bold; color:#1976D2;'>
                    {ssc.ssc_pct:.1f} %
                  </td></tr>
            </table>
          </div>
        </div>
        <div style='background:#F5F5F5; padding:8px 12px;
                     border-left:4px solid {badge_color};
                     font-size:12px; color:#212121; margin:6px 0;'>
          {ssc.interpretation or ''}
        </div>
        <div style='font-size:11px; color:#666; margin-bottom:6px;'>
          {xref_note}
        </div>
        {comp_html}
        {band_html}
        <table style='width:100%; border-collapse:collapse;
                       font-size:11px; background:#FAFAFA;
                       border:1px solid #E0E0E0; margin-top:8px;'>
          <thead>
            <tr style='background:#EEEEEE;'>
              <th style='padding:6px 10px; text-align:left;'>점프 종류</th>
              <th style='padding:6px 10px; text-align:right;'>높이</th>
              <th style='padding:6px 10px; text-align:left;'>측정일</th>
              <th style='padding:6px 10px; text-align:left;'>세션 ID</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>CMJ (반동 사용)</td>
                <td style='text-align:right;'>
                  {(ssc.cmj_height_m or 0) * 100:.1f} cm</td>
                <td>{_fmt_date(ssc.cmj_session_date)}</td>
                <td>{ssc.cmj_session_id or '—'}</td></tr>
            <tr><td>SJ (정적 출발)</td>
                <td style='text-align:right;'>
                  {(ssc.sj_height_m or 0) * 100:.1f} cm</td>
                <td>{_fmt_date(ssc.sj_session_date)}</td>
                <td>{ssc.sj_session_id or '—'}</td></tr>
          </tbody>
        </table>
        """

    @staticmethod
    def _render_skipped_html(reason: str) -> str:
        return (
            "<h2 style='margin-top:24px;'>SSC 근 탄성도 평가</h2>"
            f"<div style='background:#F5F5F5; border:1px solid #BDBDBD; "
            f"padding:12px 16px; border-radius:4px; color:#616161; "
            f"font-size:12px;'>"
            f"SSC 평가 산출 불가: {reason}"
            f"</div>"
        )

    # ── PDF ─────────────────────────────────────────────────────────────
    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        if not self._applicable(ctx):
            return []
        ssc = self._compute(ctx)
        if ssc is None:
            return []

        from reportlab.platypus import (
            Image, Paragraph, Spacer, Table, TableStyle,
        )
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        family = pdf_font_family()
        h2 = ParagraphStyle(
            "ssc_h2", fontName=family, fontSize=13,
            textColor=HexColor("#1976D2"), spaceBefore=14, spaceAfter=8)
        ctx_style = ParagraphStyle(
            "ssc_ctx", fontName=family, fontSize=9,
            textColor=HexColor("#666666"), leading=12)
        body_style = ParagraphStyle(
            "ssc_body", fontName=family, fontSize=10,
            textColor=HexColor("#212121"), leading=13)
        skip_style = ParagraphStyle(
            "ssc_skip", fontName=family, fontSize=10,
            textColor=HexColor("#616161"),
            backColor=HexColor("#F5F5F5"),
            borderColor=HexColor("#BDBDBD"), borderWidth=0.5,
            borderPadding=6, leading=13)

        flow: list = [Paragraph("SSC 근 탄성도 평가", h2)]
        if ssc.skipped_reason:
            flow.append(Paragraph(
                f"SSC 평가 산출 불가: {ssc.skipped_reason}", skip_style))
            return flow

        flow.append(Paragraph(
            "반동 점프(CMJ)와 정적 출발 점프(SJ)의 높이 차이로 "
            "신장-단축 사이클(SSC) 활용 능력을 평가합니다.<br/>"
            "EUR = CMJ높이 / SJ높이, "
            "SSC% = (CMJ-SJ)/SJ × 100.",
            ctx_style))
        flow.append(Spacer(1, 6))

        grade = int(ssc.grade or 5)
        badge_color = _SSC_GRADE_COLORS.get(grade, "#9E9E9E")

        badge_para = Paragraph(
            f"<para alignment='center'>"
            f"<font size='9'>SSC 등급</font><br/>"
            f"<font size='22'><b>{grade}</b></font><br/>"
            f"<font size='10'>{ssc.grade_label}</font></para>",
            body_style)
        metrics_para = Paragraph(
            f"CMJ <b>{(ssc.cmj_height_m or 0) * 100:.1f} cm</b> · "
            f"SJ <b>{(ssc.sj_height_m or 0) * 100:.1f} cm</b><br/>"
            f"<font size='12' color='#1976D2'>"
            f"EUR <b>{ssc.eur:.2f}</b> · SSC <b>{ssc.ssc_pct:.1f}%</b>"
            f"</font>",
            body_style)
        badge_table = Table(
            [[badge_para, metrics_para]],
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

        # Interpretation callout
        if ssc.interpretation:
            flow.append(Paragraph(
                ssc.interpretation,
                ParagraphStyle(
                    "ssc_interp", fontName=family, fontSize=10,
                    textColor=HexColor("#212121"),
                    backColor=HexColor("#F5F5F5"),
                    borderColor=HexColor(badge_color), borderWidth=1.0,
                    borderPadding=6, leading=13)))
            flow.append(Spacer(1, 6))

        # 1RM cross-reference note
        focus_label = _INTERPRETATION_FOCUS_LABELS.get(
            ssc.interpretation_focus, ssc.interpretation_focus)
        if ssc.lower_body_1rm_grade is None:
            xref = ("하체(백스쿼트) 1RM 등급이 측정되지 않아 "
                    "기본 해석을 적용했습니다.")
        else:
            xref = (f"하체(백스쿼트) 1RM 등급: "
                    f"{ssc.lower_body_1rm_grade}등급 → "
                    f"해석 모드: {focus_label}")
        flow.append(Paragraph(xref, ctx_style))
        flow.append(Spacer(1, 4))

        # Charts
        try:
            comp_png = make_ssc_jump_comparison(
                cmj_height_m=ssc.cmj_height_m or 0.0,
                sj_height_m=ssc.sj_height_m or 0.0,
                eur=ssc.eur or 0.0,
                ssc_pct=ssc.ssc_pct or 0.0,
                grade=grade)
            flow.append(Image(BytesIO(comp_png),
                              width=140 * mm, height=56 * mm))
            flow.append(Spacer(1, 4))
        except Exception:
            pass
        try:
            band_png = make_ssc_grade_band(
                eur=ssc.eur or 0.0, grade=grade)
            flow.append(Image(BytesIO(band_png),
                              width=160 * mm, height=34 * mm))
            flow.append(Spacer(1, 6))
        except Exception:
            pass

        # Source-session table
        tbl_rows: list = [["점프 종류", "높이", "측정일", "세션 ID"]]
        tbl_rows.append([
            "CMJ (반동 사용)",
            f"{(ssc.cmj_height_m or 0) * 100:.1f} cm",
            _fmt_date(ssc.cmj_session_date),
            ssc.cmj_session_id or "—",
        ])
        tbl_rows.append([
            "SJ (정적 출발)",
            f"{(ssc.sj_height_m or 0) * 100:.1f} cm",
            _fmt_date(ssc.sj_session_date),
            ssc.sj_session_id or "—",
        ])
        tbl = Table(tbl_rows,
                    colWidths=[42 * mm, 26 * mm, 30 * mm, 50 * mm])
        tbl.setStyle(TableStyle([
            ("FONTNAME",      (0, 0), (-1, -1), family),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("BACKGROUND",    (0, 0), (-1, 0), HexColor("#EEEEEE")),
            ("BACKGROUND",    (0, 1), (-1, -1), HexColor("#FAFAFA")),
            ("BOX",           (0, 0), (-1, -1), 0.4, HexColor("#BDBDBD")),
            ("ALIGN",         (1, 1), (1, -1), "RIGHT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        flow.append(tbl)
        return flow

    # ── helper ──────────────────────────────────────────────────────────
    def _compute(self, ctx: ReportContext):
        from src.analysis.ssc import compute_ssc
        subject = ctx.subject
        if subject is None:
            sid = (ctx.session_meta or {}).get("subject_id")
            if not sid:
                return None
            return compute_ssc(
                subject_id=sid,
                subject_name=(ctx.session_meta or {}).get("subject_name"))
        return compute_ssc(subject_id=subject.id, subject_name=subject.name)
