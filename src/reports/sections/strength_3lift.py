"""
Strength 3-lift report section (Phase V1-G).

Renders the V1-F analysis output as both HTML and PDF:

  * Big grade badge (1-7, Korean label) with colour matching zone
  * 1RM threshold band — visual position of subject vs Beginner / Novice /
    Intermediate / Advanced / Elite zones
  * Per-set table — load, reps, est 1RM, reliability flag
  * Per-set bar chart — same data visualised, warmups distinguished
  * Footer notes — reliability caveats, warning text when grade is 6/7

Applies only when ``ctx.test_type == "strength_3lift"``. Returns
empty content (HTML "" / PDF []) for any other test, plus a clear
notice when the grade was skipped (sex / birthdate / no-reps).
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import (
    make_strength_grade_band, make_strength_per_set_bars, png_data_uri,
)
from src.reports.fonts import pdf_font_family


# Grade → primary colour (matches the chart band) for the badge.
_GRADE_COLORS: dict[int, str] = {
    1: "#26A69A",   # 엘리트
    2: "#9CCC65",   # 좋음
    3: "#FBC02D",   # 보통
    4: "#FFA726",   # 나쁨
    5: "#EF5350",   # 위험
    6: "#C62828",   # 경고
    7: "#7F0000",   # 심각
}

# Friendly Korean exercise label for headers.
_EXERCISE_KO: dict[str, str] = {
    "bench_press": "벤치프레스 (상체)",
    "back_squat":  "백스쿼트 (하체)",
    "deadlift":    "데드리프트 (전신)",
}

_RELIABILITY_KO: dict[str, str] = {
    "excellent":  "매우 신뢰",
    "high":       "신뢰",
    "medium":     "보통",
    "low":        "낮음",
    "unreliable": "신뢰 불가",
}


def _fmt_num(v, digits: int = 1, unit: str = "—") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:                          # NaN
            return "—"
        return f"{v:.{digits}f}{unit}"
    return f"{v}{unit}"


class Strength3LiftSection(ReportSection):
    """Phase V1 strength assessment, rendered for ``strength_3lift`` only."""

    def _applicable(self, ctx: ReportContext) -> bool:
        return ctx.test_type == "strength_3lift"

    # ── HTML ────────────────────────────────────────────────────────────
    def to_html(self, ctx: ReportContext) -> str:
        if not self._applicable(ctx):
            return ""
        result = ctx.result or {}
        sets = result.get("sets") or []
        exercise = result.get("exercise") or "?"
        ex_ko = _EXERCISE_KO.get(exercise, exercise)
        best_1rm = result.get("best_1rm_kg")
        grade = result.get("grade")
        label = result.get("grade_label") or "—"
        warning = result.get("warning")
        thresholds = result.get("thresholds_kg")
        skipped = result.get("skipped_grade_reason")

        badge_color = _GRADE_COLORS.get(grade or 0, "#9E9E9E")
        warn_html = ""
        if warning == "caution":
            warn_html = (
                "<div style='background:#FFF3E0; border:1px solid #FFA726; "
                "color:#E65100; padding:6px 10px; margin-top:8px; "
                "border-radius:4px; font-size:11px;'>"
                "⚠ 1RM이 Beginner 임계값의 80~100% 구간 — 단계적 강화 운동을 권장합니다."
                "</div>")
        elif warning == "severe":
            warn_html = (
                "<div style='background:#FFEBEE; border:1px solid #C62828; "
                "color:#B71C1C; padding:6px 10px; margin-top:8px; "
                "border-radius:4px; font-size:11px; font-weight:600;'>"
                "⚠ 심각: 1RM이 Beginner 임계값의 80% 미만입니다. "
                "전문가 평가 후 맞춤 운동 처방을 권장합니다."
                "</div>")

        # Grade badge
        if grade is not None and not skipped:
            badge_html = (
                f"<div style='display:inline-block; "
                f"background:{badge_color}; color:white; "
                f"font-size:32px; font-weight:bold; "
                f"padding:14px 22px; border-radius:8px; "
                f"min-width:120px; text-align:center;'>"
                f"<div style='font-size:11px; font-weight:normal;'>등급</div>"
                f"<div>{grade}</div>"
                f"<div style='font-size:13px; font-weight:normal; "
                f"margin-top:2px;'>{label}</div>"
                f"</div>")
        else:
            reason = skipped or "정보 부족"
            badge_html = (
                f"<div style='display:inline-block; background:#9E9E9E; "
                f"color:white; font-size:14px; padding:14px 22px; "
                f"border-radius:8px; min-width:140px; text-align:center;'>"
                f"<div style='font-weight:bold; font-size:16px;'>등급 산출 불가</div>"
                f"<div style='font-size:11px; margin-top:4px;'>{reason}</div>"
                f"</div>")

        # Threshold band chart (only when grade was computed)
        chart_html = ""
        if thresholds and best_1rm is not None and best_1rm == best_1rm:
            try:
                band_png = make_strength_grade_band(
                    thresholds, float(best_1rm), exercise_label=ex_ko)
                chart_html = (
                    f"<img src='{png_data_uri(band_png)}' "
                    f"style='max-width:100%; height:auto; "
                    f"margin:8px 0;' />")
            except Exception:
                chart_html = ""

        # Per-set bar chart
        per_set_chart = ""
        if sets:
            try:
                bars_png = make_strength_per_set_bars(sets)
                per_set_chart = (
                    f"<img src='{png_data_uri(bars_png)}' "
                    f"style='max-width:100%; height:auto; "
                    f"margin:8px 0;' />")
            except Exception:
                per_set_chart = ""

        # Per-set table
        rows_html = ""
        for s in sets:
            wu_tag = " (워밍업)" if s.get("warmup") else ""
            err = s.get("error")
            one_rm_cell = (_fmt_num(s.get("one_rm_kg"), 1, " kg")
                           if not err else f"<i style='color:#999'>{err}</i>")
            rel_ko = _RELIABILITY_KO.get(s.get("reliability", "unreliable"), "—")
            rows_html += (
                f"<tr>"
                f"<td>{s.get('set_idx', 0) + 1}{wu_tag}</td>"
                f"<td>{_fmt_num(s.get('load_kg'), 1, ' kg')}</td>"
                f"<td>{s.get('n_reps', 0)} 회</td>"
                f"<td>{one_rm_cell}</td>"
                f"<td>{rel_ko}</td>"
                f"</tr>")

        # Subject context
        sex_ko = {"M": "남성", "F": "여성"}.get(result.get("sex"), "—")
        age = result.get("age")
        bw = result.get("bw_kg")
        ctx_html = (
            f"<div style='font-size:11px; color:#666; margin-bottom:8px;'>"
            f"피험자: {sex_ko} · {age or '—'}세 · {bw:.1f} kg<br>"
            f"운동: <b>{ex_ko}</b> · "
            f"세트 {result.get('n_working_sets', 0)}/{result.get('n_sets', 0)} "
            f"(워밍업 제외 / 전체)"
            f"</div>"
        )

        return f"""
        <h2 style='margin-top:24px;'>전신 근력 평가 (1RM)</h2>
        {ctx_html}
        <div style='display:flex; gap:16px; align-items:flex-start;
                     flex-wrap:wrap; margin-bottom:8px;'>
          <div>{badge_html}</div>
          <div style='flex:1; min-width:300px;'>
            <div style='font-size:13px; color:#424242;'>
              추정 최대 1회 무게(1RM):
              <b style='font-size:18px; color:#1976D2;'>
                {_fmt_num(best_1rm, 1, ' kg')}
              </b>
            </div>
            <div style='font-size:11px; color:#666; margin-top:4px;'>
              가장 높은 추정값을 보인 세트 기준 (피로 보정).
              Epley/Brzycki/Lombardi 평균.
            </div>
            {warn_html}
          </div>
        </div>
        {chart_html}
        {per_set_chart}
        <table style='width:100%; border-collapse:collapse;
                       font-size:12px; background:#FAFAFA;
                       border:1px solid #E0E0E0; margin-top:8px;'>
          <thead>
            <tr style='background:#EEEEEE;'>
              <th style='padding:6px 10px; text-align:left;'>세트</th>
              <th style='padding:6px 10px; text-align:right;'>하중</th>
              <th style='padding:6px 10px; text-align:right;'>반복</th>
              <th style='padding:6px 10px; text-align:right;'>추정 1RM</th>
              <th style='padding:6px 10px; text-align:left;'>신뢰도</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """

    # ── PDF ─────────────────────────────────────────────────────────────
    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        if not self._applicable(ctx):
            return []
        result = ctx.result or {}
        sets = result.get("sets") or []
        exercise = result.get("exercise") or "?"
        ex_ko = _EXERCISE_KO.get(exercise, exercise)
        best_1rm = result.get("best_1rm_kg")
        grade = result.get("grade")
        label = result.get("grade_label") or "—"
        warning = result.get("warning")
        thresholds = result.get("thresholds_kg")
        skipped = result.get("skipped_grade_reason")

        from reportlab.platypus import (
            Image, Paragraph, Spacer, Table, TableStyle, KeepTogether,
        )
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        family = pdf_font_family()
        h2 = ParagraphStyle(
            "s3l_h2", fontName=family, fontSize=13,
            textColor=HexColor("#1976D2"), spaceBefore=14, spaceAfter=8)
        ctx_style = ParagraphStyle(
            "s3l_ctx", fontName=family, fontSize=9,
            textColor=HexColor("#666666"), leading=12)
        body_style = ParagraphStyle(
            "s3l_body", fontName=family, fontSize=10,
            textColor=HexColor("#212121"), leading=13)
        warn_style = ParagraphStyle(
            "s3l_warn", fontName=family, fontSize=10,
            textColor=HexColor("#B71C1C"), leading=13,
            borderColor=HexColor("#C62828"), borderWidth=0.6,
            borderPadding=4, backColor=HexColor("#FFEBEE"))
        caution_style = ParagraphStyle(
            "s3l_caution", fontName=family, fontSize=10,
            textColor=HexColor("#E65100"), leading=13,
            borderColor=HexColor("#FFA726"), borderWidth=0.6,
            borderPadding=4, backColor=HexColor("#FFF3E0"))
        badge_value_style = ParagraphStyle(
            "s3l_badge_v", fontName=family, fontSize=22,
            textColor=HexColor("#FFFFFF"), alignment=1,
            leading=26, fontWeight=None)
        badge_label_style = ParagraphStyle(
            "s3l_badge_l", fontName=family, fontSize=9,
            textColor=HexColor("#FFFFFF"), alignment=1, leading=11)

        flow: list = [Paragraph("전신 근력 평가 (1RM)", h2)]

        # Subject context
        sex_ko = {"M": "남성", "F": "여성"}.get(result.get("sex"), "—")
        age = result.get("age")
        bw = result.get("bw_kg")
        flow.append(Paragraph(
            f"피험자: {sex_ko} · {age or '—'}세 · {bw:.1f} kg<br/>"
            f"운동: <b>{ex_ko}</b> · "
            f"세트 {result.get('n_working_sets', 0)}/"
            f"{result.get('n_sets', 0)} (워밍업 제외 / 전체)",
            ctx_style))
        flow.append(Spacer(1, 6))

        # Grade badge as a 1×1 colored Table
        badge_color = _GRADE_COLORS.get(grade or 0, "#9E9E9E")
        if grade is not None and not skipped:
            badge_text = (f"<para alignment='center'>"
                          f"<font size='9'>등급</font><br/>"
                          f"<font size='22'><b>{grade}</b></font><br/>"
                          f"<font size='10'>{label}</font></para>")
        else:
            reason = skipped or "정보 부족"
            badge_text = (f"<para alignment='center'>"
                          f"<font size='12'><b>등급 산출 불가</b></font><br/>"
                          f"<font size='8'>{reason}</font></para>")
        badge_para = Paragraph(badge_text, body_style)
        # 1RM value paragraph alongside the badge
        rm_para = Paragraph(
            f"추정 최대 1회 무게(1RM)<br/>"
            f"<font size='16' color='#1976D2'><b>"
            f"{_fmt_num(best_1rm, 1, ' kg')}</b></font><br/>"
            f"<font size='8' color='#666'>가장 높은 추정값을 보인 세트 기준</font>",
            body_style)
        badge_table = Table(
            [[badge_para, rm_para]],
            colWidths=[40 * mm, 130 * mm])
        badge_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0),
             HexColor(badge_color if grade is not None and not skipped
                       else "#9E9E9E")),
            ("TEXTCOLOR",     (0, 0), (0, 0), HexColor("#FFFFFF")),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",         (0, 0), (0, 0), "CENTER"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", (0, 0), (-1, -1), [4, 4, 4, 4]),
        ]))
        flow.append(badge_table)
        flow.append(Spacer(1, 6))

        # Warning callouts
        if warning == "caution":
            flow.append(Paragraph(
                "⚠ 1RM이 Beginner 임계값의 80~100% 구간 — "
                "단계적 강화 운동을 권장합니다.", caution_style))
            flow.append(Spacer(1, 6))
        elif warning == "severe":
            flow.append(Paragraph(
                "⚠ 심각: 1RM이 Beginner 임계값의 80% 미만입니다. "
                "전문가 평가 후 맞춤 운동 처방을 권장합니다.", warn_style))
            flow.append(Spacer(1, 6))

        # Threshold band chart
        if thresholds and best_1rm is not None and best_1rm == best_1rm:
            try:
                band_png = make_strength_grade_band(
                    thresholds, float(best_1rm), exercise_label=ex_ko)
                flow.append(Image(BytesIO(band_png),
                                  width=170 * mm, height=36 * mm))
                flow.append(Spacer(1, 4))
            except Exception:
                pass

        # Per-set bars
        if sets:
            try:
                bars_png = make_strength_per_set_bars(sets)
                flow.append(Image(BytesIO(bars_png),
                                  width=170 * mm, height=56 * mm))
                flow.append(Spacer(1, 6))
            except Exception:
                pass

        # Per-set table
        tbl_rows = [["세트", "하중", "반복", "추정 1RM", "신뢰도"]]
        for s in sets:
            wu_tag = " (워밍업)" if s.get("warmup") else ""
            err = s.get("error")
            one_rm_cell = (_fmt_num(s.get("one_rm_kg"), 1, " kg")
                           if not err else f"({err})")
            rel_ko = _RELIABILITY_KO.get(s.get("reliability", "unreliable"), "—")
            tbl_rows.append([
                f"{s.get('set_idx', 0) + 1}{wu_tag}",
                _fmt_num(s.get("load_kg"), 1, " kg"),
                f"{s.get('n_reps', 0)} 회",
                one_rm_cell,
                rel_ko,
            ])
        tbl = Table(tbl_rows,
                    colWidths=[28 * mm, 30 * mm, 25 * mm, 35 * mm, 30 * mm])
        tbl.setStyle(TableStyle([
            ("FONTNAME",      (0, 0), (-1, -1), family),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("BACKGROUND",    (0, 0), (-1, 0), HexColor("#EEEEEE")),
            ("BACKGROUND",    (0, 1), (-1, -1), HexColor("#FAFAFA")),
            ("BOX",           (0, 0), (-1, -1), 0.4, HexColor("#BDBDBD")),
            ("LINEBELOW",     (0, 0), (-1, 0), 0.6, HexColor("#9E9E9E")),
            ("ALIGN",         (1, 1), (3, -1), "RIGHT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        flow.append(tbl)
        return flow
