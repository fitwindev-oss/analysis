"""
Cognitive reaction section (Phase V6).

For ``test_type == "cognitive_reaction"``: summary table (mean/median RT,
hit rate, mean spatial error, body part) + three charts:

  1. RT histogram with reference band overlay
  2. Per-direction RT bars
  3. Polar accuracy chart (radial = mean spatial error per direction)

The section silently no-ops if applied to a non-cognitive_reaction
session, so report_builder can register it unconditionally.
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import (
    make_cognitive_rt_histogram, make_cognitive_rt_per_target,
    make_cognitive_accuracy_polar, png_data_uri,
)
from src.reports.fonts import pdf_font_family


_BODY_PART_KO: dict[str, str] = {
    "right_hand": "오른손",
    "left_hand":  "왼손",
    "right_foot": "오른발",
    "left_foot":  "왼발",
}

# V6-fix2 — Korean labels for failure_reason histogram
_FAIL_REASON_KO: dict[str, str] = {
    "ok_motion_onset":         "정상 (모션 onset 검출)",
    "ok_proximity_hit":        "정상 (느린 reach, 근접 도달)",
    "out_of_video":            "자극 시점이 비디오 범위 밖",
    "no_visible_kpt":          "추적 부위 visibility 부족",
    "no_visible_kpt_after_onset": "onset 후 추적 부위 사라짐",
    "no_motion_no_hit":        "동작 미검출 + 목표 미도달",
    "post_window_too_short":   "분석 창 너무 짧음 (비디오 끝)",
    "zero_image_size":         "이미지 크기 정보 없음",
    "unknown":                 "알 수 없음",
}


def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return "—"
    try:
        if v != v:           # NaN
            return "—"
        return f"{float(v):.0f} ms"
    except Exception:
        return "—"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    try:
        if v != v:
            return "—"
        return f"{float(v):.1f} %"
    except Exception:
        return "—"


def _fmt_norm(v: Optional[float]) -> str:
    if v is None:
        return "—"
    try:
        if v != v:
            return "—"
        return f"{float(v):.3f}"
    except Exception:
        return "—"


class CognitiveReactionSection(ReportSection):
    """V6 cognitive-reaction summary + diagnostic charts."""

    def _applicable(self, ctx: ReportContext) -> bool:
        return ctx.test_type == "cognitive_reaction"

    # ── HTML ────────────────────────────────────────────────────────────
    def to_html(self, ctx: ReportContext) -> str:
        if not self._applicable(ctx):
            return ""
        result = ctx.result or {}
        trials = result.get("trials") or []
        if not trials:
            return ("<h2>인지 반응 (V6)</h2>"
                    "<p style='color:#999;'>유효한 시행 데이터가 없습니다.</p>")

        rts = [t.get("rt_ms") for t in trials if t.get("rt_ms") is not None]
        per_target = result.get("per_target") or {}
        hit_pct = result.get("hit_rate_pct") or 0.0
        body_part_ko = _BODY_PART_KO.get(
            str(result.get("body_part") or ""), result.get("body_part") or "—")

        # Charts
        try:
            rt_png = make_cognitive_rt_histogram(
                rts, mean_rt_ms=result.get("mean_rt_ms"))
        except Exception:
            rt_png = None
        try:
            acc_png = make_cognitive_accuracy_polar(per_target)
        except Exception:
            acc_png = None
        try:
            per_tgt_png = make_cognitive_rt_per_target(per_target)
        except Exception:
            per_tgt_png = None

        # V6-G6 — CRI summary line on top of the table so the most
        # important number is the first thing the trainer reads.
        cri = float(result.get("cri") or 0.0)
        overall_grade = result.get("overall_grade") or "E"
        overall_label = result.get("overall_label_ko") or "부족"
        gc = result.get("grade_counts") or {}

        # Color the CRI badge by performance band
        if cri >= 85:    cri_color = "#FFB300"  # gold
        elif cri >= 70:  cri_color = "#2E7D32"  # green
        elif cri >= 55:  cri_color = "#F9A825"  # yellow
        elif cri >= 40:  cri_color = "#E64A19"  # orange
        else:            cri_color = "#C62828"  # red

        cri_html = (
            f"<div style='background:#FAFAFA; border:1px solid #E0E0E0; "
            f"padding:10px 14px; margin-bottom:10px; "
            f"display:flex; justify-content:space-between; "
            f"align-items:center; border-radius:6px;'>"
            f"<div>"
            f"<div style='font-size:11px; color:#888;'>"
            f"종합 인지반응 지수 (CRI)</div>"
            f"<div style='font-size:28px; font-weight:bold; color:{cri_color};'>"
            f"{cri:.1f} <span style='font-size:18px;'>"
            f"/ 100</span></div>"
            f"</div>"
            f"<div style='text-align:right;'>"
            f"<div style='font-size:11px; color:#888;'>등급</div>"
            f"<div style='font-size:24px; font-weight:bold; color:{cri_color};'>"
            f"{overall_grade} <span style='font-size:14px; "
            f"color:#666;'>({overall_label})</span></div>"
            f"</div>"
            f"</div>"
        )

        # Grade-counts strip
        grade_chips_html = (
            "<div style='display:flex; gap:8px; margin-bottom:10px; "
            "font-size:11px;'>"
            f"<span style='background:#FFB300; color:#000; padding:3px 8px; "
            f"border-radius:3px;'>탁월 {gc.get('great', 0)}</span>"
            f"<span style='background:#2E7D32; color:#fff; padding:3px 8px; "
            f"border-radius:3px;'>양호 {gc.get('good', 0)}</span>"
            f"<span style='background:#F9A825; color:#000; padding:3px 8px; "
            f"border-radius:3px;'>보통 {gc.get('normal', 0)}</span>"
            f"<span style='background:#E64A19; color:#fff; padding:3px 8px; "
            f"border-radius:3px;'>지연 {gc.get('bad', 0)}</span>"
            f"<span style='background:#9E9E9E; color:#fff; padding:3px 8px; "
            f"border-radius:3px;'>미응답 {gc.get('miss', 0)}</span>"
            "</div>"
        )

        rows = [
            ("추적 부위",         body_part_ko),
            ("위치 수",           f"{int(result.get('n_positions') or 0)} 방향"),
            ("총 시행 / 유효 / 무응답",
             f"{result.get('n_trials', 0)} / "
             f"{result.get('n_valid', 0)} / "
             f"{result.get('n_no_response', 0)}"),
            ("적중률",            _fmt_pct(hit_pct)),
            # V6-G6 — sub-scores
            ("평균 점수 MS (속도+정확도)",
             f"{float(result.get('mean_score') or 0):.1f} / 100"),
            ("정확도 점수 AS",
             f"{float(result.get('accuracy_score') or 0):.1f} / 100"),
            ("일관성 점수 CS (1−CV)",
             f"{float(result.get('consistency_score') or 0):.1f} / 100"),
            ("평균 RT (반응 시간)",  _fmt_ms(result.get("mean_rt_ms"))),
            ("중앙값 RT",          _fmt_ms(result.get("median_rt_ms"))),
            ("평균 MT (이동 시간)",  _fmt_ms(result.get("mean_mt_ms"))),
            ("평균 Total (RT+MT)", _fmt_ms(result.get("mean_total_ms"))),
            ("평균 공간 오차 (정규화)",
             _fmt_norm(result.get("mean_spatial_error_norm"))),
            ("RT 변동계수 CV", f"{float(result.get('cv_rt') or 0):.3f}"),
        ]
        rows_html = "".join(
            f"<tr><td style='padding:4px 10px; color:#555;'>{k}</td>"
            f"<td style='padding:4px 10px; font-weight:600;'>{v}</td></tr>"
            for k, v in rows
        )

        chart_blocks = []
        if rt_png is not None:
            chart_blocks.append(
                f"<img class='chart' src='{png_data_uri(rt_png)}' "
                f"style='max-width:100%; height:auto;'>")
        if per_tgt_png is not None:
            chart_blocks.append(
                f"<img class='chart' src='{png_data_uri(per_tgt_png)}' "
                f"style='max-width:100%; height:auto;'>")
        if acc_png is not None:
            chart_blocks.append(
                f"<img class='chart' src='{png_data_uri(acc_png)}' "
                f"style='max-width:60%; height:auto; "
                f"display:block; margin:0 auto;'>")

        # V6-fix2 — failure-reason histogram. Always render when there
        # are any non-OK reasons OR when n_valid==0 (operator needs to
        # see why the run came up empty).
        fail_counts = result.get("failure_reason_counts") or {}
        non_ok = {k: v for k, v in fail_counts.items()
                  if not k.startswith("ok_")}
        n_valid = int(result.get("n_valid") or 0)
        diag_block = ""
        if non_ok or n_valid == 0:
            rows = []
            for k in sorted(fail_counts.keys()):
                v = fail_counts[k]
                ko = _FAIL_REASON_KO.get(k, k)
                color = "#2E7D32" if k.startswith("ok_") else "#C62828"
                rows.append(
                    f"<tr><td style='padding:3px 8px; color:{color};'>{ko}</td>"
                    f"<td style='padding:3px 8px; font-weight:600;'>{v} 회</td></tr>")
            empty_warn = ""
            if n_valid == 0:
                empty_warn = (
                    "<div style='color:#C62828; margin-top:6px; "
                    "font-size:11px;'>유효 측정 0회 — 분석기가 모든 시행을 "
                    "무응답으로 분류했습니다. 아래 사유 분포를 확인해 "
                    "원인을 파악해주세요.</div>")
            diag_block = (
                "<details style='margin-top:10px;'>"
                "<summary style='font-weight:600; color:#555; "
                "cursor:pointer;'>시행별 분류 사유 (디버그)</summary>"
                f"{empty_warn}"
                "<table style='border-collapse:collapse; font-size:11px; "
                "margin-top:6px; background:#FAFAFA; "
                "border:1px solid #E0E0E0;'>"
                f"{''.join(rows)}"
                "</table></details>")

        return f"""
        <h2 style='margin-top:20px;'>인지 반응 (V6)</h2>
        <p style='color:#666; font-size:12px; margin:0 0 8px;'>
          화면에 표시된 위치로 {body_part_ko}을(를) 이동시키는 시지각·인지·운동
          통합 반응 검사. CRI는 속도(50%) + 정확도(30%) + 일관성(20%)
          가중 합성. RT 임계: 350ms 탁월 / 500ms 양호 / 750ms 보통.
        </p>
        {cri_html}
        {grade_chips_html}
        <table style='width:100%; border-collapse:collapse;
                      font-size:12px; background:#FAFAFA;
                      border:1px solid #E0E0E0; margin-bottom:12px;'>
          {rows_html}
        </table>
        {''.join(chart_blocks)}
        {diag_block}
        """

    # ── PDF ─────────────────────────────────────────────────────────────
    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        if not self._applicable(ctx):
            return []
        result = ctx.result or {}
        trials = result.get("trials") or []

        from reportlab.platypus import (
            Image, Paragraph, Spacer, Table, TableStyle, KeepTogether,
        )
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        family = pdf_font_family()
        h2 = ParagraphStyle(
            "cog_h2", fontName=family, fontSize=13,
            textColor=HexColor("#1976D2"), spaceBefore=14, spaceAfter=8)
        body_style = ParagraphStyle(
            "cog_body", fontName=family, fontSize=10,
            textColor=HexColor("#212121"), leading=13)

        if not trials:
            return [
                Paragraph("인지 반응 (V6)", h2),
                Paragraph("유효한 시행 데이터가 없습니다.", body_style),
            ]

        rts = [t.get("rt_ms") for t in trials if t.get("rt_ms") is not None]
        per_target = result.get("per_target") or {}
        hit_pct = result.get("hit_rate_pct") or 0.0
        body_part_ko = _BODY_PART_KO.get(
            str(result.get("body_part") or ""), result.get("body_part") or "—")

        label_style = ParagraphStyle(
            "cog_lbl", fontName=family, fontSize=9,
            textColor=HexColor("#555555"))
        val_style = ParagraphStyle(
            "cog_val", fontName=family, fontSize=10,
            textColor=HexColor("#212121"))

        tbl_rows = []
        for label, value in [
            ("추적 부위",         body_part_ko),
            ("위치 수",           f"{int(result.get('n_positions') or 0)} 방향"),
            ("총 시행 / 유효 / 무응답",
             f"{result.get('n_trials', 0)} / "
             f"{result.get('n_valid', 0)} / "
             f"{result.get('n_no_response', 0)}"),
            ("적중률",            _fmt_pct(hit_pct)),
            ("평균 RT",           _fmt_ms(result.get("mean_rt_ms"))),
            ("중앙값 RT",         _fmt_ms(result.get("median_rt_ms"))),
            ("평균 MT",           _fmt_ms(result.get("mean_mt_ms"))),
            ("평균 Total (RT+MT)", _fmt_ms(result.get("mean_total_ms"))),
            ("평균 공간 오차 (정규화)",
             _fmt_norm(result.get("mean_spatial_error_norm"))),
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

        # Charts
        try:
            rt_png = make_cognitive_rt_histogram(
                rts, mean_rt_ms=result.get("mean_rt_ms"))
        except Exception:
            rt_png = None
        try:
            per_tgt_png = make_cognitive_rt_per_target(per_target)
        except Exception:
            per_tgt_png = None
        try:
            acc_png = make_cognitive_accuracy_polar(per_target)
        except Exception:
            acc_png = None

        flow: list = [Paragraph("인지 반응 (V6)", h2), table, Spacer(1, 8)]
        if rt_png is not None:
            flow.append(Image(BytesIO(rt_png), width=170 * mm, height=58 * mm))
            flow.append(Spacer(1, 4))
        if per_tgt_png is not None:
            flow.append(
                Image(BytesIO(per_tgt_png), width=170 * mm, height=55 * mm))
            flow.append(Spacer(1, 4))
        if acc_png is not None:
            flow.append(KeepTogether([
                Image(BytesIO(acc_png), width=110 * mm, height=110 * mm),
            ]))
        return flow
