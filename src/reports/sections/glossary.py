"""
GlossarySection — subject-facing glossary of technical terms.

The trainer report expects the reader to know what "peak vGRF" or "MCV"
means. The subject report's audience often doesn't, so we append a short
plain-language glossary keyed to the test type.

Only rendered for ``audience == "subject"``.
"""
from __future__ import annotations

from src.reports.base import ReportContext, ReportSection


# Per-test glossary. Each entry: (term, plain-language explanation).
# Keep explanations under ~50 Korean characters for readability.
_GLOSSARY: dict[str, list[tuple[str, str]]] = {
    "balance_eo": [
        ("CoP", "Center of Pressure. 발바닥이 바닥에 가하는 힘의 중심점."),
        ("경로 길이", "측정 시간 동안 CoP가 이동한 총 거리. 짧을수록 안정적."),
        ("95% 타원 면적",
         "CoP 이동의 95%가 담기는 타원 면적. 작을수록 조절 능력 우수."),
        ("RMS ML / AP",
         "좌우(ML) / 앞뒤(AP) 흔들림의 평균적 크기. 작을수록 좋음."),
        ("평균 이동 속도",
         "CoP가 1초당 이동한 평균 거리 (mm/s). 빠를수록 자세 조절이 불안정."),
    ],
    "balance_ec": [
        ("CoP", "Center of Pressure. 발바닥이 바닥에 가하는 힘의 중심점."),
        ("경로 길이", "측정 시간 동안 CoP가 이동한 총 거리. 짧을수록 안정적."),
        ("95% 타원 면적",
         "CoP 이동의 95%가 담기는 타원 면적. 작을수록 조절 능력 우수."),
        ("눈 감은 조건의 의미",
         "시각 정보 없이 평형을 유지하는 능력. 전정기관/체성감각 기여도 확인."),
    ],
    "cmj": [
        ("vGRF", "Vertical Ground Reaction Force. 바닥을 누르는 수직 힘 (N)."),
        ("BW", "Bodyweight. 자기 체중 × 중력 (체중의 배수 표기)."),
        ("점프 높이", "이륙 시 수직 속도로부터 계산한 최대 비행 높이 (m)."),
        ("peak Power",
         "점프 추진 구간에서 기록된 최대 힘 × 속도 (W)."),
        ("RFD",
         "Rate of Force Development. 힘이 얼마나 빠르게 증가했는지 (N/s)."),
        ("체공 시간", "발이 바닥에서 떨어져 있던 시간 (s). 길수록 점프 높이 큼."),
        ("이륙 속도",
         "발이 떨어지는 순간의 수직 속도 (m/s). 점프 퍼포먼스의 핵심."),
    ],
    "squat": [
        ("vGRF peak",
         "스쿼트 중 기록된 최대 수직 힘. 체중의 배수(×BW)로 표시."),
        ("WBA",
         "Weight Bearing Asymmetry. 좌우 체중 분배 차이 (%). 작을수록 균형."),
        ("반복 횟수", "분석이 인식한 완료된 rep 수."),
    ],
    "overhead_squat": [
        ("vGRF peak", "스쿼트 중 최대 수직 힘 (×BW)."),
        ("WBA",
         "Weight Bearing Asymmetry. 좌우 체중 분배 차이 (%)."),
        ("오버헤드 스쿼트 특성",
         "팔을 머리 위로 들어올린 자세로 스쿼트 — 어깨/코어 가동성 평가."),
    ],
    "reaction": [
        ("RT", "Reaction Time. 자극 제시부터 움직임 시작까지 걸린 시간 (ms)."),
        ("peak 변위",
         "반응 동작 중 CoP가 목표 방향으로 움직인 최대 거리 (mm)."),
        ("회복 시간",
         "반응 이후 안정 자세로 돌아오는 데 걸린 시간 (s)."),
        ("trial 수",
         "성공적으로 기록된 개별 자극-반응 쌍의 수."),
    ],
    "proprio": [
        ("절대 오차",
         "목표 위치와 실제 위치의 평균 차이 (mm). 작을수록 정확."),
        ("Constant Error (CE)",
         "부호를 포함한 평균 오차 — 일관되게 한쪽으로 치우쳤는지 평가."),
        ("Variable Error (VE)",
         "시행 간 오차의 표준편차 — 일관성 척도."),
        ("trial 수", "완료된 위치 매칭 시도 수."),
    ],
    "free_exercise": [
        ("MCV",
         "Mean Concentric Velocity. 미는 구간에서 바벨의 평균 속도 (m/s)."),
        ("ROM",
         "Range of Motion. 바벨이 1 rep 동안 움직인 거리 (mm)."),
        ("peak 속도",
         "1 rep 중 기록된 최고 순간 속도 (m/s)."),
        ("peak power / mean power",
         "힘 × 속도. 파워가 높을수록 폭발적 움직임."),
        ("자중 하중",
         "외부 무게 없이 본인 체중을 저항으로 사용 (푸쉬업, 풀업)."),
        ("rep",
         "Repetition. 완전히 내려갔다가 올라온 한 번의 동작."),
    ],
}


class GlossarySection(ReportSection):
    """Subject-audience-only glossary table."""

    def enabled_for(self, audience: str) -> bool:
        return audience == "subject"

    def _entries(self, ctx: ReportContext) -> list[tuple[str, str]]:
        return _GLOSSARY.get(ctx.test_type, [])

    def to_html(self, ctx: ReportContext) -> str:
        entries = self._entries(ctx)
        if not entries:
            return ""
        rows = "".join(
            f"<tr><td style='padding:4px 8px; color:#1976D2; "
            f"font-weight:bold; width:30%; vertical-align:top;'>{term}</td>"
            f"<td style='padding:4px 8px; color:#444;'>{desc}</td></tr>"
            for term, desc in entries
        )
        return f"""
        <h2 style='margin-top:24px;'>용어 해설</h2>
        <table style='width:100%; border-collapse:collapse;
                      font-size:12px; background:#FAFAFA;
                      border:1px solid #E0E0E0;'>
          {rows}
        </table>
        """

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        entries = self._entries(ctx)
        if not entries:
            return []

        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        from src.reports.fonts import pdf_font_family
        family = pdf_font_family()

        title_style = ParagraphStyle(
            "glos_title", fontName=family, fontSize=13,
            textColor=HexColor("#1976D2"), spaceBefore=14, spaceAfter=6)
        term_style = ParagraphStyle(
            "glos_term", fontName=family, fontSize=10,
            textColor=HexColor("#1976D2"), leading=13)
        desc_style = ParagraphStyle(
            "glos_desc", fontName=family, fontSize=10,
            textColor=HexColor("#333333"), leading=14)

        rows = [[Paragraph(t, term_style), Paragraph(d, desc_style)]
                for t, d in entries]
        tbl = Table(rows, colWidths=[45 * mm, 125 * mm])
        tbl.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#FAFAFA")),
            ("BOX",           (0, 0), (-1, -1), 0.3, HexColor("#E0E0E0")),
            ("LINEBELOW",     (0, 0), (-1, -2), 0.3, HexColor("#EEEEEE")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))

        return [Paragraph("용어 해설", title_style), tbl, Spacer(1, 8)]
