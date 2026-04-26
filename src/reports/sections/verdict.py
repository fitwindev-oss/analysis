"""
ExecutiveSummarySection — a one-line verdict banner + top-3 headline
metrics, placed right after the header (below the cover in PDF) so the
first thing the reader sees is "🟢 전반적으로 양호" or similar.

The verdict aggregates the same status flags that drive the KPI cards:
if any metric is ``warning`` → overall warning; else any ``caution`` →
caution; else ``ok``. When there are no classifiable norms for this test
(e.g. encoder/free_exercise has no standardised reference ranges yet) we
fall back to a neutral "데이터 요약" banner.

This section is shared by trainer + subject audiences.
"""
from __future__ import annotations

from typing import Callable

from src.reports.base import ReportContext, ReportSection


_VERDICTS = {
    "ok": {
        "bg":    "#E8F5E9",
        "fg":    "#1B5E20",
        "icon":  "🟢",
        "title": "전반적으로 양호",
        "body":  "핵심 지표가 모두 권고 범위 안에 있습니다.",
    },
    "caution": {
        "bg":    "#FFF8E1",
        "fg":    "#E65100",
        "icon":  "🟡",
        "title": "일부 지표 주의",
        "body":  "아래 '핵심 지표' 카드에서 주황색으로 표시된 항목을 확인하세요.",
    },
    "warning": {
        "bg":    "#FFEBEE",
        "fg":    "#B71C1C",
        "icon":  "🔴",
        "title": "개선이 필요한 항목 있음",
        "body":  "빨간색으로 표시된 지표는 권고 범위를 벗어났습니다. "
                 "트레이너와 상의해 개선 계획을 세우세요.",
    },
    "neutral": {
        "bg":    "#ECEFF1",
        "fg":    "#37474F",
        "icon":  "📊",
        "title": "측정 데이터 요약",
        "body":  "이 테스트 유형은 표준 권고 범위가 아직 제공되지 않아 "
                 "지표의 절대값만 표시됩니다. 히스토리 트렌드로 개인 내 "
                 "변화를 추적하세요.",
    },
}


def _aggregate_status(statuses: list[str]) -> str:
    classifiable = [s for s in statuses if s in ("ok", "caution", "warning")]
    if not classifiable:
        return "neutral"
    if "warning" in classifiable:
        return "warning"
    if "caution" in classifiable:
        return "caution"
    return "ok"


class ExecutiveSummarySection(ReportSection):
    """Big verdict banner at the very top of the content area.

    Reuses the same ``cards_builder`` as SummaryCardsSection so the
    verdict is derived from the exact same metrics the reader sees below.
    """

    def __init__(self, cards_builder: Callable):
        self._build = cards_builder

    def _verdict_for(self, ctx: ReportContext) -> dict:
        try:
            cards = self._build(ctx) or []
        except Exception:
            cards = []
        overall = _aggregate_status([c.status for c in cards])
        return _VERDICTS[overall]

    # ── HTML ────────────────────────────────────────────────────────────
    def to_html(self, ctx: ReportContext) -> str:
        v = self._verdict_for(ctx)
        return f"""
        <div style="margin:12px 0 8px 0; padding:14px 18px;
                    background:{v['bg']}; border-left:6px solid {v['fg']};
                    border-radius:4px;">
          <div style="color:{v['fg']}; font-weight:bold; font-size:16px;
                      margin-bottom:4px;">
            {v['icon']} {v['title']}
          </div>
          <div style="color:#444; font-size:12px;">{v['body']}</div>
        </div>
        """

    # ── PDF ─────────────────────────────────────────────────────────────
    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        from src.reports.fonts import pdf_font_family
        v = self._verdict_for(ctx)
        family = pdf_font_family()

        title_style = ParagraphStyle(
            "verdict_title", fontName=family, fontSize=14,
            textColor=HexColor(v["fg"]), leading=17, spaceAfter=2)
        body_style = ParagraphStyle(
            "verdict_body", fontName=family, fontSize=10,
            textColor=HexColor("#444444"), leading=14)

        cell = [
            Paragraph(f"{v['icon']} {v['title']}", title_style),
            Paragraph(v["body"], body_style),
        ]
        tbl = Table([[cell]], colWidths=[170 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), HexColor(v["bg"])),
            ("LINEBEFORE",   (0, 0), (0, 0), 4, HexColor(v["fg"])),
            ("LEFTPADDING",  (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING",   (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ]))
        return [tbl, Spacer(1, 10)]
