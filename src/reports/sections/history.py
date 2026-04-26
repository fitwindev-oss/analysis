"""
History section — shows 1~3 trend charts across past sessions of the same
subject / test / variant. Skipped when fewer than 2 sessions exist.

Metrics shown are the first 3 entries of ``KEY_METRICS[test_type]`` for
readability. The X axis is the session date, Y axis the metric value, with
a normal-range band overlay when a NormRange is available.
"""
from __future__ import annotations

import datetime as _dt
from io import BytesIO
from typing import Any, Optional

from src.reports.base import ReportContext, ReportSection, SessionMetrics
from src.reports.charts import make_history_trend, png_data_uri
from src.reports.fonts import pdf_font_family
from src.reports.key_metrics import key_metric_labels
from src.reports.norms import get_norm


def _short_date(iso: str) -> str:
    """ISO timestamp → 'MM-DD' for compact x-axis labels."""
    try:
        return _dt.datetime.fromisoformat(iso).strftime("%m-%d")
    except Exception:
        return (iso or "?")[:10]


class HistorySection(ReportSection):
    """Cross-session trend lines for the top key metrics."""

    def enabled_for(self, audience: str) -> bool:
        # Trainer only — subject's 1-pager omits this to stay focused
        return audience == "trainer"

    # ── HTML ───────────────────────────────────────────────────────────────
    def to_html(self, ctx: ReportContext) -> str:
        history = ctx.history or []
        # Always include current session as the rightmost data point by
        # synthesising one from ctx.result if not already in history
        history = _ensure_current_session(ctx, history)
        if len(history) < 2:
            return (
                "<h2>📈 히스토리 트렌드</h2>"
                "<div style='color:#888; padding:8px;'>"
                "비교 가능한 이전 세션이 없습니다 (2개 이상 필요)."
                "</div>"
            )

        # Oldest → newest for plotting
        history = sorted(history, key=lambda r: r.session_date)
        dates = [_short_date(r.session_date) for r in history]

        labels = key_metric_labels(ctx.test_type)[:3]
        if not labels:
            return ""
        parts: list[str] = ["<h2>📈 히스토리 트렌드 (최근 "
                            f"{len(history)} 세션)</h2>"]
        # Per-metric trend charts
        variant = _variant_for_norm(ctx)
        for key, label, unit in labels:
            values = [r.metrics.get(key) for r in history]
            if sum(1 for v in values if v is not None) < 2:
                continue
            # Norm range for band overlay
            n = get_norm(ctx.test_type, key, variant=variant,
                         subject=ctx.subject)
            norm_range = (n.ok_low, n.ok_high) if n is not None else None
            png = make_history_trend(
                values=values, dates=dates,
                metric_label=label, unit=(" " + unit) if unit else "",
                norm_range=norm_range)
            parts.append(
                f"<img class='chart' src='{png_data_uri(png)}' "
                f"style='max-width:100%; height:auto;'>"
            )
        # Improvement table
        parts.append(_improvement_table_html(history, labels))
        return "".join(parts)

    # ── PDF ────────────────────────────────────────────────────────────────
    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        history = ctx.history or []
        history = _ensure_current_session(ctx, history)
        from reportlab.platypus import Image, Paragraph, Spacer, KeepTogether
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm
        family = pdf_font_family()
        h2 = ParagraphStyle("h2", fontName=family, fontSize=12,
                             textColor=HexColor("#1976D2"),
                             spaceBefore=10, spaceAfter=6)
        body = ParagraphStyle("b", fontName=family, fontSize=9,
                               textColor=HexColor("#666"))
        if len(history) < 2:
            return [Paragraph("📈 히스토리 트렌드", h2),
                    Paragraph("비교 가능한 이전 세션이 없습니다 (2개 이상 필요).",
                              body)]
        history = sorted(history, key=lambda r: r.session_date)
        dates = [_short_date(r.session_date) for r in history]
        labels = key_metric_labels(ctx.test_type)[:3]
        if not labels:
            return []
        variant = _variant_for_norm(ctx)
        flowables: list = [
            Paragraph(f"📈 히스토리 트렌드 (최근 {len(history)} 세션)", h2)
        ]
        for key, label, unit in labels:
            values = [r.metrics.get(key) for r in history]
            if sum(1 for v in values if v is not None) < 2:
                continue
            n = get_norm(ctx.test_type, key, variant=variant,
                         subject=ctx.subject)
            norm_range = (n.ok_low, n.ok_high) if n is not None else None
            png = make_history_trend(
                values=values, dates=dates,
                metric_label=label, unit=(" " + unit) if unit else "",
                norm_range=norm_range)
            flowables.append(Image(BytesIO(png), width=170*mm, height=60*mm))
            flowables.append(Spacer(1, 4))
        # Improvement table (PDF)
        table = _improvement_table_pdf(history, labels, family)
        if table is not None:
            flowables.append(table)
        flowables.append(Spacer(1, 6))
        return flowables


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _variant_for_norm(ctx: ReportContext) -> Optional[str]:
    tt = ctx.test_type
    if tt in ("balance_eo", "balance_ec"):
        return ctx.session_meta.get("stance") or "two"
    return None


def _ensure_current_session(ctx: ReportContext,
                            history: list[SessionMetrics]
                            ) -> list[SessionMetrics]:
    """If the current session is analyzed but not in history (e.g. metrics
    cache not yet populated), synthesize a record from ctx.result so the
    trend includes the point that triggered the report."""
    from src.reports.key_metrics import extract_key_metrics
    session_date = ctx.session_meta.get("record_start_wall_s")
    if session_date is None:
        return history
    try:
        iso = _dt.datetime.fromtimestamp(float(session_date))\
            .astimezone().isoformat(timespec="seconds")
    except Exception:
        return history
    metrics = extract_key_metrics(ctx.test_type, ctx.result or {})
    if not metrics:
        return history
    # Is this session already in history? Compare by date prefix
    existing = any(iso[:10] in h.session_date and
                   abs(float(session_date) - _parse_wall(h.session_date)) < 60
                   for h in history)
    if existing:
        return history
    sid = ctx.session_dir.name
    return history + [SessionMetrics(
        session_id=sid, session_date=iso, metrics=metrics,
    )]


def _parse_wall(iso: str) -> float:
    try:
        return _dt.datetime.fromisoformat(iso).timestamp()
    except Exception:
        return 0.0


def _improvement_table_html(history: list[SessionMetrics],
                             labels: list[tuple[str, str, str]]) -> str:
    """Simple table: metric | first | latest | Δ | Δ%."""
    if len(history) < 2:
        return ""
    first = history[0].metrics
    last  = history[-1].metrics
    rows: list[str] = []
    for key, label, unit in labels:
        a = first.get(key); b = last.get(key)
        if a is None or b is None:
            continue
        delta = float(b) - float(a)
        pct = (delta / float(a) * 100) if a else None
        pct_s = "—" if pct is None else f"{pct:+.1f}%"
        rows.append(
            "<tr>"
            f"<td style='padding:4px 8px;'>{label}</td>"
            f"<td style='padding:4px 8px; text-align:right;'>"
            f"{a:.1f}<span style='color:#888;'> {unit}</span></td>"
            f"<td style='padding:4px 8px; text-align:right; color:#fff; "
            f"font-weight:bold;'>{b:.1f}<span style='color:#888;'> {unit}</span></td>"
            f"<td style='padding:4px 8px; text-align:right; "
            f"color:{'#E53935' if delta > 0 else '#4CAF50'};'>"
            f"{delta:+.1f}</td>"
            f"<td style='padding:4px 8px; text-align:right; color:#90caf9;'>"
            f"{pct_s}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<h3>개선율 (처음 → 최근)</h3>"
        "<table class='report'>"
        "<thead><tr>"
        "<th>지표</th><th style='text-align:right;'>처음</th>"
        "<th style='text-align:right;'>최근</th>"
        "<th style='text-align:right;'>Δ</th>"
        "<th style='text-align:right;'>%</th>"
        "</tr></thead><tbody>"
        + "".join(rows) +
        "</tbody></table>"
    )


def _improvement_table_pdf(history: list[SessionMetrics],
                            labels: list[tuple[str, str, str]],
                            family: str):
    if len(history) < 2:
        return None
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.lib.units import mm
    cell = ParagraphStyle("c", fontName=family, fontSize=9, leading=11)
    hdr  = ParagraphStyle("h", fontName=family, fontSize=9,
                           textColor=HexColor("#FFFFFF"), leading=11)
    first = history[0].metrics
    last  = history[-1].metrics
    data = [[Paragraph("지표", hdr), Paragraph("처음", hdr),
             Paragraph("최근", hdr), Paragraph("Δ", hdr),
             Paragraph("%", hdr)]]
    any_row = False
    for key, label, unit in labels:
        a = first.get(key); b = last.get(key)
        if a is None or b is None:
            continue
        any_row = True
        delta = float(b) - float(a)
        pct = (delta / float(a) * 100) if a else None
        pct_s = "—" if pct is None else f"{pct:+.1f}%"
        data.append([
            Paragraph(label, cell),
            Paragraph(f"{a:.1f} {unit}", cell),
            Paragraph(f"{b:.1f} {unit}", cell),
            Paragraph(f"{delta:+.1f}", cell),
            Paragraph(pct_s, cell),
        ])
    if not any_row:
        return None
    tbl = Table(data, colWidths=[60*mm, 30*mm, 30*mm, 22*mm, 22*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1976D2")),
        ("FONTNAME",   (0, 0), (-1, -1), family),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [HexColor("#FFFFFF"), HexColor("#F7F7F7")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW",     (0, 0), (-1, -1), 0.3, HexColor("#DDDDDD")),
    ]))
    return tbl
