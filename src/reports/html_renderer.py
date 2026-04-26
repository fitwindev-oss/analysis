"""
HTML renderer — turns a list of ReportSection into a self-contained HTML
string suitable for QTextBrowser or a browser.
"""
from __future__ import annotations

from typing import Iterable

from src.reports.base import ReportContext, ReportSection
from src.reports.palette import (
    CARD_BG, CARD_BORDER, HEADER_FG, REPORT_BG, REPORT_FG, SECTION_FG,
)


_CSS = f"""
body {{
    font-family: 'Malgun Gothic', 'NanumGothic', 'Nanum Gothic',
                 'Apple SD Gothic Neo', sans-serif;
    background: {REPORT_BG};
    color: {REPORT_FG};
    padding: 14px;
    line-height: 1.5;
}}
h1 {{ color: {SECTION_FG}; font-size: 20px; margin: 8px 0; }}
h2 {{ color: {HEADER_FG}; font-size: 15px; margin: 18px 0 8px 0;
       border-bottom: 1px solid #333; padding-bottom: 4px; }}
h3 {{ color: #ddd; font-size: 13px; margin-top: 14px; }}
p  {{ margin: 4px 0; }}
small {{ color: #888; }}

/* Metric cards (T1 headline) */
.card-row {{ margin: 8px 0; }}
.metric-card {{
    display: inline-block;
    padding: 8px 14px; margin: 4px;
    background: {CARD_BG}; border: 1px solid {CARD_BORDER};
    border-radius: 6px; min-width: 110px; vertical-align: top;
}}
.metric-card .label {{ color: #888; font-size: 11px; }}
.metric-card .value {{ font-size: 22px; font-weight: bold; color: #fff; }}
.metric-card .unit  {{ color: #bbb; font-size: 11px; }}
.metric-card.ok      {{ border-left: 4px solid #4CAF50; }}
.metric-card.caution {{ border-left: 4px solid #FFB300; }}
.metric-card.warning {{ border-left: 4px solid #E53935; }}
.metric-card.neutral {{ border-left: 4px solid #1976D2; }}

/* Tables */
table.report {{ border-collapse: collapse; width: 100%; margin: 6px 0 12px 0; }}
table.report th {{ background: #222; color: {HEADER_FG};
                   padding: 6px 8px; text-align: left; font-weight: bold; }}
table.report td {{ padding: 4px 8px; border-bottom: 1px solid #222;
                   color: #ddd; }}
table.report tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}

/* Chart image */
.chart {{ display: block; margin: 10px 0; max-width: 100%; height: auto;
          background: white; border: 1px solid #333; }}

/* Status badges in tables */
.badge-ok      {{ color: #4CAF50; font-weight: bold; }}
.badge-caution {{ color: #FFB300; font-weight: bold; }}
.badge-warning {{ color: #E53935; font-weight: bold; }}

/* Callout (hint / warning) boxes */
.callout {{ padding: 8px 12px; margin: 8px 0; border-radius: 4px;
            background: #1a2a1e; border-left: 4px solid {SECTION_FG};
            color: #ddd; }}
.callout.warn {{ background: #2a1a00; border-left-color: #FFB300; }}
.callout.alert {{ background: #3a0000; border-left-color: #E53935; }}
"""


def render_html(sections: Iterable[ReportSection],
                ctx: ReportContext,
                standalone: bool = True) -> str:
    """Render selected sections to an HTML string.

    standalone=True wraps in <html>/<head>/<body> with embedded CSS.
    standalone=False returns just the body fragment (useful when the
    embedding widget has its own CSS scope).
    """
    parts: list[str] = []
    for s in sections:
        if s.enabled_for(ctx.audience):
            parts.append(s.to_html(ctx))
    body = "\n".join(parts)
    if not standalone:
        return body
    return (f"<html><head><meta charset='utf-8'>"
            f"<style>{_CSS}</style></head><body>{body}</body></html>")
