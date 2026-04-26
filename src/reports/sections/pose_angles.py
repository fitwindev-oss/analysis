"""
Shared pose-angle table section (MediaPipe BlazePose, 12 joint angles).

Shown whenever the analyzer's result includes ``pose_mean`` or
``pose_per_cam``. Produces the same 12-row camera-average table and
per-camera breakdown that the app's ReportViewer used to render inline.
"""
from __future__ import annotations

from typing import Any, Optional

from src.reports.base import ReportContext, ReportSection
from src.reports.fonts import pdf_font_family


ANGLE_LABELS: dict[str, str] = {
    "knee_L":     "좌 무릎 (hip-knee-ankle)",
    "knee_R":     "우 무릎 (hip-knee-ankle)",
    "hip_L":      "좌 고관절 (shoulder-hip-knee)",
    "hip_R":      "우 고관절 (shoulder-hip-knee)",
    "ankle_L":    "좌 발목 (knee-ankle-toe)",
    "ankle_R":    "우 발목 (knee-ankle-toe)",
    "shoulder_L": "좌 어깨 (hip-shoulder-elbow)",
    "shoulder_R": "우 어깨 (hip-shoulder-elbow)",
    "elbow_L":    "좌 팔꿈치 (shoulder-elbow-wrist)",
    "elbow_R":    "우 팔꿈치 (shoulder-elbow-wrist)",
    "trunk_lean": "몸통 기울기 (수직 대비)",
    "neck_lean":  "목 기울기 (수직 대비)",
}


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


class PoseAnglesSection(ReportSection):
    """12-angle mean table + per-camera raw table."""

    def __init__(self):
        pass

    def _has_pose(self, ctx: ReportContext) -> bool:
        r = ctx.result or {}
        pm = r.get("pose_mean") or {}
        pp = r.get("pose_per_cam") or {}
        return bool(pm) or bool(pp)

    # ── HTML ────────────────────────────────────────────────────────────────
    def to_html(self, ctx: ReportContext) -> str:
        r = ctx.result or {}
        if not self._has_pose(ctx):
            return (
                "<h2>🎯 2D 관절 각도</h2>"
                "<div style='color:#888; padding:8px;'>"
                "2D 포즈 처리가 실행되지 않았습니다. "
                "'🎯 2D 포즈 처리' 버튼으로 추정을 실행하세요."
                "</div>"
            )
        pose_mean    = r.get("pose_mean")    or {}
        pose_per_cam = r.get("pose_per_cam") or {}

        # Mean table
        rows = [
            "<tr style='background:#222; color:#90caf9;'>"
            "<th style='padding:4px 8px;'>관절</th>"
            "<th style='padding:4px 8px;'>평균 (°)</th>"
            "<th style='padding:4px 8px;'>최소 (°)</th>"
            "<th style='padding:4px 8px;'>최대 (°)</th>"
            "<th style='padding:4px 8px;'>범위 (°)</th></tr>"
        ]
        for key, label in ANGLE_LABELS.items():
            s = pose_mean.get(key) or {}
            rows.append(
                "<tr>"
                f"<td style='padding:3px 8px; color:#ddd;'>{label}</td>"
                f"<td style='padding:3px 8px; text-align:right; color:#fff;"
                f" font-weight:bold;'>{_fmt(s.get('mean'))}</td>"
                f"<td style='padding:3px 8px; text-align:right;'>{_fmt(s.get('min'))}</td>"
                f"<td style='padding:3px 8px; text-align:right;'>{_fmt(s.get('max'))}</td>"
                f"<td style='padding:3px 8px; text-align:right; color:#90caf9;'>"
                f"{_fmt(s.get('range'))}</td></tr>"
            )
        mean_table = (
            "<table class='report' style='background:#1a1a1a;'>"
            + "".join(rows) + "</table>"
        )

        per_cam_html = ""
        if pose_per_cam:
            cams = sorted(pose_per_cam.keys())
            head = "".join(
                f"<th style='padding:4px 8px;'>{c}</th>" for c in cams)
            per_rows = [
                "<tr style='background:#222; color:#bbb;'>"
                f"<th style='padding:4px 8px;'>관절 (평균 °)</th>{head}</tr>"
            ]
            for key, label in ANGLE_LABELS.items():
                cells = []
                for c in cams:
                    s = (pose_per_cam.get(c) or {}).get(key) or {}
                    cells.append(
                        f"<td style='padding:3px 8px; text-align:right;'>"
                        f"{_fmt(s.get('mean'))}</td>")
                per_rows.append(
                    "<tr>"
                    f"<td style='padding:3px 8px; color:#aaa;'>{label}</td>"
                    + "".join(cells) + "</tr>")
            per_cam_html = (
                "<div style='color:#888; margin-top:8px; font-size:11px;'>"
                "카메라별 원본 평균값</div>"
                "<table class='report' style='background:#1a1a1a;'>"
                + "".join(per_rows) + "</table>"
            )
        return "<h2>🎯 2D 관절 각도 (카메라 평균)</h2>" + mean_table + per_cam_html

    # ── PDF ─────────────────────────────────────────────────────────────────
    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm

        r = ctx.result or {}
        family = pdf_font_family()
        h2 = ParagraphStyle("h2", fontName=family, fontSize=12,
                             textColor=HexColor("#1976D2"),
                             spaceBefore=10, spaceAfter=6)
        body_small = ParagraphStyle("body_small", fontName=family, fontSize=9,
                                     textColor=HexColor("#666"))

        if not self._has_pose(ctx):
            return [Paragraph("🎯 2D 관절 각도", h2),
                    Paragraph("2D 포즈 처리가 실행되지 않았습니다.",
                              body_small)]

        pose_mean = r.get("pose_mean") or {}
        cell = ParagraphStyle("c", fontName=family, fontSize=9, leading=11)
        hdr  = ParagraphStyle("h", fontName=family, fontSize=9,
                               textColor=HexColor("#FFFFFF"), leading=11)
        data = [[Paragraph("관절",      hdr), Paragraph("평균 (°)", hdr),
                 Paragraph("최소 (°)", hdr), Paragraph("최대 (°)", hdr),
                 Paragraph("범위 (°)", hdr)]]
        for key, label in ANGLE_LABELS.items():
            s = pose_mean.get(key) or {}
            data.append([
                Paragraph(label,              cell),
                Paragraph(_fmt(s.get("mean")), cell),
                Paragraph(_fmt(s.get("min")),  cell),
                Paragraph(_fmt(s.get("max")),  cell),
                Paragraph(_fmt(s.get("range")), cell),
            ])
        tbl = Table(data, colWidths=[80*mm, 20*mm, 20*mm, 20*mm, 20*mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1976D2")),
            ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
            ("FONTNAME",   (0, 0), (-1, -1), family),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                [HexColor("#FFFFFF"), HexColor("#F7F7F7")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("LINEBELOW",     (0, 0), (-1, -1), 0.3, HexColor("#DDDDDD")),
        ]))
        return [Paragraph("🎯 2D 관절 각도 (카메라 평균)", h2), tbl,
                Spacer(1, 6)]
