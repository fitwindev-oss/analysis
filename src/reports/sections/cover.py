"""
CoverPageSection — the deliverable first page.

Renders a single full-bleed cover page with:

  - Clinic logo (if ``config.REPORT_LOGO_PATH`` set) or text-only masthead
  - Big title "바이오메카닉스 측정 리포트"
  - Subject info block (name / ID / anthropometry)
  - Test info (label + date + duration + load/stance/etc.)
  - Trainer name (from subject record)
  - Clinic footer

In HTML mode this section renders nothing — the HTML viewer has its own
in-page header, so a cover page would just be awkward whitespace. The
section lives for PDF output only.

The cover page is detected by the branded-page frame painter and skipped
(so the header/footer band doesn't overlap with the cover content).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import config
from src.reports.base import ReportContext, ReportSection
from src.reports.fonts import pdf_font_family
from src.reports.sections.common import TEST_KO, _rich_test_label


def _cover_marker_flowable():
    """Build a zero-size Flowable that flags the canvas as having a cover.

    Lazy-built because importing reportlab at module top causes side
    effects in environments without a display.
    """
    from reportlab.platypus import Flowable

    class _CoverMarker(Flowable):
        """Zero-height flowable that tags the canvas on draw."""

        def __init__(self):
            super().__init__()
            self.width = 0
            self.height = 0

        def wrap(self, _avail_w, _avail_h):
            return (0, 0)

        def draw(self):
            # ``self.canv`` is set by the Frame when drawing.
            canvas = getattr(self, "canv", None)
            if canvas is not None:
                setattr(canvas, "_has_cover", True)

    return _CoverMarker()


class CoverPageSection(ReportSection):
    """First-page branded cover. PDF-only."""

    def enabled_for(self, audience: str) -> bool:
        return True

    def to_html(self, ctx: ReportContext) -> str:
        # HTML viewer already has an in-page header — no cover needed.
        return ""

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        from reportlab.platypus import (
            Paragraph, Spacer, Image, PageBreak, Table, TableStyle,
            KeepTogether,
        )
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader

        family = pdf_font_family()
        # Text-safe darker variant for titles, fill variant for the big
        # accent bar on the info table left edge.
        accent_text = getattr(config, "REPORT_ACCENT_TEXT_HEX",
                              getattr(config, "REPORT_ACCENT_HEX", "#5F8A00"))
        accent_fill = getattr(config, "REPORT_ACCENT_FILL_HEX",
                              getattr(config, "REPORT_ACCENT_HEX", "#AAF500"))
        clinic = getattr(config, "REPORT_CLINIC_NAME", "Biomech MoCap")
        subtitle = getattr(config, "REPORT_CLINIC_SUBTITLE", "") or ""

        # ── Styles ─────────────────────────────────────────────────────
        title_style = ParagraphStyle(
            "cov_title", fontName=family, fontSize=30,
            textColor=HexColor(accent_text), alignment=1, leading=36,
            spaceBefore=16, spaceAfter=4)
        subtitle_style = ParagraphStyle(
            "cov_subtitle", fontName=family, fontSize=13,
            textColor=HexColor("#666666"), alignment=1, leading=16,
            spaceAfter=40)
        clinic_style = ParagraphStyle(
            "cov_clinic", fontName=family, fontSize=18,
            textColor=HexColor("#212121"), alignment=1, leading=22,
            spaceAfter=2)
        clinic_sub_style = ParagraphStyle(
            "cov_clinic_sub", fontName=family, fontSize=10,
            textColor=HexColor("#888888"), alignment=1, leading=14,
            spaceAfter=30)
        label_style = ParagraphStyle(
            "cov_label", fontName=family, fontSize=10,
            textColor=HexColor("#888888"), alignment=0, leading=12)
        value_style = ParagraphStyle(
            "cov_value", fontName=family, fontSize=13,
            textColor=HexColor("#212121"), alignment=0, leading=18)
        footer_style = ParagraphStyle(
            "cov_footer", fontName=family, fontSize=9,
            textColor=HexColor("#999999"), alignment=1, leading=12)

        # ── Compose info block ─────────────────────────────────────────
        meta = ctx.session_meta or {}
        test_label = _rich_test_label(ctx.test_type, meta)

        # Date
        rec_ts = meta.get("record_start_wall_s")
        if rec_ts:
            try:
                date_str = _dt.datetime.fromtimestamp(
                    float(rec_ts)).strftime("%Y년 %m월 %d일  %H:%M")
            except Exception:
                date_str = str(rec_ts)
        else:
            date_str = ctx.session_dir.name

        # Subject
        subj = ctx.subject
        if subj is not None:
            subj_line1 = f"{getattr(subj, 'name', '—')}"
            subj_id = getattr(subj, "id", "")
            weight = getattr(subj, "weight_kg", 0.0) or 0.0
            height = getattr(subj, "height_cm", 0.0) or 0.0
            gender = {"M": "남", "F": "여"}.get(
                getattr(subj, "gender", ""), "")
            subj_line2 = (f"ID {subj_id}"
                          f"  ·  {weight:.1f} kg"
                          f"  ·  {height:.1f} cm"
                          + (f"  ·  {gender}" if gender else ""))
            trainer = getattr(subj, "trainer", "") or "—"
        else:
            subj_line1 = "(피험자 정보 없음)"
            subj_line2 = ""
            trainer = "—"

        # Test-specific extras
        duration = float(meta.get("duration_s") or 0.0)
        extras = []
        if ctx.test_type == "free_exercise":
            ex_name = meta.get("exercise_name") or "—"
            load_kg = meta.get("load_kg") or 0.0
            use_bw  = bool(meta.get("use_bodyweight_load"))
            subject_kg = float(meta.get("subject_kg") or 0.0)
            if use_bw:
                ext = max(0.0, float(load_kg) - subject_kg)
                load_str = (f"자중({subject_kg:.0f} kg) + {ext:.0f} kg"
                            if ext > 0 else f"자중({subject_kg:.0f} kg)")
            else:
                load_str = f"{float(load_kg):.0f} kg"
            extras.append(("운동 종목", ex_name))
            extras.append(("하중", load_str))

        # ── Layout ─────────────────────────────────────────────────────
        story = []
        # Side-effect marker so the branded-page canvas knows page 1 is cover
        story.append(_cover_marker_flowable())
        story.append(Spacer(1, 40))

        # Logo (if configured)
        logo_path = getattr(config, "REPORT_LOGO_PATH", None)
        if logo_path:
            try:
                img = ImageReader(str(logo_path))
                iw, ih = img.getSize()
                target_w = 200.0
                target_h = ih * target_w / max(iw, 1)
                story.append(Image(
                    str(logo_path), width=target_w, height=target_h,
                    hAlign="CENTER"))
                story.append(Spacer(1, 16))
            except Exception:
                pass

        # Clinic masthead
        story.append(Paragraph(clinic, clinic_style))
        if subtitle:
            story.append(Paragraph(subtitle, clinic_sub_style))
        else:
            story.append(Spacer(1, 24))

        # Big accent divider
        # Report title
        audience_tag = ("피험자 리포트"
                        if ctx.audience == "subject" else "트레이너 리포트")
        story.append(Paragraph("바이오메카닉스 측정 리포트", title_style))
        story.append(Paragraph(audience_tag, subtitle_style))

        # Info card (2 columns: label | value)
        info_rows: list[tuple[str, str]] = [
            ("피험자", subj_line1),
        ]
        if subj_line2:
            info_rows.append(("", subj_line2))
        info_rows += [
            ("테스트", test_label),
            ("측정일", date_str),
            ("측정 시간", f"{duration:.0f} s"),
        ]
        info_rows += extras
        info_rows.append(("담당 트레이너", trainer))

        cell_rows = []
        for k, v in info_rows:
            cell_rows.append([
                Paragraph(k, label_style) if k else Paragraph("", label_style),
                Paragraph(v, value_style),
            ])
        info_tbl = Table(
            cell_rows, colWidths=[36 * mm, 110 * mm])
        info_tbl.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LINEBELOW",    (0, 0), (-1, -2), 0.3, HexColor("#E0E0E0")),
            ("BACKGROUND",   (0, 0), (-1, -1), HexColor("#FAFAFA")),
            ("BOX",          (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("LINEBEFORE",   (0, 0), (0, -1), 3, HexColor(accent_fill)),
        ]))

        story.append(KeepTogether([info_tbl]))
        story.append(Spacer(1, 48))
        story.append(Paragraph(
            f"이 리포트는 {clinic}의 측정 데이터에 기반해 생성되었습니다.",
            footer_style))
        story.append(PageBreak())
        return story
