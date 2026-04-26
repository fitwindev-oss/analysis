"""
PDF renderer (reportlab) — turns a list of ReportSection into a PDF file.

Each section returns a list of reportlab ``Flowable`` objects from its
``to_pdf_flowables(ctx)`` method. We concatenate them and let
``SimpleDocTemplate`` handle pagination.

Commercial-quality branding lives here:

- ``NumberedCanvas`` runs a two-pass trick so the footer can print
  "페이지 X / Y" with the true total page count.
- ``_draw_branded_page_frame`` paints a header (logo + test name) and a
  footer (clinic + contact + page indicator) on every page EXCEPT the
  cover. Cover detection relies on the special Flowable marker
  ``_CoverMarker`` inserted by CoverPageSection.
- Watermark + accent color pulled from ``config.REPORT_*`` constants.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, Optional

from reportlab.pdfgen import canvas as _rl_canvas

import config
from src.reports.base import ReportContext, ReportSection
from src.reports.fonts import setup_korean_fonts, pdf_font_family


# ────────────────────────────────────────────────────────────────────────────
# Public entry
# ────────────────────────────────────────────────────────────────────────────

def render_pdf(sections: Iterable[ReportSection],
               ctx: ReportContext,
               out_path: str | Path,
               page_size: str = "A4",
               orientation: str = "portrait") -> Path:
    """Build a PDF at ``out_path``. Returns the path."""
    setup_korean_fonts()
    from reportlab.platypus import SimpleDocTemplate
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.pagesizes import landscape

    size = {"A4": A4, "letter": letter}.get(page_size, A4)
    if orientation == "landscape":
        size = landscape(size)

    out_path = Path(out_path)
    doc = SimpleDocTemplate(
        str(out_path), pagesize=size,
        topMargin=56, bottomMargin=48,
        leftMargin=40, rightMargin=40,
        title=f"{config.REPORT_CLINIC_NAME} — {ctx.session_dir.name}",
        author=config.REPORT_CLINIC_NAME,
        subject=f"Biomech report ({ctx.test_type})",
    )
    flowables: list = []
    for s in sections:
        if s.enabled_for(ctx.audience):
            flowables.extend(s.to_pdf_flowables(ctx))

    # Precompute the short title used in the page header (skips on cover).
    header_title = _build_header_title(ctx)

    def _make_canvas(*args, **kwargs):
        c = NumberedCanvas(*args, **kwargs)
        c._ctx = ctx                         # type: ignore[attr-defined]
        c._header_title = header_title       # type: ignore[attr-defined]
        return c

    doc.build(flowables, canvasmaker=_make_canvas)
    return out_path


# ────────────────────────────────────────────────────────────────────────────
# Canvas with "X / Y" page numbers
# ────────────────────────────────────────────────────────────────────────────

class NumberedCanvas(_rl_canvas.Canvas):
    """Two-pass canvas so the footer can say "3 / 7" with the real total.

    reportlab calls ``showPage()`` at every page break — we stash the page
    state instead of flushing, then in ``save()`` replay them with the
    known total count so branding can print "page X of Y".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for i, state in enumerate(self._saved_page_states, 1):
            self.__dict__.update(state)
            self._page_index = i
            self._page_total = total
            _draw_branded_page_frame(self, i, total)
            super().showPage()
        super().save()


# ────────────────────────────────────────────────────────────────────────────
# Page frame — header + footer + watermark
# ────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hx: str) -> tuple[float, float, float]:
    h = hx.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def _is_cover_page(canvas_obj, page_index: int) -> bool:
    """Cover is page 1 only if CoverPageSection was the first section. We
    keep this simple — inspect the canvas' stashed _has_cover attribute,
    set by CoverPageSection via its flowables side-effect (see cover.py)."""
    return page_index == 1 and bool(getattr(canvas_obj, "_has_cover", False))


def _draw_branded_page_frame(canvas, page_index: int, page_total: int) -> None:
    """Paint header + footer + optional watermark on one page.

    Skipped on the cover page (CoverPageSection draws its own full-bleed
    layout). Uses ``config.REPORT_*`` constants so deployment tuning does
    not require code changes.
    """
    if _is_cover_page(canvas, page_index):
        # Cover owns the whole page — do not overlay a header/footer.
        return

    canvas.saveState()
    family = pdf_font_family()
    page_w, page_h = canvas._pagesize
    # Use the FILL variant for thick divider bars; fails WCAG for text
    # but is fine as a solid line. Text elsewhere uses TEXT variant.
    accent_rgb = _hex_to_rgb(
        getattr(config, "REPORT_ACCENT_FILL_HEX",
                getattr(config, "REPORT_ACCENT_HEX", "#AAF500")))

    # ── Header ─────────────────────────────────────────────────────────
    # Left: clinic name (logo would go here if/when REPORT_LOGO_PATH is set).
    # Center: short session title (ctx-derived).
    # Right: page X / Y
    header_baseline = page_h - 36
    canvas.setFillColorRGB(*accent_rgb)
    canvas.setFont(family, 9)
    logo_path = getattr(config, "REPORT_LOGO_PATH", None)
    left_x = 40
    if logo_path:
        try:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(str(logo_path))
            iw, ih = img.getSize()
            target_h = 22.0
            target_w = iw * target_h / max(ih, 1)
            canvas.drawImage(
                img, left_x, header_baseline - 4,
                width=target_w, height=target_h,
                preserveAspectRatio=True, mask="auto")
            left_x += target_w + 8
        except Exception:
            pass
    canvas.drawString(
        left_x, header_baseline,
        getattr(config, "REPORT_CLINIC_NAME", "Biomech MoCap"))

    title = getattr(canvas, "_header_title", "") or ""
    if title:
        canvas.setFillColorRGB(0.25, 0.25, 0.25)
        canvas.setFont(family, 9)
        canvas.drawCentredString(page_w / 2, header_baseline, title)

    # Right: page indicator
    canvas.setFillColorRGB(0.45, 0.45, 0.45)
    canvas.setFont(family, 9)
    canvas.drawRightString(
        page_w - 40, header_baseline,
        f"페이지 {page_index} / {page_total}")

    # Accent underline below the header text
    canvas.setStrokeColorRGB(*accent_rgb)
    canvas.setLineWidth(1.2)
    canvas.line(40, header_baseline - 6, page_w - 40, header_baseline - 6)

    # ── Footer ─────────────────────────────────────────────────────────
    # Left: clinic name + contact (if set). Right: generated timestamp.
    footer_y = 22
    canvas.setStrokeColorRGB(0.85, 0.85, 0.85)
    canvas.setLineWidth(0.5)
    canvas.line(40, footer_y + 10, page_w - 40, footer_y + 10)

    canvas.setFont(family, 7.5)
    canvas.setFillColorRGB(0.5, 0.5, 0.5)
    left_line = getattr(config, "REPORT_CLINIC_NAME", "Biomech MoCap")
    extra = getattr(config, "REPORT_FOOTER_LINE", "") or ""
    if extra:
        left_line = f"{left_line}   ·   {extra}"
    canvas.drawString(40, footer_y, left_line)

    gen = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    canvas.drawRightString(
        page_w - 40, footer_y,
        f"생성 {gen}   ·   v{getattr(config, 'APP_VERSION', '0.0.0')}")

    # ── Watermark (optional) ───────────────────────────────────────────
    watermark = getattr(config, "REPORT_WATERMARK", None)
    if watermark:
        canvas.saveState()
        canvas.setFont(family, 80)
        canvas.setFillColorRGB(0.9, 0.9, 0.9)
        canvas.translate(page_w / 2, page_h / 2)
        canvas.rotate(35)
        canvas.drawCentredString(0, 0, str(watermark))
        canvas.restoreState()

    canvas.restoreState()


# ────────────────────────────────────────────────────────────────────────────
# Short per-session title used in the page header
# ────────────────────────────────────────────────────────────────────────────

def _build_header_title(ctx: ReportContext) -> str:
    from src.reports.sections.common import _rich_test_label

    meta = ctx.session_meta or {}
    label = _rich_test_label(ctx.test_type, meta)
    subj_name = ""
    if ctx.subject is not None:
        subj_name = getattr(ctx.subject, "name", "") or ""
    rec_ts = meta.get("record_start_wall_s")
    date = ""
    if rec_ts:
        try:
            date = _dt.datetime.fromtimestamp(float(rec_ts)).strftime("%Y-%m-%d")
        except Exception:
            date = ""
    parts = [p for p in (subj_name, label, date) if p]
    return "   ·   ".join(parts)
