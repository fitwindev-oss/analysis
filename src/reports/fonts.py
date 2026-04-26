"""
Korean font registration for matplotlib and reportlab.

Lookup order (first hit wins):
  1. TTF bundled under ``src/reports/resources/``
     (NanumGothic-Regular.ttf / NanumGothic-Bold.ttf)
  2. System-installed NanumGothic
  3. Windows default Korean face: Malgun Gothic

Both matplotlib (for charts) and reportlab (for PDF text) are set up so
HTML (rendered charts embedded as PNG) and PDF (native text) look
consistent.

Call ``setup_korean_fonts()`` once before generating any chart or PDF.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


# Bundled TTFs live in resources/fonts/ (preferred). The legacy path
# (resources/ directly) is still honoured for backwards compatibility
# with earlier deployments.
#
# Filename variants accepted for each weight — the Naver distribution
# ships both the hyphenated form (``NanumGothic-Regular.ttf``) and the
# terser legacy form (``NanumGothic.ttf`` / ``NanumGothicBold.ttf``), so
# we probe both and take the first that exists. This avoids a rename
# step during deployment.
_FONTS_DIR  = Path(__file__).parent / "resources" / "fonts"
_LEGACY_DIR = Path(__file__).parent / "resources"

_BUNDLED_REGULAR_CANDIDATES = [
    _FONTS_DIR  / "NanumGothic-Regular.ttf",
    _FONTS_DIR  / "NanumGothic.ttf",
    _LEGACY_DIR / "NanumGothic-Regular.ttf",
    _LEGACY_DIR / "NanumGothic.ttf",
]
_BUNDLED_BOLD_CANDIDATES = [
    _FONTS_DIR  / "NanumGothic-Bold.ttf",
    _FONTS_DIR  / "NanumGothicBold.ttf",
    _LEGACY_DIR / "NanumGothic-Bold.ttf",
    _LEGACY_DIR / "NanumGothicBold.ttf",
]

# Backwards-compat aliases — other modules may reference these names.
_BUNDLED_REGULAR = _BUNDLED_REGULAR_CANDIDATES[0]
_BUNDLED_BOLD    = _BUNDLED_BOLD_CANDIDATES[0]

# System locations to probe for NanumGothic / Malgun Gothic (Windows).
_SYSTEM_CANDIDATES_REGULAR = [
    Path(r"C:\Windows\Fonts\NanumGothic.ttf"),
    Path(r"C:\Windows\Fonts\malgun.ttf"),
]
_SYSTEM_CANDIDATES_BOLD = [
    Path(r"C:\Windows\Fonts\NanumGothicBold.ttf"),
    Path(r"C:\Windows\Fonts\malgunbd.ttf"),
]


def _first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _resolve_paths() -> tuple[Optional[Path], Optional[Path]]:
    regular = _first_existing(
        [*_BUNDLED_REGULAR_CANDIDATES, *_SYSTEM_CANDIDATES_REGULAR])
    bold    = _first_existing(
        [*_BUNDLED_BOLD_CANDIDATES,    *_SYSTEM_CANDIDATES_BOLD])
    return regular, bold


_STATE = {"set_up": False, "matplotlib_family": None, "pdf_family": None}


def setup_korean_fonts() -> dict:
    """Idempotent: registers fonts once, returns the resolved family names.

    Returns a dict like ``{"matplotlib_family": "NanumGothic",
    "pdf_family": "Korean"}``. If nothing is found, matplotlib falls back
    to its default and the PDF uses Helvetica — still works, but non-ASCII
    may render as boxes.
    """
    if _STATE["set_up"]:
        return _STATE.copy()

    regular, bold = _resolve_paths()

    # ── matplotlib ─────────────────────────────────────────────────────
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm
        if regular is not None:
            try:
                fm.fontManager.addfont(str(regular))
            except Exception:
                pass
        if bold is not None:
            try:
                fm.fontManager.addfont(str(bold))
            except Exception:
                pass
        # Pick whatever the font manager now knows about
        available = {f.name for f in fm.fontManager.ttflist}
        for cand in ("NanumGothic", "Nanum Gothic",
                     "Malgun Gothic", "맑은 고딕"):
            if cand in available:
                mpl.rcParams["font.family"] = cand
                _STATE["matplotlib_family"] = cand
                break
        mpl.rcParams["axes.unicode_minus"] = False
        # Print-friendly defaults — matplotlib's defaults (10pt labels)
        # are too small on A4 when rendered at 150 dpi. Bumping these
        # once here applies to every chart without per-call tuning.
        mpl.rcParams["axes.titlesize"]  = 13
        mpl.rcParams["axes.labelsize"]  = 11
        mpl.rcParams["xtick.labelsize"] = 10
        mpl.rcParams["ytick.labelsize"] = 10
        mpl.rcParams["legend.fontsize"] = 10
        mpl.rcParams["figure.titlesize"] = 14
        # Chart line palette — chosen for readability on WHITE report
        # pages. Each color has ≥4.5:1 contrast vs white (WCAG AA). The
        # FITWIN brand green #AAF500 is intentionally NOT a series color
        # because it fails contrast; we use the darker #5F8A00 variant
        # only when we explicitly want brand-coherent emphasis (e.g.
        # "recommended range" overlays).
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
            color=["#1565C0",   # primary — iOS blue (safe on white)
                   "#D32F2F",   # red
                   "#5F8A00",   # FITWIN green (dark variant)
                   "#7B1FA2",   # purple
                   "#F57C00",   # amber
                   "#00897B",   # teal
                   "#546E7A",   # blue-grey
                   "#5D4037"])  # brown
    except Exception:
        pass

    # ── reportlab ──────────────────────────────────────────────────────
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        if regular is not None:
            try:
                pdfmetrics.registerFont(TTFont("Korean", str(regular)))
                _STATE["pdf_family"] = "Korean"
            except Exception:
                pass
        if bold is not None:
            try:
                pdfmetrics.registerFont(TTFont("Korean-Bold", str(bold)))
            except Exception:
                pass
        # Register a family mapping so Helvetica-like styling just works
        try:
            from reportlab.pdfbase.pdfmetrics import registerFontFamily
            if _STATE["pdf_family"] == "Korean":
                registerFontFamily(
                    "Korean",
                    normal="Korean",
                    bold="Korean-Bold" if bold is not None else "Korean",
                    italic="Korean",
                    boldItalic="Korean-Bold" if bold is not None else "Korean",
                )
        except Exception:
            pass
    except Exception:
        pass

    _STATE["set_up"] = True
    return _STATE.copy()


def pdf_font_family() -> str:
    """Return the reportlab font family name to use in Paragraph styles."""
    setup_korean_fonts()
    return _STATE.get("pdf_family") or "Helvetica"
