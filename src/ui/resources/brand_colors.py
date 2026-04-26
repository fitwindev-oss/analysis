"""
FITWIN brand palette — single source of truth for colors used across the
Qt UI, matplotlib charts, pyqtgraph live plots, and the PDF renderer.

**Primary green (#AAF500)** was extracted from the official FITWIN logo
(Logo1.png / Logo2.png, Mar 2026) via pixel-frequency analysis. The
neutral scale mirrors Apple iOS dark mode so the UI reads as "modern
fitness tech" rather than "engineering tool".

**Usage rule:** FITWIN_PRIMARY is reserved for *action / emphasis* only —
active tab underline, primary CTA, selected row accent, verdict OK
banner. Using it on more than 2–3 elements per screen dilutes the
emotional pop. For status colors, use STATUS_* constants (Apple iOS
system palette).

Import into any module:

    from src.ui.resources.brand_colors import FITWIN_PRIMARY, BG_DEEP, ...

Matplotlib / pyqtgraph consumers should read these strings directly.
"""
from __future__ import annotations

# ── FITWIN brand ──────────────────────────────────────────────────────
FITWIN_PRIMARY          = "#AAF500"   # neon lime green — logo mark & type
FITWIN_PRIMARY_DIM      = "#96D700"   # hover / muted state
FITWIN_PRIMARY_BRIGHT   = "#C3FF2E"   # highlight / focus ring
FITWIN_PRIMARY_ON_DARK_TEXT = "#0A0A0A"  # black text sitting on green fill

# ── Neutral / background (iOS-inspired dark) ──────────────────────────
BG_DEEP         = "#0A0A0A"   # QMainWindow / app background
BG_ELEVATED     = "#1C1C1E"   # cards, panels, group boxes
BG_TERTIARY     = "#2C2C2E"   # nested sub-areas, table headers
BG_HOVER        = "#3A3A3C"   # button hover
SEPARATOR       = "#38383A"   # thin dividers

# ── Text ──────────────────────────────────────────────────────────────
TEXT_PRIMARY    = "#FFFFFF"
TEXT_SECONDARY  = "#A1A1A6"
TEXT_TERTIARY   = "#6E6E73"   # placeholders / disabled labels
TEXT_DISABLED   = "#48484A"

# ── Status (semantic) ─────────────────────────────────────────────────
STATUS_OK       = "#30D158"   # iOS green
STATUS_CAUTION  = "#FFD60A"   # iOS yellow
STATUS_DANGER   = "#FF453A"   # iOS red
STATUS_INFO     = "#0A84FF"   # iOS blue

# Soft tints for status backgrounds (≈ 10% opacity on deep black)
STATUS_OK_BG      = "#14371C"
STATUS_CAUTION_BG = "#3A2F02"
STATUS_DANGER_BG  = "#3A1512"
STATUS_INFO_BG    = "#06223F"

# ── Chart / plot series (curated, WCAG-friendly on dark bg) ───────────
# For force/encoder plots — left=blue, right=orange, total=FITWIN green
CHART_LEFT      = "#64B5F6"   # left channel (Board1 / enc1) — sky blue
CHART_RIGHT     = "#FFB74D"   # right channel (Board2 / enc2) — amber
CHART_TOTAL     = FITWIN_PRIMARY
CHART_NEUTRAL   = "#BA68C8"   # purple — for tertiary series (CoP trail, etc.)
CHART_GRID      = SEPARATOR

# Historical / reference bands
CHART_NORM_BAND = "#1E3A20"   # subtle green tint for "normal range" overlay


def apply_pyqtgraph_defaults() -> None:
    """Configure pyqtgraph's global background + foreground to match the
    brand. Call once at app startup (before any PlotWidget is created)."""
    try:
        import pyqtgraph as pg
        pg.setConfigOption("background", BG_DEEP)
        pg.setConfigOption("foreground", TEXT_SECONDARY)
        pg.setConfigOption("antialias", True)
    except ImportError:
        pass


def apply_matplotlib_defaults() -> None:
    """matplotlib is used for the REPORT charts that sit on a white
    page (HTML viewer card, PDF paper). Keep the default white-bg
    rendering — DO NOT invert to dark here, or else text ends up as
    grey-on-white with poor contrast.

    We only tune grid alpha (subtler gridlines) and the line-color
    cycler so the first series picks up the FITWIN dark-green accent.
    """
    try:
        import matplotlib as mpl
        mpl.rcParams.update({
            "grid.alpha":  0.3,
            # Keep default white facecolor, black text (matplotlib defaults)
            # so Korean labels render at maximum contrast on the report page.
        })
    except ImportError:
        pass
