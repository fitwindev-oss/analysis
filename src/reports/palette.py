"""Central color palette for reports (HTML + PDF + matplotlib charts)."""
from __future__ import annotations

# Status / traffic light
STATUS_OK       = "#4CAF50"
STATUS_CAUTION  = "#FFB300"
STATUS_WARNING  = "#E53935"
STATUS_NEUTRAL  = "#1976D2"
STATUS_UNKNOWN  = "#757575"

# Force / plate
BOARD1_COLOR    = "#4FC3F7"   # light blue — left / Board1
BOARD2_COLOR    = "#FF8A65"   # orange     — right / Board2
TOTAL_COLOR     = "#A5D6A7"   # soft green — combined

# Pose
ANGLE_COLORS    = ["#4FC3F7", "#FF8A65", "#A5D6A7"]   # 3-slot palette
COORD_COLORS    = ["#FFEB3B", "#CE93D8"]              # 2-slot palette

# Cursors / overlays
CURSOR_COLOR    = "#FFEB3B"   # playback cursor
HOVER_LINE_COLOR= "#555555"

# History trend
HISTORY_LINE    = "#1976D2"
HISTORY_DOT     = "#0D47A1"
NORM_BAND_FILL  = "#E8F5E9"   # soft green — "normal range" band
NORM_BAND_LINE  = "#66BB6A"

# Background / surface (for HTML report embedded in dark app theme)
REPORT_BG       = "#161616"
REPORT_FG       = "#dddddd"
CARD_BG         = "#1a1a1a"
CARD_BORDER     = "#333333"
HEADER_FG       = "#90caf9"
SECTION_FG      = "#A5D6A7"


def status_to_color(status: str) -> str:
    """Map a classification string from norms.classify() to a hex color."""
    return {
        "ok":      STATUS_OK,
        "caution": STATUS_CAUTION,
        "warning": STATUS_WARNING,
    }.get(status, STATUS_NEUTRAL)
