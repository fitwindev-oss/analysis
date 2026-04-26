"""Palette constants shared by the replay panel's visualizations.

Slot order (selector position) determines color — picking the same joint
into a different slot will change its displayed color.
"""
from __future__ import annotations

# Hex colors. Keep reasonably distinct on a dark background.
ANGLE_COLORS: list[str] = ["#4FC3F7", "#FF8A65", "#A5D6A7"]  # 하늘·주황·연두
COORD_COLORS: list[str] = ["#FFEB3B", "#CE93D8"]             # 노랑·연보라
HOVER_LINE_COLOR = "#555"                                     # dashed crosshair
HOVER_TEXT_BG    = "rgba(20, 20, 20, 200)"                    # tooltip bg (css)
