"""
Sidebar collapse/expand helpers.

``make_toggle_button()`` returns a compact ≡ push-button styled to sit in
a toolbar. ``attach_toggle(btn, splitter, panel_index, expanded_size)``
wires it up so clicking flips the splitter between the saved expanded
width and 0 (fully collapsed).

Works with any QSplitter; caller controls which child index is the
collapsible panel.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QPushButton, QSplitter


def make_toggle_button(tooltip: str = "사이드바 접기 / 펼치기",
                        size: int = 28) -> QPushButton:
    btn = QPushButton("≡")
    btn.setFixedSize(QSize(size, size))
    btn.setCheckable(True)
    btn.setChecked(True)     # starts as "expanded" (pressed)
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(
        "QPushButton { background:#2a2a2a; color:#bbb; border:1px solid #333;"
        "              border-radius:4px; font-size:16px; font-weight:bold; }"
        "QPushButton:hover   { background:#3a3a3a; color:#fff; }"
        "QPushButton:checked { background:#2E7D32; color:#fff; border-color:#43A047; }"
    )
    return btn


def attach_toggle(button: QPushButton, splitter: QSplitter,
                   panel_index: int, expanded_size: int = 280,
                   on_toggle: Optional[Callable[[bool], None]] = None) -> None:
    """Wire ``button`` so it toggles ``splitter``'s ``panel_index`` child
    between 0 (collapsed) and its last non-zero size (or ``expanded_size``
    fallback on first expand).

    Optional ``on_toggle(expanded: bool)`` callback runs after each flip.
    """
    state = {"last_size": expanded_size}

    def _apply(expanded: bool) -> None:
        sizes = splitter.sizes()
        if len(sizes) < 2:
            return
        other_total = sum(sizes) - sizes[panel_index]
        if not expanded:
            # save current size if non-zero, then collapse
            if sizes[panel_index] > 10:
                state["last_size"] = sizes[panel_index]
            sizes[panel_index] = 0
        else:
            sizes[panel_index] = state["last_size"]
        # re-proportion the remaining widgets
        total = sum(sizes)
        if total <= 0:
            return
        splitter.setSizes(sizes)
        button.setText("≡" if expanded else "»")
        if on_toggle:
            try:
                on_toggle(expanded)
            except Exception:
                pass

    button.toggled.connect(_apply)


def programmatic_collapse(button: QPushButton, collapsed: bool = True) -> None:
    """Convenience: programmatically set the collapsed state without
    triggering extra signal noise. ``collapsed=True`` hides; False expands."""
    expanded = not collapsed
    if button.isChecked() != expanded:
        button.setChecked(expanded)
