"""
One-call theme bootstrap — loads the global QSS, registers the
Pretendard Variable font, and configures pyqtgraph + matplotlib so every
plot matches the Qt UI colors.

Call ``apply_theme(app)`` exactly once in the QApplication setup path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
_QSS_PATH      = _RESOURCE_ROOT / "app.qss"
_FONT_PATH     = _RESOURCE_ROOT / "fonts" / "PretendardVariable.ttf"

_loaded_families: list[str] = []


def _register_pretendard() -> Optional[str]:
    """Register the bundled Pretendard Variable TTF with Qt so the QSS
    ``font-family: "Pretendard Variable"`` actually resolves. Returns the
    resolved family name, or None if the file is missing / unreadable."""
    if not _FONT_PATH.exists():
        return None
    try:
        from PyQt6.QtGui import QFontDatabase
        font_id = QFontDatabase.addApplicationFont(str(_FONT_PATH))
        if font_id < 0:
            return None
        families = QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            return None
        fam = families[0]
        _loaded_families.append(fam)
        return fam
    except Exception:
        return None


def _load_qss() -> str:
    if not _QSS_PATH.exists():
        return ""
    try:
        return _QSS_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


def apply_theme(app) -> dict:
    """Apply the FITWIN brand theme to an already-constructed QApplication.

    Returns a dict describing what was applied for logging:
    ``{"qss_loaded": bool, "pretendard": str|None, "fallbacks": [...]}``.
    Errors are captured so the app still boots even if a resource is
    missing — UI just falls back to system defaults.
    """
    summary = {
        "qss_loaded": False,
        "pretendard": None,
        "fallbacks": [],
    }

    # ── Register Pretendard (bundled font, opt-in via QFont) ─────────
    # We register Pretendard with the font database so QSS can pick it
    # up BUT we don't call app.setFont(Pretendard). Reason: Qt 6's
    # variable-font rendering tends to look fuzzier at 13 pt than
    # static TTFs like Malgun Gothic. The default Qt font falls back
    # to the QSS font-family chain — Malgun Gothic first.
    # If a future screen (say, a hero title) wants Pretendard, it can
    # use `QFont("Pretendard Variable", 24, QFont.Weight.Bold)` directly.
    fam = _register_pretendard()
    if fam:
        summary["pretendard"] = fam
    else:
        summary["fallbacks"].append(
            "Pretendard TTF not found — system fonts only")

    # ── Apply QSS ────────────────────────────────────────────────────
    qss = _load_qss()
    if qss:
        app.setStyleSheet(qss)
        summary["qss_loaded"] = True
    else:
        summary["fallbacks"].append(
            "app.qss not found → Qt default styling")

    # ── pyqtgraph + matplotlib palette ───────────────────────────────
    try:
        from src.ui.resources.brand_colors import (
            apply_pyqtgraph_defaults, apply_matplotlib_defaults,
        )
        apply_pyqtgraph_defaults()
        apply_matplotlib_defaults()
    except Exception as e:
        summary["fallbacks"].append(f"brand_colors apply failed: {e}")

    return summary
