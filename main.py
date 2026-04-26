"""
Biomech MoCap — application entry point.

Usage:
    python main.py

Checklist before running:
    1) pip install -r requirements.txt
    2) python scripts/generate_charuco_a4.py  (once, for calibration board)
    3) Launch this app → press "Start Preview" to verify cameras.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root on PYTHONPATH regardless of how main.py is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtWidgets import QApplication

# force stdout to UTF-8 so Korean / em-dash prints don't crash on cp949 consoles
# (the bug that bit the existing force_plate project).
try:
    sys.stdout.reconfigure(encoding="utf-8")       # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")       # type: ignore[attr-defined]
except Exception:
    pass

from src.ui.main_window import MainWindow
import config


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_TITLE)
    app.setApplicationVersion(config.APP_VERSION)

    # FITWIN brand theme (Phase R) — unified with app.py
    from src.ui.theme import apply_theme
    apply_theme(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
