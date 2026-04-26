"""
ExcelExportWorker — background xlsx export for one session.

Long-running (up to tens of seconds for pose-rich sessions), so it runs
in a QThread to keep the GUI responsive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.analysis.excel_export import export_session_xlsx


class ExcelExportWorker(QThread):
    progress    = pyqtSignal(str)           # free-text progress messages
    finished_ok = pyqtSignal(str)           # final xlsx path
    failed      = pyqtSignal(str)           # error message

    def __init__(self, session_dir: str | Path,
                 out_path: Optional[str | Path] = None, parent=None):
        super().__init__(parent)
        self._session_dir = str(session_dir)
        self._out_path    = str(out_path) if out_path else None

    def run(self) -> None:
        try:
            out = export_session_xlsx(
                self._session_dir, self._out_path,
                progress_cb=lambda m: self.progress.emit(m),
            )
            self.finished_ok.emit(str(out))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
