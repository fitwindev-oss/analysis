"""
CsvExportWorker — background plain-text CSV export for one session.

Runs in a QThread so the GUI stays responsive while pose-rich sessions
spend a few seconds interpolating + computing velocities.

Two files are produced (mirrors the dict returned by
``export_session_csv``):

  <session>_timeseries.csv   100 Hz force grid + interpolated pose
  <session>_pose_native.csv  pose at native fps (if available)

The ``finished_ok`` signal carries the timeseries path; the pose-native
path can be derived next to it (or just use the value the user sees in
the success dialog).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.analysis.csv_export import export_session_csv


class CsvExportWorker(QThread):
    progress    = pyqtSignal(str)           # free-text progress messages
    finished_ok = pyqtSignal(str, str)      # (timeseries_path, pose_path|"")
    failed      = pyqtSignal(str)           # error message

    def __init__(self, session_dir: str | Path,
                 out_path: Optional[str | Path] = None, parent=None):
        super().__init__(parent)
        self._session_dir = str(session_dir)
        self._out_path    = str(out_path) if out_path else None

    def run(self) -> None:
        try:
            out = export_session_csv(
                self._session_dir, self._out_path,
                progress_cb=lambda m: self.progress.emit(m),
            )
            ts_path  = str(out["timeseries"])
            pose_path = str(out["pose_native"]) if out["pose_native"] else ""
            self.finished_ok.emit(ts_path, pose_path)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
