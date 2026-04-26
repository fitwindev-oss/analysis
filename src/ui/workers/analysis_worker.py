"""
AnalysisWorker — background QThread that runs the analysis dispatcher
on a session directory and emits signals for UI consumption.

Signals:
    started(session_dir: str)
    finished_ok(session_dir: str, result: dict)    # includes error payload
    log_message(str)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.analysis.dispatcher import analyze_session


class AnalysisWorker(QThread):
    started_ok  = pyqtSignal(str)
    finished_ok = pyqtSignal(str, dict)
    log_message = pyqtSignal(str)

    def __init__(self, session_dir: str | Path,
                 test_type: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._session_dir = str(session_dir)
        self._test_type = test_type

    def run(self) -> None:
        self.started_ok.emit(self._session_dir)
        self.log_message.emit(f"[analyze] start: {self._session_dir}")
        try:
            result = analyze_session(self._session_dir,
                                     test_type=self._test_type,
                                     write_result=True)
            if result.get("error"):
                self.log_message.emit(
                    f"[analyze] error: {result['error']}")
            else:
                self.log_message.emit(
                    f"[analyze] done in {result.get('duration_analysis_s', 0):.2f} s"
                )
                # Populate the session_metrics cache so history queries pick
                # this session up immediately.
                try:
                    self._upsert_metrics_cache(result)
                except Exception as e:
                    self.log_message.emit(
                        f"[analyze] metrics cache write failed: {e}")
        except Exception as e:
            self.log_message.emit(f"[analyze] exception: {e}")
            result = {
                "test":  self._test_type,
                "error": f"{type(e).__name__}: {e}",
                "result": None,
            }
        self.finished_ok.emit(self._session_dir, result)

    # ── session_metrics cache population ───────────────────────────────────
    def _upsert_metrics_cache(self, result_payload: dict) -> None:
        """Match this session's dir to a DB session row, extract the key
        metrics, and upsert the session_metrics cache row."""
        import json
        from src.db import models
        from src.reports.key_metrics import (
            extract_key_metrics, variant_from_meta,
        )
        sdir = Path(self._session_dir)
        sname = sdir.name
        # Look up the matching DB session (match by session_dir path or name)
        candidates = models.list_sessions(limit=200)
        session = None
        for s in candidates:
            if s.session_dir and (s.session_dir == str(sdir)
                                  or Path(s.session_dir).name == sname):
                session = s; break
        if session is None:
            self.log_message.emit(
                "[analyze] (no DB session row yet — skip metrics cache)")
            return
        # session.json for variant lookup
        meta = {}
        sj = sdir / "session.json"
        if sj.exists():
            try:
                meta = json.loads(sj.read_text(encoding="utf-8"))
            except Exception:
                pass
        result = result_payload.get("result") or {}
        metrics = extract_key_metrics(session.test_type, result)
        if not metrics:
            return
        row = models.SessionMetricsRow(
            session_id=session.id, subject_id=session.subject_id,
            test_type=session.test_type,
            variant=variant_from_meta(session.test_type, meta),
            session_date=session.session_date,
            metrics=metrics,
        )
        models.upsert_session_metrics(row)
        self.log_message.emit(
            f"[analyze] metrics cached ({len(metrics)} keys)")
