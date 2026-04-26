"""
PdfExportWorker — background PDF export for one session.

Runs report_builder + render_pdf in a QThread because matplotlib chart
generation + reportlab assembly can take a couple of seconds for rich
trainer reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal


class PdfExportWorker(QThread):
    progress    = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed      = pyqtSignal(str)

    def __init__(self, session_dir: str | Path, out_path: str | Path,
                 audience: str = "trainer",
                 parent=None):
        super().__init__(parent)
        self._session_dir = str(session_dir)
        self._out_path    = str(out_path)
        self._audience    = audience

    def run(self) -> None:
        try:
            self.progress.emit("loading session…")
            sdir = Path(self._session_dir)
            meta = {}
            try:
                meta = json.loads(
                    (sdir / "session.json").read_text(encoding="utf-8"))
            except Exception:
                pass
            result_payload = {}
            try:
                result_payload = json.loads(
                    (sdir / "result.json").read_text(encoding="utf-8"))
            except Exception:
                pass
            test_type = meta.get("test") or ""

            # Subject + history lookup
            subject = None
            history = []
            sid = meta.get("subject_id")
            if sid:
                try:
                    from src.db import models as _db_models
                    subject = _db_models.get_subject(sid)
                except Exception:
                    subject = None
                try:
                    from src.db import models as _db_models2
                    from src.reports.base import SessionMetrics as _SM
                    variant = None
                    if test_type in ("balance_eo", "balance_ec"):
                        variant = meta.get("stance") or "two"
                    rows = _db_models2.list_session_metrics(
                        subject_id=sid, test_type=test_type,
                        variant=variant, limit=10)
                    history = [
                        _SM(session_id=r.session_id,
                            session_date=r.session_date,
                            metrics=r.metrics)
                        for r in rows
                    ]
                except Exception:
                    history = []

            self.progress.emit("building report sections…")
            from src.reports.base import ReportContext
            from src.reports.report_builder import (
                build_trainer_report, build_subject_report,
            )
            from src.reports.pdf_renderer import render_pdf

            ctx = ReportContext(
                session_dir=sdir, session_meta=meta,
                result=result_payload.get("result") or {},
                test_type=test_type, subject=subject,
                history=history,
                audience=self._audience,
            )
            sections = (build_subject_report(ctx)
                        if self._audience == "subject"
                        else build_trainer_report(ctx))

            self.progress.emit(f"rendering PDF ({len(sections)} sections)…")
            out = render_pdf(sections, ctx, self._out_path)
            self.finished_ok.emit(str(out))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
