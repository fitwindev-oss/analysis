"""
ReportViewer — trainer-facing HTML render of a session's analysis result.

Receives a result payload (from result.json) and renders metrics + optional
plot thumbnails. Falls back to a "no analysis yet" state with a "분석 실행"
button when no result is cached.

Signals:
    analyze_requested(session_dir: str, test_type: str)
    open_folder_requested(session_dir: str)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser,
    QFrame, QSplitter, QFileDialog, QMessageBox, QRadioButton, QButtonGroup,
    QComboBox,
)

from src.ui.workers.excel_export_worker import ExcelExportWorker
from src.ui.workers.csv_export_worker import CsvExportWorker
from src.ui.workers.pdf_export_worker import PdfExportWorker

# New report system
import json as _json
from src.reports.base import ReportContext
from src.reports.html_renderer import render_html
from src.reports.report_builder import (
    build_trainer_report, build_subject_report,
)
from src.db import models as _db_models


TEST_KO = {
    "balance_eo":     "눈 뜨고 밸런스",
    "balance_ec":     "눈 감고 밸런스",
    "cmj":            "CMJ 점프",
    "squat":          "스쿼트",
    "overhead_squat": "오버헤드 스쿼트",
    "encoder":        "엔코더 (바)",
    "reaction":       "반응 시간",
    "proprio":        "고유감각",
    "free_exercise":  "자유 운동",
}

_STANCE_KO = {"two": "양발", "left": "좌측발", "right": "우측발"}


def test_label_for_session(session) -> str:
    """Rich Korean label for a DB Session — includes stance/variant.

    Reads test-specific options from session.options_json to build labels
    like "눈 감고 밸런스 · 좌측발" or "반응 시간 · 10회 수동".
    """
    base = TEST_KO.get(session.test_type, session.test_type)
    opts = session.options() or {}
    extras: list[str] = []
    if session.test_type in ("balance_eo", "balance_ec"):
        stance = opts.get("stance", "two")
        extras.append(_STANCE_KO.get(stance, stance))
    elif session.test_type == "reaction":
        n = opts.get("reaction_responses")
        if isinstance(n, list):
            extras.append(f"{len(n)}종")
        trig = opts.get("reaction_trigger")
        if trig:
            extras.append("수동" if trig == "manual" else "자동")
    elif session.test_type == "encoder":
        p = opts.get("encoder_prompt")
        if p:
            extras.append(p[:20])
    elif session.test_type == "free_exercise":
        name = opts.get("exercise_name")
        if name:
            extras.append(str(name)[:20])
        extras.append(_free_load_label(opts))
    if extras:
        return f"{base} · {' · '.join(extras)}"
    return base


def _free_load_label(opts: dict) -> str:
    """Render free-exercise load as '자중' / 'N kg' / '자중+N kg'.

    ``load_kg`` in meta is already the EFFECTIVE load (external + bodyweight
    if the flag was set), resolved at record time in RecorderConfig.
    ``subject_kg`` lets us back out the external portion for display.
    """
    try:
        load = float(opts.get("load_kg") or 0.0)
    except (TypeError, ValueError):
        load = 0.0
    if opts.get("use_bodyweight_load"):
        subj = float(opts.get("subject_kg") or 0.0)
        external = max(0.0, load - subj)
        if external > 0:
            return f"자중+{external:.0f} kg"
        return "자중"
    return f"{load:.0f} kg" if load > 0 else "0 kg"


def test_label_for_opts(test_type: str, opts: dict) -> str:
    """Same as test_label_for_session but from a raw options dict."""
    base = TEST_KO.get(test_type, test_type)
    extras: list[str] = []
    if test_type in ("balance_eo", "balance_ec"):
        extras.append(_STANCE_KO.get(opts.get("stance", "two"), "양발"))
    elif test_type == "reaction":
        resp = opts.get("reaction_responses") or opts.get("responses")
        if isinstance(resp, list):
            extras.append(f"{len(resp)}종")
        trig = opts.get("reaction_trigger") or opts.get("trigger")
        if trig:
            extras.append("수동" if trig == "manual" else "자동")
    elif test_type == "encoder":
        p = opts.get("encoder_prompt")
        if p:
            extras.append(p[:20])
    elif test_type == "free_exercise":
        name = opts.get("exercise_name")
        if name:
            extras.append(str(name)[:20])
        extras.append(_free_load_label(opts))
    return f"{base} · {' · '.join(extras)}" if extras else base


class ReportViewer(QWidget):
    analyze_requested     = pyqtSignal(str, str)   # session_dir, test_type
    pose_requested        = pyqtSignal(str, str)   # session_dir, test_type
    open_folder_requested = pyqtSignal(str)        # session_dir
    replay_requested      = pyqtSignal(str, str)   # session_dir, test_type
                                                    # — handled by ReportsTab
                                                    # to switch into the
                                                    # independent replay page

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session_dir: Optional[str] = None
        self._test_type:   Optional[str] = None
        self._subject_label: str = ""
        self._session_date:  str = ""
        self._result_payload: Optional[dict] = None   # result.json contents
        self._audience: str = "trainer"               # "trainer" | "subject"
        self._history_limit: Optional[int] = 10       # None = all sessions
        self._build_ui()
        self.show_empty()

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # T7: minimum width keeps the secondary action row (Analyze /
        # Pose / Excel / PDF — 4 buttons) from clipping when the
        # session-list pane is dragged wide.
        self.setMinimumWidth(560)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Top header row: title (left) + primary actions (right) ────────
        # The action buttons sit in the upper-right so trainers don't have
        # to scroll past a long report body to find them.
        header = QHBoxLayout()
        header.setSpacing(8)
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        self._title = QLabel("리포트")
        tf = QFont(); tf.setPointSize(14); tf.setBold(True)
        self._title.setFont(tf)
        self._title.setStyleSheet("color:#fff;")
        title_col.addWidget(self._title)
        self._subtitle = QLabel("")
        self._subtitle.setStyleSheet("color:#bbb; font-size:12px;")
        title_col.addWidget(self._subtitle)
        header.addLayout(title_col, stretch=1)

        # Primary actions (upper-right): replay + folder. The other
        # actions (analyze / pose / excel / pdf) sit in a secondary row
        # below the audience controls, visually grouped.
        self._btn_replay = QPushButton("▶ 리플레이")
        self._btn_replay.setProperty("kind", "primary")
        self._btn_replay.setToolTip(
            "녹화 영상 + force/CoP/엔코더 그래프를 동기화하여 전용 페이지에서 "
            "재생합니다 (단축키 Enter).")
        self._btn_replay.clicked.connect(self._on_replay_clicked)

        self._btn_folder = QPushButton("📁 세션 폴더")
        self._btn_folder.setProperty("kind", "ghost")
        self._btn_folder.setToolTip(
            "이 세션의 원본 파일(forces.csv, mp4 등) 폴더를 파일 탐색기로 엽니다.")
        self._btn_folder.clicked.connect(self._on_folder_clicked)

        header.addWidget(self._btn_replay)
        header.addWidget(self._btn_folder)
        root.addLayout(header)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # ── Audience toggle (trainer / subject) + history combo ──────────
        aud_row = QHBoxLayout()
        aud_row.setSpacing(8)
        aud_row.addWidget(QLabel("리포트 유형:"))
        self._btn_trainer = QRadioButton("트레이너용")
        self._btn_subject = QRadioButton("피험자용")
        self._btn_trainer.setChecked(True)
        self._audience_group = QButtonGroup(self)
        self._audience_group.addButton(self._btn_trainer)
        self._audience_group.addButton(self._btn_subject)
        self._btn_trainer.toggled.connect(self._on_audience_changed)
        self._btn_subject.toggled.connect(self._on_audience_changed)
        aud_row.addWidget(self._btn_trainer)
        aud_row.addWidget(self._btn_subject)
        aud_row.addSpacing(20)
        aud_row.addWidget(QLabel("히스토리 범위:"))
        self._history_combo = QComboBox()
        for label, limit in [("최근 10 세션", 10),
                              ("최근 30 세션", 30),
                              ("최근 90일", "90days"),
                              ("전체", None)]:
            self._history_combo.addItem(label, limit)
        self._history_combo.setCurrentIndex(0)
        self._history_combo.currentIndexChanged.connect(
            self._on_history_changed)
        aud_row.addWidget(self._history_combo)
        aud_row.addStretch(1)
        root.addLayout(aud_row)

        # ── Secondary actions row: analyze / pose / excel / pdf ───────────
        # Grouped under audience/history because they relate to the report
        # content rather than to navigation.
        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)
        # Secondary actions use default "outlined" style from global QSS.
        self._btn_analyze = QPushButton("🔍 분석 실행")
        self._btn_analyze.setToolTip(
            "선택한 세션을 다시 분석해 result.json을 갱신합니다 (단축키 Ctrl+R).")
        self._btn_analyze.clicked.connect(self._on_analyze_clicked)

        self._btn_pose = QPushButton("🎯 2D 포즈 처리")
        self._btn_pose.setToolTip(
            "녹화된 mp4에서 RTMPose로 2D 관절을 추정한 뒤 재분석합니다. "
            "GPU 사용, 세션당 10~30초 소요.")
        self._btn_pose.clicked.connect(self._on_pose_clicked)

        self._btn_excel = QPushButton("📊 Excel 내보내기")
        self._btn_excel.setToolTip(
            "세션의 힘/CoP/관절 좌표·속도·각도·각속도 로우를 단일 xlsx로 내보냅니다.\n"
            "TimeSeries 시트(100 Hz 통일) + Pose_native + Summary + Per_rep 구성.\n"
            "(단축키 Ctrl+E)")
        self._btn_excel.clicked.connect(self._on_excel_clicked)

        self._btn_csv = QPushButton("📝 CSV 내보내기")
        self._btn_csv.setToolTip(
            "메모장으로 바로 열 수 있는 UTF-8 CSV로 내보냅니다.\n"
            "Excel 내보내기와 동일한 데이터를 평문으로 저장하며, 시간 옆에\n"
            "이벤트 컬럼(자극, 이탈)이 배치됩니다.\n"
            "  • <name>_timeseries.csv   — 100 Hz force + 보간 포즈\n"
            "  • <name>_pose_native.csv  — 네이티브 fps 포즈 (포즈 있을 때)")
        self._btn_csv.clicked.connect(self._on_csv_clicked)

        self._btn_pdf = QPushButton("📄 PDF 내보내기")
        self._btn_pdf.setToolTip(
            "현재 선택된 '리포트 유형' (트레이너용/피험자용) 으로 PDF를 생성합니다.\n"
            "(단축키 Ctrl+P)")
        self._btn_pdf.clicked.connect(self._on_pdf_clicked)

        actions_row.addWidget(self._btn_analyze)
        actions_row.addWidget(self._btn_pose)
        actions_row.addWidget(self._btn_excel)
        actions_row.addWidget(self._btn_csv)
        actions_row.addWidget(self._btn_pdf)
        actions_row.addStretch(1)
        root.addLayout(actions_row)

        # ── Report body ───────────────────────────────────────────────────
        self._body = QTextBrowser()
        self._body.setOpenExternalLinks(True)
        self._body.setStyleSheet(
            "QTextBrowser { background:#161616; color:#ddd; "
            "border:1px solid #2a2a2a; padding:8px; font-size:12px; }")
        root.addWidget(self._body, stretch=1)

        # Export worker references (kept alive while running)
        self._excel_worker: Optional[ExcelExportWorker] = None
        self._csv_worker:   Optional[CsvExportWorker]   = None
        self._pdf_worker:   Optional[PdfExportWorker]   = None

    # ── public API ─────────────────────────────────────────────────────────
    def show_empty(self, msg: str = "세션을 선택하면 리포트가 표시됩니다.") -> None:
        self._session_dir = None
        self._test_type = None
        self._result_payload = None
        self._title.setText("리포트")
        self._subtitle.setText("")
        self._body.setHtml(
            f"<div style='color:#777; padding:20px; text-align:center;'>"
            f"{msg}</div>")
        self._btn_analyze.setEnabled(False)
        self._btn_pose.setEnabled(False)
        self._btn_excel.setEnabled(False)
        self._btn_csv.setEnabled(False)
        self._btn_pdf.setEnabled(False)
        self._btn_replay.setEnabled(False)
        self._btn_folder.setEnabled(False)
        # Context-aware tooltip — explain why the buttons are disabled.
        no_sel = "좌측 세션 리스트에서 세션을 먼저 선택하세요."
        for b in (self._btn_analyze, self._btn_pose, self._btn_excel,
                  self._btn_csv, self._btn_pdf, self._btn_replay,
                  self._btn_folder):
            b.setToolTip(no_sel)

    def show_session(self, session_dir: str, test_type: str,
                     subject_label: str, session_date: str,
                     result: Optional[dict],
                     full_label: Optional[str] = None) -> None:
        """Render a selected session via the report system.

        `result` is the full result.json payload (or None / {"error": ...}).
        """
        self._session_dir = session_dir
        self._test_type = test_type
        self._subject_label = subject_label
        self._session_date  = session_date
        self._result_payload = result
        label = full_label or TEST_KO.get(test_type, test_type)
        self._title.setText(f"🎯 {label}")
        self._subtitle.setText(
            f"{subject_label}   |   {session_date}   |   {Path(session_dir).name}")
        self._btn_analyze.setEnabled(True)
        self._btn_pose.setEnabled(True)
        self._btn_excel.setEnabled(True)
        self._btn_csv.setEnabled(True)
        self._btn_replay.setEnabled(True)
        self._btn_folder.setEnabled(True)
        # Restore the informative tooltips (may have been overwritten by
        # show_empty's "select a session first" hint).
        self._btn_analyze.setToolTip(
            "선택한 세션을 다시 분석해 result.json을 갱신합니다 (단축키 Ctrl+R).")
        self._btn_replay.setToolTip(
            "녹화 영상 + force/CoP/엔코더 그래프를 동기화하여 전용 페이지에서 "
            "재생합니다 (단축키 Enter).")
        self._btn_folder.setToolTip(
            "이 세션의 원본 파일(forces.csv, mp4 등) 폴더를 파일 탐색기로 엽니다.")
        self._btn_excel.setToolTip(
            "세션의 힘/CoP/관절 좌표·속도·각도·각속도 로우를 단일 xlsx로 내보냅니다.\n"
            "TimeSeries 시트(100 Hz 통일) + Pose_native + Summary + Per_rep 구성.\n"
            "(단축키 Ctrl+E)")
        self._btn_csv.setToolTip(
            "메모장으로 바로 열 수 있는 UTF-8 CSV로 내보냅니다.\n"
            "Excel 내보내기와 동일한 데이터를 평문으로 저장하며, 시간 옆에\n"
            "이벤트 컬럼(자극, 이탈)이 배치됩니다.")
        self._btn_pdf.setToolTip(
            "현재 선택된 '리포트 유형' (트레이너용/피험자용) 으로 PDF를 생성합니다.\n"
            "(단축키 Ctrl+P)")

        # No analysis yet
        if result is None:
            self._body.setHtml(
                "<div style='color:#FFA726; padding:20px; text-align:center;'>"
                "<h3>아직 분석되지 않은 세션입니다</h3>"
                "<p>아래 '🔍 분석 실행' 버튼을 눌러 분석을 시작하세요.</p>"
                "</div>")
            self._btn_pdf.setEnabled(False)
            return

        # Analysis error — show the fixed error HTML (no section system)
        if result.get("error"):
            self._body.setHtml(self._render_error_html(result))
            self._btn_pdf.setEnabled(False)
            return

        # Normal path — render via report system
        self._btn_pdf.setEnabled(True)
        self._render_current()

    # ── report rendering (via section system) ──────────────────────────────
    def _build_context(self) -> Optional[ReportContext]:
        if not self._session_dir or not self._test_type or not self._result_payload:
            return None
        sdir = Path(self._session_dir)
        try:
            meta = _json.loads((sdir / "session.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        # Subject lookup (optional)
        subject = None
        sid = meta.get("subject_id")
        if sid:
            try:
                subject = _db_models.get_subject(sid)
            except Exception:
                subject = None
        # History lookup from session_metrics cache (same subject+test)
        history = []
        if sid:
            try:
                # Variant matching: for balance tests, match on stance so
                # two/left/right histories don't cross-contaminate.
                variant = None
                if self._test_type in ("balance_eo", "balance_ec"):
                    variant = meta.get("stance") or "two"
                # Resolve history filter (limit vs since_date)
                limit_arg = None
                since_arg = None
                lim = self._history_limit
                if isinstance(lim, int):
                    limit_arg = lim
                elif lim == "90days":
                    import datetime as _dtm
                    cutoff = (_dtm.datetime.now().astimezone()
                              - _dtm.timedelta(days=90))
                    since_arg = cutoff.isoformat(timespec="seconds")
                rows = _db_models.list_session_metrics(
                    subject_id=sid, test_type=self._test_type,
                    variant=variant, since_date=since_arg, limit=limit_arg)
                from src.reports.base import SessionMetrics as _SM
                history = [
                    _SM(session_id=r.session_id,
                        session_date=r.session_date,
                        metrics=r.metrics)
                    for r in rows
                ]
            except Exception:
                history = []
        return ReportContext(
            session_dir=sdir, session_meta=meta,
            result=self._result_payload.get("result") or {},
            test_type=self._test_type, subject=subject,
            history=history,
            audience=self._audience,
        )

    def _render_current(self) -> None:
        ctx = self._build_context()
        if ctx is None:
            return
        sections = (build_subject_report(ctx)
                    if self._audience == "subject"
                    else build_trainer_report(ctx))
        html = render_html(sections, ctx)
        self._body.setHtml(html)

    def _on_history_changed(self, _i: int) -> None:
        val = self._history_combo.currentData()
        self._history_limit = val      # may be int, None, or "90days"
        if self._result_payload is not None \
                and not self._result_payload.get("error"):
            self._render_current()

    def _on_audience_changed(self, _checked: bool) -> None:
        # Radio emits twice (one off, one on). React on the 'on' event only.
        new_audience = "subject" if self._btn_subject.isChecked() else "trainer"
        if new_audience == self._audience:
            return
        self._audience = new_audience
        if self._result_payload is not None \
                and not self._result_payload.get("error"):
            self._render_current()

    # ── rendering ──────────────────────────────────────────────────────────
    def _render_error_html(self, result: dict) -> str:
        err = result.get("error", "unknown error")
        tb  = result.get("traceback", "").replace("<", "&lt;").replace(">", "&gt;")
        return (
            f"<div style='color:#EF5350;'>"
            f"<h3>⚠ 분석 실패</h3>"
            f"<p><b>{err}</b></p>"
            f"<pre style='background:#1a1a1a; padding:8px; font-size:11px; "
            f"white-space:pre-wrap;'>{tb}</pre>"
            f"</div>"
        )

    # NOTE: _render_result_html + _render_balance/cmj/squat/encoder/reaction/
    # proprio/generic/pose_section used to live here. They were replaced by
    # the composable section system in src/reports/. _render_error_html (above)
    # and _on_* click handlers remain.
    # ── signals ────────────────────────────────────────────────────────────
    def _on_analyze_clicked(self) -> None:
        if self._session_dir and self._test_type:
            self.analyze_requested.emit(self._session_dir, self._test_type)

    def _on_pose_clicked(self) -> None:
        if self._session_dir and self._test_type:
            self.pose_requested.emit(self._session_dir, self._test_type)

    def _on_folder_clicked(self) -> None:
        if self._session_dir:
            self.open_folder_requested.emit(self._session_dir)

    # ── PDF export ─────────────────────────────────────────────────────────
    def _on_pdf_clicked(self) -> None:
        if not self._session_dir:
            return
        if self._pdf_worker is not None and self._pdf_worker.isRunning():
            return
        sd = Path(self._session_dir)
        audience_suffix = "_trainer" if self._audience == "trainer" else "_subject"
        default_path = str(sd / f"{sd.name}{audience_suffix}.pdf")
        chosen, _ = QFileDialog.getSaveFileName(
            self, "PDF 저장 위치", default_path,
            "PDF files (*.pdf)",
        )
        if not chosen:
            return
        self._btn_pdf.setEnabled(False)
        self._btn_pdf.setText("📄 PDF 생성 중…")
        worker = PdfExportWorker(self._session_dir, chosen,
                                  audience=self._audience, parent=self)
        self._pdf_worker = worker

        def _on_done(path: str) -> None:
            self._btn_pdf.setEnabled(True)
            self._btn_pdf.setText("📄 PDF 내보내기")
            QMessageBox.information(
                self, "PDF 내보내기 완료",
                f"저장됨:\n{path}")
            self._pdf_worker = None

        def _on_fail(err: str) -> None:
            self._btn_pdf.setEnabled(True)
            self._btn_pdf.setText("📄 PDF 내보내기")
            QMessageBox.warning(
                self, "PDF 내보내기 실패",
                f"{err}\n\n"
                f"가능한 원인:\n"
                f"  • 대상 파일이 다른 프로그램(예: PDF 뷰어)에서 열려 있음 → 닫고 다시 시도\n"
                f"  • 저장 위치에 쓰기 권한이 없음\n"
                f"  • result.json이 비어있거나 손상됨 → '🔍 분석 실행' 먼저 실행"
            )
            self._pdf_worker = None

        worker.finished_ok.connect(_on_done)
        worker.failed.connect(_on_fail)
        worker.start()

    # ── Excel export ───────────────────────────────────────────────────────
    def _on_excel_clicked(self) -> None:
        if not self._session_dir:
            return
        if self._excel_worker is not None and self._excel_worker.isRunning():
            return
        sd = Path(self._session_dir)
        default_path = str(sd / f"{sd.name}.xlsx")
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Excel 파일 저장 위치", default_path,
            "Excel files (*.xlsx)",
        )
        if not chosen:
            return
        self._btn_excel.setEnabled(False)
        self._btn_excel.setText("📊 Excel 생성 중…")
        worker = ExcelExportWorker(self._session_dir, chosen, parent=self)
        self._excel_worker = worker

        def _on_done(path: str) -> None:
            self._btn_excel.setEnabled(True)
            self._btn_excel.setText("📊 Excel 내보내기")
            QMessageBox.information(self, "Excel 내보내기 완료",
                                    f"저장됨:\n{path}")
            self._excel_worker = None

        def _on_fail(err: str) -> None:
            self._btn_excel.setEnabled(True)
            self._btn_excel.setText("📊 Excel 내보내기")
            QMessageBox.warning(
                self, "Excel 내보내기 실패",
                f"{err}\n\n"
                f"가능한 원인:\n"
                f"  • 대상 xlsx가 Excel에서 열려 있음 → Excel 닫고 다시 시도\n"
                f"  • 저장 위치에 쓰기 권한이 없음\n"
                f"  • forces.csv 또는 pose 파일이 손상됨"
            )
            self._excel_worker = None

        worker.finished_ok.connect(_on_done)
        worker.failed.connect(_on_fail)
        worker.start()

    # ── CSV export (Notepad-friendly, same data as Excel) ─────────────────
    def _on_csv_clicked(self) -> None:
        if not self._session_dir:
            return
        if self._csv_worker is not None and self._csv_worker.isRunning():
            return
        sd = Path(self._session_dir)
        default_path = str(sd / f"{sd.name}_timeseries.csv")
        chosen, _ = QFileDialog.getSaveFileName(
            self, "CSV 파일 저장 위치", default_path,
            "CSV files (*.csv)",
        )
        if not chosen:
            return
        self._btn_csv.setEnabled(False)
        self._btn_csv.setText("📝 CSV 생성 중…")
        worker = CsvExportWorker(self._session_dir, chosen, parent=self)
        self._csv_worker = worker

        def _on_done(ts_path: str, pose_path: str) -> None:
            self._btn_csv.setEnabled(True)
            self._btn_csv.setText("📝 CSV 내보내기")
            extra = f"\n포즈 (네이티브 fps):\n{pose_path}" if pose_path else ""
            QMessageBox.information(
                self, "CSV 내보내기 완료",
                f"저장됨:\n{ts_path}{extra}")
            self._csv_worker = None

        def _on_fail(err: str) -> None:
            self._btn_csv.setEnabled(True)
            self._btn_csv.setText("📝 CSV 내보내기")
            QMessageBox.warning(
                self, "CSV 내보내기 실패",
                f"{err}\n\n"
                f"가능한 원인:\n"
                f"  • 대상 csv가 다른 프로그램에서 열려 있음\n"
                f"  • 저장 위치에 쓰기 권한이 없음\n"
                f"  • forces.csv 또는 pose 파일이 손상됨"
            )
            self._csv_worker = None

        worker.finished_ok.connect(_on_done)
        worker.failed.connect(_on_fail)
        worker.start()

    # ── replay ──────────────────────────────────────────────────────────
    def _on_replay_clicked(self) -> None:
        """Emit a request to open the session in the independent replay
        page. The ReportsTab handles the page switch + session load —
        ReportViewer no longer embeds the ReplayPanel itself."""
        if self._session_dir is None or self._test_type is None:
            return
        self.replay_requested.emit(self._session_dir, self._test_type)

    # ── external toggles used while background workers run ────────────────
    def set_pose_busy(self, busy: bool) -> None:
        self._btn_pose.setEnabled(not busy and self._session_dir is not None)
        if busy:
            self._btn_pose.setText("🎯 2D 포즈 처리 중…")
        else:
            self._btn_pose.setText("🎯 2D 포즈 처리")

    def set_analyze_busy(self, busy: bool) -> None:
        self._btn_analyze.setEnabled(not busy and self._session_dir is not None)
        if busy:
            self._btn_analyze.setText("🔍 분석 중…")
        else:
            self._btn_analyze.setText("🔍 분석 실행 (재분석)")
