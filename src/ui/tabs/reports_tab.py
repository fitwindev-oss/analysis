"""
Reports tab — session list + analysis result viewer, with an independent
replay page.

Layout (two pages, switched via QStackedWidget):

Page 0 — Browse:
    ┌ Filter bar ──────────────────────────────────────────────────┐
    ├──────────────────────────────┬───────────────────────────────┤
    │  세션 리스트 (테이블)          │  리포트 뷰어                    │
    │                              │  ▸ action buttons 우측 상단     │
    └──────────────────────────────┴───────────────────────────────┘

Page 1 — Replay (ReportViewer's "▶ 리플레이" button switches here):
    ┌ header: [← 뒤로] · 세션 라벨 · 피험자 · 날짜 ─────────────────┐
    ├─────────────────────────────────────────────────────────────┤
    │   ReplayPanel (video + encoder bars + CoP/Coord + graphs)    │
    └─────────────────────────────────────────────────────────────┘

Features:
  - Filter by subject / test type
  - Click a row → renders its result.json in the viewer
  - "▶ 리플레이" → switches to dedicated replay page
  - "← 뒤로" on replay page returns to browse with selection preserved
  - "분석 실행" / "2D 포즈" / "Excel" / "PDF" in the viewer's upper-right
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QTableWidget, QTableWidgetItem, QSplitter, QHeaderView, QMessageBox,
    QStackedWidget, QFrame,
)

from src.db import models as db_models
from src.analysis.dispatcher import read_result
from src.ui.widgets.report_viewer import ReportViewer, TEST_KO, test_label_for_session
from src.ui.widgets.replay_panel import ReplayPanel
from src.ui.workers.analysis_worker import AnalysisWorker
from src.ui.workers.pose_worker import PoseWorker


_STATUS_KO = {
    "recorded":        "녹화됨",
    "analyzing":       "분석 중…",
    "analyzed":        "분석 완료",
    "analyzed_full":   "분석 완료 (pose 포함)",
    "analysis_failed": "분석 실패",
    "cancelled":       "취소됨",
    "completed":       "완료",
}


class ReportsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._subjects: dict[str, str] = {}        # id → name
        self._sessions: list[db_models.Session] = []
        self._workers: dict[str, AnalysisWorker] = {}
        self._pose_workers: dict[str, PoseWorker] = {}
        self._build_ui()
        self.refresh()

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        from src.ui.widgets.sidebar_toggle import make_toggle_button, attach_toggle

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Filter bar — with ≡ collapse toggle on the left
        bar = QHBoxLayout()
        self._list_toggle = make_toggle_button("세션 목록 접기 / 펼치기")
        bar.addWidget(self._list_toggle)

        bar.addWidget(QLabel("피험자"))
        self._subject_combo = QComboBox()
        self._subject_combo.addItem("(전체)", None)
        self._subject_combo.currentIndexChanged.connect(self._apply_filters)
        bar.addWidget(self._subject_combo, stretch=1)

        bar.addWidget(QLabel("테스트"))
        self._test_combo = QComboBox()
        self._test_combo.addItem("(전체)", None)
        for key, ko in TEST_KO.items():
            self._test_combo.addItem(ko, key)
        self._test_combo.currentIndexChanged.connect(self._apply_filters)
        bar.addWidget(self._test_combo, stretch=1)

        self._btn_refresh = QPushButton("↻ 새로고침")
        self._btn_refresh.setToolTip("세션/피험자 목록을 DB에서 다시 불러옵니다 (단축키 F5)")
        self._btn_refresh.clicked.connect(self.refresh)
        bar.addWidget(self._btn_refresh)
        root.addLayout(bar)

        # Split: [session list | viewer]
        self._main_split = QSplitter(Qt.Orientation.Horizontal)
        self._main_split.setChildrenCollapsible(True)

        # Left: session table
        left_wrap = QWidget()
        left_wrap.setMinimumWidth(260)
        left_lay = QVBoxLayout(left_wrap)
        left_lay.setContentsMargins(0, 0, 0, 0)

        self._count_label = QLabel("세션 0")
        self._count_label.setStyleSheet("color:#888; padding:2px;")
        left_lay.addWidget(self._count_label)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["날짜", "피험자", "테스트", "상태", "duration"])
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        left_lay.addWidget(self._table, stretch=1)

        # Right: viewer
        self._viewer = ReportViewer()
        self._viewer.analyze_requested.connect(self._on_analyze_requested)
        self._viewer.pose_requested.connect(self._on_pose_requested)
        self._viewer.open_folder_requested.connect(self._on_open_folder)
        # Replay is now an independent page — clicking the button in the
        # viewer just asks us to swap to page 1.
        self._viewer.replay_requested.connect(self._on_replay_requested)

        self._main_split.addWidget(left_wrap)
        self._main_split.addWidget(self._viewer)
        self._main_split.setStretchFactor(0, 1)
        self._main_split.setStretchFactor(1, 2)
        self._main_split.setSizes([600, 900])

        # Wire the ≡ toggle → collapses/expands panel 0 (session list)
        attach_toggle(self._list_toggle, self._main_split,
                      panel_index=0, expanded_size=600)

        # ── Stacked pages: 0 = browse (filter + list + viewer),
        # 1 = replay (back button + ReplayPanel fullscreen). This avoids
        # cramming the replay video into the viewer's right half and fixes
        # the layout overflow that hid controls in fullscreen mode.
        self._stack = QStackedWidget()

        # Page 0 — browse
        browse = QWidget()
        bl = QVBoxLayout(browse)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self._main_split, stretch=1)
        self._stack.addWidget(browse)   # index 0

        # Page 1 — replay (built lazily to avoid loading pyqtgraph/video
        # widgets for users who never open replay; also keeps startup fast)
        self._replay_page: Optional[QWidget] = None
        self._replay_panel: Optional[ReplayPanel] = None
        self._replay_header_label: Optional[QLabel] = None

        root.addWidget(self._stack, stretch=1)

        # Keyboard shortcuts — scoped to this tab via
        # ``WidgetWithChildrenShortcut`` so they don't fire while Measure /
        # Subjects tabs are active.
        self._register_shortcuts()

    def _register_shortcuts(self) -> None:
        def sc(key, handler):
            QShortcut(QKeySequence(key), self,
                      activated=handler,
                      context=Qt.ShortcutContext.WidgetWithChildrenShortcut)

        # Browse-mode shortcuts (all guarded by stack page + selection)
        sc("F5",      self._shortcut_refresh)
        sc("Ctrl+R",  self._shortcut_analyze)
        sc("Ctrl+E",  self._shortcut_excel)
        sc("Ctrl+P",  self._shortcut_pdf)
        sc("Return",  self._shortcut_open_replay)
        sc("Enter",   self._shortcut_open_replay)   # numpad Enter
        # Replay-mode escape — back to browse. The ReplayPanel manages
        # Space / arrows / 1-5 / K internally (P1c).
        sc("Escape",  self._shortcut_escape)

    def _shortcut_refresh(self) -> None:
        # Only in browse mode (replay has no refresh concept)
        if self._stack.currentIndex() == 0:
            self.refresh()

    def _shortcut_analyze(self) -> None:
        if self._stack.currentIndex() != 0:
            return
        if self._viewer._btn_analyze.isEnabled():
            self._viewer._on_analyze_clicked()

    def _shortcut_excel(self) -> None:
        if self._stack.currentIndex() != 0:
            return
        if self._viewer._btn_excel.isEnabled():
            self._viewer._on_excel_clicked()

    def _shortcut_pdf(self) -> None:
        if self._stack.currentIndex() != 0:
            return
        if self._viewer._btn_pdf.isEnabled():
            self._viewer._on_pdf_clicked()

    def _shortcut_open_replay(self) -> None:
        if self._stack.currentIndex() != 0:
            return
        if self._viewer._btn_replay.isEnabled():
            self._viewer._on_replay_clicked()

    def _shortcut_escape(self) -> None:
        # Only meaningful from the replay page
        if self._stack.currentIndex() == 1:
            self._on_replay_back()

    def _ensure_replay_page(self) -> None:
        """Lazy-build the replay page on first use."""
        if self._replay_page is not None:
            return
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(8)

        # Header: [← 뒤로] + session label
        header = QHBoxLayout()
        header.setSpacing(10)
        btn_back = QPushButton("← 리포트로 돌아가기")
        btn_back.setProperty("kind", "ghost")
        btn_back.clicked.connect(self._on_replay_back)
        header.addWidget(btn_back)

        self._replay_header_label = QLabel("리플레이")
        rf = QFont(); rf.setPointSize(12); rf.setBold(True)
        self._replay_header_label.setFont(rf)
        self._replay_header_label.setStyleSheet("color:#ddd;")
        header.addWidget(self._replay_header_label, stretch=1)
        lay.addLayout(header)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(sep)

        self._replay_panel = ReplayPanel()
        # T4: restore previously saved replay layout if any.
        from src.ui.settings_store import restore_splitter
        if hasattr(self._replay_panel, "_outer_split"):
            restore_splitter(self._replay_panel._outer_split,
                             "replay/outer", default_sizes=(560, 900))
        lay.addWidget(self._replay_panel, stretch=1)

        self._replay_page = page
        self._stack.addWidget(page)   # index 1

    # ── replay navigation ──────────────────────────────────────────────────
    def _on_replay_requested(self, session_dir: str, test_type: str) -> None:
        """ReportViewer asked to enter replay mode for this session."""
        self._ensure_replay_page()
        assert self._replay_panel is not None
        assert self._replay_header_label is not None

        # Build the header label from the currently-selected session so
        # it matches the row the user just clicked.
        sess = self._selected_session()
        if sess is not None:
            subj = self._subjects.get(sess.subject_id, sess.subject_id or "—")
            date = (sess.session_date.split("T")[0]
                    if sess.session_date else "—")
            label = test_label_for_session(sess)
            self._replay_header_label.setText(
                f"{label}   |   {subj}   |   {date}"
            )
        else:
            self._replay_header_label.setText(f"{test_type} · {Path(session_dir).name}")

        self._replay_panel.load_session(session_dir)
        self._stack.setCurrentIndex(1)

    def _on_replay_back(self) -> None:
        """Leave replay mode — unload media to free resources + return to
        browse page. Selection in the table is preserved."""
        if self._replay_panel is not None:
            self._replay_panel.unload()
        self._stack.setCurrentIndex(0)

    # ── public API ─────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Reload subjects + sessions from DB and repaint."""
        try:
            subjects = db_models.list_subjects()
            self._subjects = {s.id: s.name for s in subjects}
            # Rebuild subject combo preserving selection
            prev = self._subject_combo.currentData()
            self._subject_combo.blockSignals(True)
            self._subject_combo.clear()
            self._subject_combo.addItem("(전체)", None)
            for s in subjects:
                self._subject_combo.addItem(f"{s.name} ({s.id})", s.id)
            if prev is not None:
                idx = self._subject_combo.findData(prev)
                if idx >= 0:
                    self._subject_combo.setCurrentIndex(idx)
            self._subject_combo.blockSignals(False)
        except Exception as e:
            QMessageBox.warning(
                self, "DB 오류",
                f"피험자 목록을 불러오지 못했습니다:\n{e}\n\n"
                f"잠시 후 '↻ 새로고침' (F5) 을 다시 시도해보세요. "
                f"문제가 계속되면 앱을 재시작해야 할 수 있습니다."
            )
            return

        self._apply_filters()

    # ── filtering ──────────────────────────────────────────────────────────
    def _apply_filters(self) -> None:
        subject_id = self._subject_combo.currentData()
        test_type  = self._test_combo.currentData()
        try:
            self._sessions = db_models.list_sessions(
                subject_id=subject_id, test_type=test_type, limit=500)
        except Exception as e:
            QMessageBox.warning(
                self, "DB 오류",
                f"세션 목록을 불러오지 못했습니다:\n{e}\n\n"
                f"필터 조건을 바꾸거나 '↻ 새로고침' (F5) 을 다시 시도해보세요."
            )
            self._sessions = []
        self._repaint_table()

    def _repaint_table(self) -> None:
        self._table.setRowCount(len(self._sessions))
        for row, s in enumerate(self._sessions):
            date = s.session_date.split("T")[0] if s.session_date else "—"
            subj_name = self._subjects.get(s.subject_id, s.subject_id or "—")
            test_ko = test_label_for_session(s)   # includes stance etc.
            status_ko = _STATUS_KO.get(s.status, s.status)
            dur = f"{s.duration_s:.0f}s" if s.duration_s else "—"
            items = [
                QTableWidgetItem(date),
                QTableWidgetItem(subj_name),
                QTableWidgetItem(test_ko),
                QTableWidgetItem(status_ko),
                QTableWidgetItem(dur),
            ]
            # Color the status cell
            if s.status in ("analyzed", "analyzed_full"):
                items[3].setForeground(Qt.GlobalColor.green)
            elif s.status == "analyzing":
                items[3].setForeground(Qt.GlobalColor.yellow)
            elif s.status in ("cancelled", "analysis_failed"):
                items[3].setForeground(Qt.GlobalColor.red)
            for col, it in enumerate(items):
                self._table.setItem(row, col, it)
        self._count_label.setText(f"세션 {len(self._sessions)}")
        # Clear viewer if selection empty
        if self._table.rowCount() == 0:
            self._viewer.show_empty("선택한 필터에 해당하는 세션이 없습니다.")

    # ── row selection ──────────────────────────────────────────────────────
    def _selected_session(self) -> Optional[db_models.Session]:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._sessions):
            return None
        return self._sessions[row]

    def _on_row_selected(self) -> None:
        s = self._selected_session()
        if s is None:
            self._viewer.show_empty()
            return
        if not s.session_dir:
            self._viewer.show_empty(
                "세션 폴더 경로가 DB에 없습니다. 원본 파일이 이동/삭제되었을 수 있습니다.")
            return
        subj_name = self._subjects.get(s.subject_id, s.subject_id)
        subj_label = f"{subj_name} ({s.subject_id})"
        result = read_result(s.session_dir)
        self._viewer.show_session(
            session_dir=s.session_dir,
            test_type=s.test_type,
            subject_label=subj_label,
            session_date=s.session_date,
            result=result,
            full_label=test_label_for_session(s),
        )

    # ── analysis actions ───────────────────────────────────────────────────
    def _on_analyze_requested(self, session_dir: str, test_type: str) -> None:
        if session_dir in self._workers and self._workers[session_dir].isRunning():
            return
        sess = self._selected_session()
        if sess is None:
            return
        worker = AnalysisWorker(session_dir, test_type=test_type, parent=self)
        self._workers[session_dir] = worker

        def _on_done(sd: str, res: dict) -> None:
            ok = res.get("error") is None
            new_status = "analyzed" if ok else "analysis_failed"
            try:
                db_models.update_session_status(sess.id, new_status)
            except Exception as e:
                print(f"[reports] DB status update error: {e}", flush=True)
            self._workers.pop(sd, None)
            # Refresh the table to update status column
            self._apply_filters()
            # If the same row is still selected, re-render viewer
            cur = self._selected_session()
            if cur is not None and cur.session_dir == sd:
                self._on_row_selected()

        worker.finished_ok.connect(_on_done)
        worker.start()
        # Optimistically update UI status
        try:
            db_models.update_session_status(sess.id, "analyzing")
            self._apply_filters()
        except Exception:
            pass

    def _on_pose_requested(self, session_dir: str, test_type: str) -> None:
        sess = self._selected_session()
        if sess is None:
            return
        if session_dir in self._pose_workers \
                and self._pose_workers[session_dir].isRunning():
            return
        self._viewer.set_pose_busy(True)
        try:
            db_models.update_session_status(sess.id, "analyzing")
            self._apply_filters()
        except Exception:
            pass

        pose = PoseWorker(session_dir, parent=self)
        self._pose_workers[session_dir] = pose

        def _on_all_done(sd: str, success: bool, err: str) -> None:
            self._pose_workers.pop(sd, None)
            if not success:
                QMessageBox.warning(
                    self, "2D 포즈 처리 실패",
                    f"{err or '원인 불명'}\n\n"
                    f"가능한 원인:\n"
                    f"  • MediaPipe 모델 파일을 다운로드하지 못했음 (네트워크 확인)\n"
                    f"  • 영상 파일이 손상되었거나 코덱 지원 안됨\n"
                    f"  • 'Force 분석'만 먼저 성공했는지 확인하고 다시 시도"
                )
                try:
                    db_models.update_session_status(sess.id, "analyzed")
                except Exception:
                    pass
                self._viewer.set_pose_busy(False)
                self._apply_filters()
                return
            # Re-run analysis so joint angles are incorporated into result.json
            self._viewer.set_analyze_busy(True)
            rerun = AnalysisWorker(sd, test_type=test_type, parent=self)
            self._workers[f"{sd}:rerun"] = rerun

            def _on_rerun_done(sd2: str, res: dict) -> None:
                ok = res.get("error") is None
                status = "analyzed_full" if ok else "analysis_failed"
                try:
                    db_models.update_session_status(sess.id, status)
                except Exception:
                    pass
                self._workers.pop(f"{sd}:rerun", None)
                self._viewer.set_analyze_busy(False)
                self._viewer.set_pose_busy(False)
                self._apply_filters()
                cur = self._selected_session()
                if cur is not None and cur.session_dir == sd:
                    self._on_row_selected()

            rerun.finished_ok.connect(_on_rerun_done)
            rerun.start()

        pose.all_done.connect(_on_all_done)
        pose.start()

    def _on_open_folder(self, session_dir: str) -> None:
        p = Path(session_dir)
        if not p.exists():
            QMessageBox.warning(self, "폴더 없음",
                                f"폴더를 찾을 수 없습니다:\n{session_dir}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(p))      # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            QMessageBox.warning(self, "열기 실패", str(e))
