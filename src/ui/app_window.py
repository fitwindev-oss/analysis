"""
AppWindow — the new 3-tab main window for the biomech desktop app.

    [Subjects]   [Measure]   [Reports]

Phase 1 focus: Subjects tab fully working; Measure / Reports are stubs.
Inter-tab wiring: when a subject is marked "active" from the Subjects tab,
the Measure tab updates its active-subject panel.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QLabel, QMessageBox,
)

import config
from src.db import schema as db_schema
from src.ui.tabs.subjects_tab import SubjectsTab
from src.ui.tabs.measure_tab  import MeasureTab
from src.ui.tabs.reports_tab  import ReportsTab


class AppWindow(QMainWindow):
    def __init__(self, camera_probe_results=None):
        super().__init__()
        self.setWindowTitle(f"{config.APP_TITLE} — v{config.APP_VERSION}")
        # 24-inch Full HD (1920×1080) is our design baseline. Minimum
        # size is 75 % of that so the app still works on 22" 1440×810
        # legacy clinic monitors. Below this the layout starts to
        # break — we lock it via setMinimumSize.
        self.setMinimumSize(1440, 810)
        self.resize(1920, 1080)
        self._camera_probe_results = camera_probe_results or []

        # Ensure DB exists
        try:
            db_schema.initialise()
        except Exception as e:
            QMessageBox.critical(
                self, "DB 초기화 실패",
                f"데이터베이스를 열 수 없습니다:\n\n{e}\n\n"
                f"해결 방법:\n"
                f"  • data/ 폴더의 쓰기 권한을 확인하세요.\n"
                f"  • 앱이 중복 실행 중이면 하나만 남기고 모두 종료 후 재시작하세요.\n"
                f"  • data/biomech.db 파일이 손상된 경우 백업 후 삭제하면 새로 생성됩니다."
            )
            raise

        # ── Tabs ───────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._subjects_tab = SubjectsTab()
        self._measure_tab  = MeasureTab()
        self._reports_tab  = ReportsTab()

        self._tabs.addTab(self._subjects_tab, "Subjects")
        self._tabs.addTab(self._measure_tab,  "Measure")
        self._tabs.addTab(self._reports_tab,  "Reports")

        # ── FITWIN 로고: 탭바 우상단 corner widget으로 배치 ────────────
        # setCornerWidget은 QTabBar의 빈 여백 영역에 위젯을 꽂아넣어
        # 모든 탭(Subjects/Measure/Reports)에서 동일하게 보이게 한다.
        logo_path = (Path(__file__).resolve().parent
                     / "resources" / "brand" / "Logo1.png")
        if logo_path.exists():
            logo_label = QLabel()
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                target_h = 90       # 1.5× of the previous 60 px
                scaled = pix.scaledToHeight(
                    target_h, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled)
                logo_label.setToolTip(f"{config.REPORT_CLINIC_NAME}")
                logo_label.setContentsMargins(0, 4, 14, 4)
                self._tabs.setCornerWidget(
                    logo_label, Qt.Corner.TopRightCorner)

        self.setCentralWidget(self._tabs)

        # ── Status bar ─────────────────────────────────────────────────────
        self.setStatusBar(QStatusBar())
        self._status_cams = QLabel(self._cam_status_text())
        self._status_cams.setToolTip(self._cam_status_tooltip())
        cam_color = self._cam_status_color()
        self._status_cams.setStyleSheet(
            f"color:{cam_color}; padding:0 12px; font-weight:bold;")
        self.statusBar().addPermanentWidget(self._status_cams)

        self._status_subject = QLabel("측정 대상: (미선택)")
        self._status_subject.setStyleSheet("color:#888; padding:0 12px;")
        self.statusBar().addPermanentWidget(self._status_subject)
        self.statusBar().showMessage(
            "준비 완료. 먼저 Subjects 탭에서 피험자를 선택하세요."
        )

        # ── Wire signals ───────────────────────────────────────────────────
        self._subjects_tab.subject_selected.connect(self._on_subject_selected)
        # Any completed recording refreshes Reports so the new row appears.
        self._measure_tab.session_completed.connect(
            lambda _r: self._reports_tab.refresh())

        # ── T4: restore persisted window geometry + splitter sizes ────
        from src.ui.settings_store import (
            restore_window, restore_splitter,
        )
        restore_window(self)
        # Each tab exposes its main splitter; restore each.
        if hasattr(self._measure_tab, "_main_split"):
            restore_splitter(self._measure_tab._main_split,
                             "measure/main", default_sizes=(320, 700, 500))
        if hasattr(self._reports_tab, "_main_split"):
            restore_splitter(self._reports_tab._main_split,
                             "reports/main", default_sizes=(600, 900))

    # ── persistence on close ────────────────────────────────────────────
    def closeEvent(self, event):
        from src.ui.settings_store import (
            save_window, save_splitter,
        )
        save_window(self)
        if hasattr(self._measure_tab, "_main_split"):
            save_splitter(self._measure_tab._main_split, "measure/main")
        if hasattr(self._reports_tab, "_main_split"):
            save_splitter(self._reports_tab._main_split, "reports/main")
        # Replay panel (if it was lazy-built) has multiple splitters.
        if (hasattr(self._reports_tab, "_replay_panel")
                and self._reports_tab._replay_panel is not None):
            rp = self._reports_tab._replay_panel
            for attr, key in (
                ("_outer_split",  "replay/outer"),
                ("_right_split",  "replay/right"),
                ("_top_split",    "replay/top"),
            ):
                if hasattr(rp, attr):
                    save_splitter(getattr(rp, attr), key)
        super().closeEvent(event)

    # ── camera probe helpers (status bar) ──────────────────────────────────
    def _cam_status_text(self) -> str:
        total = len(self._camera_probe_results)
        ok = sum(1 for r in self._camera_probe_results if r.available)
        if total == 0:
            return f"카메라 {len(config.CAMERAS)} 활성"
        return f"카메라 {ok}/{total} 활성"

    def _cam_status_color(self) -> str:
        total = len(self._camera_probe_results) or len(config.CAMERAS)
        ok = sum(1 for r in self._camera_probe_results if r.available) \
            if self._camera_probe_results else len(config.CAMERAS)
        if total == 0 or ok == total:
            return "#4CAF50"
        if ok == 0:
            return "#EF5350"
        return "#FFA726"

    def _cam_status_tooltip(self) -> str:
        if not self._camera_probe_results:
            return "camera probe 미실행"
        lines = []
        for r in self._camera_probe_results:
            mark = "✓" if r.available else f"✗ ({r.reason})"
            lines.append(f"{r.id}  idx={r.index}  {r.label}  {mark}")
        return "\n".join(lines)

    def _on_subject_selected(self, subject):
        self._measure_tab.set_active_subject(subject)
        self._status_subject.setText(
            f"측정 대상: {subject.name} ({subject.id})"
        )
        self._status_subject.setStyleSheet(
            "color:#4CAF50; font-weight:bold; padding:0 12px;"
        )
        self._tabs.setCurrentWidget(self._measure_tab)
