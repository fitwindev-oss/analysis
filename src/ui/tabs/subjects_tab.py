"""
Subjects tab — browse, create, edit, and delete subject profiles.

Layout:
    ┌──────────────────────────────────────────────────────┐
    │ [Search...]                          [+ New Subject] │
    ├────────────────────┬─────────────────────────────────┤
    │ Subject list       │ Subject details (read-only)     │
    │ (table)            │ or editor                       │
    │                    │                                 │
    │                    │ Injury list                     │
    │                    │                                 │
    │                    │ [Edit]  [Delete]  [Set Active] │
    └────────────────────┴─────────────────────────────────┘

Emits `subject_selected(Subject)` when the user clicks "Set Active",
so the Measure tab can target that subject for recording.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QMessageBox, QSplitter, QFrame, QComboBox, QDoubleSpinBox,
    QDateEdit, QTextEdit, QDialog, QDialogButtonBox, QFormLayout,
    QListWidget, QListWidgetItem, QCheckBox,
)
from PyQt6.QtCore import QDate

from src.db import models as db
from src.db.models import Subject, Injury


# ─────────────────────────────────────────────────────────────────────────────
# Subject edit dialog
# ─────────────────────────────────────────────────────────────────────────────

class SubjectDialog(QDialog):
    """Create or edit a Subject + Injury list."""

    def __init__(self, parent=None, subject: Optional[Subject] = None):
        super().__init__(parent)
        self.setWindowTitle("피험자 정보" if subject else "신규 피험자 등록")
        self.resize(520, 600)
        self._subject = subject

        self._name     = QLineEdit()
        self._id       = QLineEdit()
        self._id.setPlaceholderText("비워두면 자동 생성")
        self._birthday = QDateEdit()
        self._birthday.setCalendarPopup(True)
        self._birthday.setDisplayFormat("yyyy-MM-dd")
        self._birthday.setDate(QDate.currentDate().addYears(-30))
        self._birthday_optional = QCheckBox("생년월일 없음")

        self._gender   = QComboBox()
        self._gender.addItems(["M", "F", "Other"])

        self._weight   = QDoubleSpinBox()
        self._weight.setRange(1, 400); self._weight.setSuffix(" kg")
        self._weight.setDecimals(1); self._weight.setSingleStep(0.5)

        self._height   = QDoubleSpinBox()
        self._height.setRange(30, 250); self._height.setSuffix(" cm")
        self._height.setDecimals(1); self._height.setSingleStep(0.5)

        self._dom_leg  = QComboBox()
        self._dom_leg.addItems(["", "L", "R", "Both"])
        self._dom_hand = QComboBox()
        self._dom_hand.addItems(["", "L", "R"])

        self._trainer  = QLineEdit()
        self._purpose  = QLineEdit()
        self._notes    = QTextEdit()
        self._notes.setFixedHeight(60)

        # Injury list
        self._injury_list = QListWidget()
        self._injury_list.setFixedHeight(100)
        self._injury_add_btn    = QPushButton("+ 부상 이력 추가")
        self._injury_remove_btn = QPushButton("선택 삭제")
        self._injury_add_btn.clicked.connect(self._add_injury)
        self._injury_remove_btn.clicked.connect(self._remove_injury)

        self._build_ui()
        if subject is not None:
            self._load_from_subject(subject)

    def _build_ui(self):
        form = QFormLayout()
        form.addRow("이름 *",     self._name)
        form.addRow("피험자 ID",  self._id)
        bd_layout = QHBoxLayout()
        bd_layout.addWidget(self._birthday)
        bd_layout.addWidget(self._birthday_optional)
        bd_wrap = QWidget(); bd_wrap.setLayout(bd_layout)
        form.addRow("생년월일",   bd_wrap)
        form.addRow("성별 *",     self._gender)
        form.addRow("체중 *",     self._weight)
        form.addRow("키 *",       self._height)
        form.addRow("주 다리",    self._dom_leg)
        form.addRow("주 손",      self._dom_hand)
        form.addRow("트레이너 *", self._trainer)
        form.addRow("측정 목적",  self._purpose)
        form.addRow("메모",       self._notes)

        inj_layout = QHBoxLayout()
        inj_layout.addWidget(self._injury_add_btn)
        inj_layout.addWidget(self._injury_remove_btn)
        inj_layout.addStretch()
        inj_wrap = QWidget(); inj_wrap.setLayout(inj_layout)
        form.addRow("부상 이력",  self._injury_list)
        form.addRow("",           inj_wrap)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._try_accept)
        btns.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(btns)

    def _add_injury(self):
        dlg = InjuryDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            desc, date_str = dlg.get_values()
            item = QListWidgetItem(f"{date_str or '날짜 미상'} — {desc}")
            item.setData(Qt.ItemDataRole.UserRole, (desc, date_str))
            self._injury_list.addItem(item)

    def _remove_injury(self):
        for item in self._injury_list.selectedItems():
            self._injury_list.takeItem(self._injury_list.row(item))

    def _load_from_subject(self, s: Subject):
        self._name.setText(s.name)
        self._id.setText(s.id); self._id.setReadOnly(True)
        if s.birthdate:
            self._birthday.setDate(QDate.fromString(s.birthdate, "yyyy-MM-dd"))
        else:
            self._birthday_optional.setChecked(True)
        if s.gender in ("M", "F", "Other"):
            self._gender.setCurrentText(s.gender)
        self._weight.setValue(float(s.weight_kg))
        self._height.setValue(float(s.height_cm))
        if s.dominant_leg:
            self._dom_leg.setCurrentText(s.dominant_leg)
        if s.dominant_hand:
            self._dom_hand.setCurrentText(s.dominant_hand)
        self._trainer.setText(s.trainer or "")
        self._purpose.setText(s.purpose or "")
        self._notes.setPlainText(s.notes or "")
        for inj in s.injuries:
            item = QListWidgetItem(f"{inj.date or '날짜 미상'} — {inj.description}")
            item.setData(Qt.ItemDataRole.UserRole,
                         (inj.description, inj.date))
            self._injury_list.addItem(item)

    def _try_accept(self):
        if not self._name.text().strip():
            QMessageBox.warning(self, "입력 오류", "이름은 필수입니다.")
            return
        if not self._trainer.text().strip():
            QMessageBox.warning(self, "입력 오류", "트레이너 이름은 필수입니다.")
            return
        if self._weight.value() < 10:
            QMessageBox.warning(self, "입력 오류", "체중을 입력하세요.")
            return
        if self._height.value() < 50:
            QMessageBox.warning(self, "입력 오류", "키를 입력하세요.")
            return
        self.accept()

    def get_subject(self) -> tuple[Subject, list[Injury]]:
        """Build a Subject + injuries from dialog inputs."""
        birthdate = None
        if not self._birthday_optional.isChecked():
            birthdate = self._birthday.date().toString("yyyy-MM-dd")

        subj_id = self._id.text().strip()
        if self._subject is None:
            # create
            subj = Subject.new(
                name=self._name.text().strip(),
                weight_kg=float(self._weight.value()),
                height_cm=float(self._height.value()),
                birthdate=birthdate,
                gender=self._gender.currentText(),
                dominant_leg=self._dom_leg.currentText() or None,
                dominant_hand=self._dom_hand.currentText() or None,
                trainer=self._trainer.text().strip(),
                purpose=self._purpose.text().strip() or None,
                notes=self._notes.toPlainText().strip() or None,
            )
            if subj_id:
                subj.id = subj_id       # user override
        else:
            subj = self._subject
            subj.name         = self._name.text().strip()
            subj.weight_kg    = float(self._weight.value())
            subj.height_cm    = float(self._height.value())
            subj.birthdate    = birthdate
            subj.gender       = self._gender.currentText()
            subj.dominant_leg = self._dom_leg.currentText() or None
            subj.dominant_hand = self._dom_hand.currentText() or None
            subj.trainer      = self._trainer.text().strip()
            subj.purpose      = self._purpose.text().strip() or None
            subj.notes        = self._notes.toPlainText().strip() or None

        # injuries
        injuries: list[Injury] = []
        for i in range(self._injury_list.count()):
            desc, date_str = self._injury_list.item(i).data(Qt.ItemDataRole.UserRole)
            injuries.append(Injury(subject_id=subj.id, description=desc, date=date_str))

        return subj, injuries


class InjuryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("부상 이력 추가")
        self.setFixedSize(400, 150)
        self._desc = QLineEdit()
        self._desc.setPlaceholderText("예: 우측 무릎 연골 수술")
        self._date = QDateEdit()
        self._date.setCalendarPopup(True)
        self._date.setDisplayFormat("yyyy-MM-dd")
        self._date.setDate(QDate.currentDate())
        self._no_date = QCheckBox("날짜 미상")

        form = QFormLayout()
        form.addRow("설명", self._desc)
        row = QHBoxLayout(); row.addWidget(self._date); row.addWidget(self._no_date)
        wrap = QWidget(); wrap.setLayout(row)
        form.addRow("날짜", wrap)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(btns)

    def get_values(self) -> tuple[str, Optional[str]]:
        desc = self._desc.text().strip()
        d = None if self._no_date.isChecked() else self._date.date().toString("yyyy-MM-dd")
        return desc, d


# ─────────────────────────────────────────────────────────────────────────────
# Subject detail viewer (right-side panel)
# ─────────────────────────────────────────────────────────────────────────────

class SubjectDetailView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self._title = QLabel("피험자를 선택하세요")
        t_font = QFont(); t_font.setPointSize(14); t_font.setBold(True)
        self._title.setFont(t_font)
        layout.addWidget(self._title)

        self._meta = QLabel("")
        self._meta.setStyleSheet("color:#aaa; font-size:11px;")
        layout.addWidget(self._meta)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color:#444;")
        layout.addWidget(divider)

        self._body = QLabel("")
        self._body.setTextFormat(Qt.TextFormat.RichText)
        self._body.setWordWrap(True)
        layout.addWidget(self._body, stretch=1)

    def show_subject(self, s: Optional[Subject]):
        if s is None:
            self._title.setText("피험자를 선택하세요")
            self._meta.setText("")
            self._body.setText("")
            return
        self._title.setText(f"{s.name}  ({s.id})")
        self._meta.setText(
            f"created: {s.created_at}   updated: {s.updated_at}"
        )
        injuries_html = ""
        if s.injuries:
            injuries_html = "<br>".join(
                f"• {i.date or '날짜 미상'} — {i.description}" for i in s.injuries
            )
        else:
            injuries_html = "<i>없음</i>"
        html = (
            f"<p><b>생년월일</b>: {s.birthdate or '—'}<br>"
            f"<b>성별</b>: {s.gender or '—'}<br>"
            f"<b>체중</b>: {s.weight_kg:.1f} kg &nbsp;&nbsp;"
            f"<b>키</b>: {s.height_cm:.1f} cm<br>"
            f"<b>주 다리</b>: {s.dominant_leg or '—'} &nbsp;&nbsp;"
            f"<b>주 손</b>: {s.dominant_hand or '—'}<br>"
            f"<b>트레이너</b>: {s.trainer or '—'}<br>"
            f"<b>측정 목적</b>: {s.purpose or '—'}</p>"
            f"<p><b>메모</b><br>{(s.notes or '—').replace(chr(10), '<br>')}</p>"
            f"<p><b>부상 이력</b><br>{injuries_html}</p>"
        )
        self._body.setText(html)


# ─────────────────────────────────────────────────────────────────────────────
# Subjects tab
# ─────────────────────────────────────────────────────────────────────────────

class SubjectsTab(QWidget):
    """Top-level tab. Emits `subject_selected(Subject)` when user chooses one."""

    subject_selected = pyqtSignal(object)     # Subject

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_subject: Optional[Subject] = None
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # Top control row
        top = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("이름 / ID 검색")
        self._search.textChanged.connect(lambda _: self.refresh())
        top.addWidget(self._search)

        top.addStretch()

        self._btn_new = QPushButton("+ 신규 피험자")
        self._btn_new.clicked.connect(self._on_new)
        top.addWidget(self._btn_new)

        self._btn_refresh = QPushButton("↻ 새로고침")
        self._btn_refresh.clicked.connect(lambda: self.refresh())
        top.addWidget(self._btn_refresh)

        root.addLayout(top)

        # Splitter: list | detail
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["ID", "이름", "성별", "체중", "키", "최근 수정"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.currentCellChanged.connect(self._on_row_changed)
        splitter.addWidget(self._table)

        right = QWidget()
        right_v = QVBoxLayout(right); right_v.setContentsMargins(0, 0, 0, 0)
        self._detail = SubjectDetailView()
        right_v.addWidget(self._detail, stretch=1)

        btn_row = QHBoxLayout()
        self._btn_edit    = QPushButton("편집")
        self._btn_delete  = QPushButton("삭제")
        self._btn_active  = QPushButton("측정 대상으로 선택")
        self._btn_edit.clicked.connect(self._on_edit)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_active.clicked.connect(self._on_set_active)
        self._btn_active.setStyleSheet(
            "background:#2E7D32; color:white; font-weight:bold; padding:8px;"
        )
        btn_row.addWidget(self._btn_edit)
        btn_row.addWidget(self._btn_delete)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_active)
        right_v.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([500, 600])
        root.addWidget(splitter, stretch=1)

    # ── Data operations ──────────────────────────────────────────────────────
    def refresh(self):
        term = self._search.text().strip()
        subjects = db.list_subjects(term or None)
        self._table.setRowCount(len(subjects))
        for row, s in enumerate(subjects):
            def _it(txt):
                it = QTableWidgetItem(str(txt)); it.setFlags(
                    it.flags() ^ Qt.ItemFlag.ItemIsEditable); return it
            self._table.setItem(row, 0, _it(s.id))
            self._table.setItem(row, 1, _it(s.name))
            self._table.setItem(row, 2, _it(s.gender or "—"))
            self._table.setItem(row, 3, _it(f"{s.weight_kg:.1f} kg"))
            self._table.setItem(row, 4, _it(f"{s.height_cm:.1f} cm"))
            self._table.setItem(row, 5, _it(s.updated_at or "—"))
            self._table.item(row, 0).setData(Qt.ItemDataRole.UserRole, s.id)
        if self._table.rowCount() > 0:
            self._table.selectRow(0)
        else:
            self._detail.show_subject(None)

    def _selected_subject(self) -> Optional[Subject]:
        row = self._table.currentRow()
        if row < 0:
            return None
        subj_id = self._table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        return db.get_subject(subj_id)

    def _on_row_changed(self, row, col, prev_row, prev_col):
        s = self._selected_subject()
        self._detail.show_subject(s)

    def _on_new(self):
        dlg = SubjectDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            subj, injuries = dlg.get_subject()
            try:
                db.create_subject(subj, injuries)
            except Exception as e:
                QMessageBox.critical(self, "오류",
                                     f"저장 실패: {e}")
                return
            self.refresh()
            self._select_by_id(subj.id)

    def _on_edit(self):
        s = self._selected_subject()
        if s is None:
            return
        dlg = SubjectDialog(self, subject=s)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            subj, injuries = dlg.get_subject()
            db.update_subject(subj, injuries)
            self.refresh()
            self._select_by_id(subj.id)

    def _on_delete(self):
        s = self._selected_subject()
        if s is None:
            return
        reply = QMessageBox.question(
            self, "삭제 확인",
            f"{s.name} ({s.id}) 및 모든 측정 기록을 삭제할까요?\n"
            f"이 동작은 되돌릴 수 없습니다."
        )
        if reply == QMessageBox.StandardButton.Yes:
            db.delete_subject(s.id)
            self.refresh()

    def _on_set_active(self):
        s = self._selected_subject()
        if s is None:
            QMessageBox.information(self, "알림", "피험자를 먼저 선택하세요.")
            return
        self._active_subject = s
        self.subject_selected.emit(s)
        QMessageBox.information(
            self, "측정 대상 설정됨",
            f"{s.name} ({s.id}) 을(를) 측정 대상으로 설정했습니다.\n"
            f"'Measure' 탭으로 이동하세요."
        )

    def _select_by_id(self, subject_id: str):
        for row in range(self._table.rowCount()):
            if self._table.item(row, 0).data(Qt.ItemDataRole.UserRole) == subject_id:
                self._table.selectRow(row)
                return

    @property
    def active_subject(self) -> Optional[Subject]:
        return self._active_subject
