"""
Data access layer for the biomech app.

Thin dataclass + CRUD wrapper over sqlite3. No ORM — the schema is small
enough that raw SQL is clearer and keeps PyInstaller deps minimal.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

from . import schema
from .schema import get_connection, now_iso


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Subject:
    id: str
    name: str
    weight_kg: float
    height_cm: float
    birthdate:      Optional[str] = None   # ISO date YYYY-MM-DD
    gender:         Optional[str] = None   # 'M' / 'F' / 'Other'
    dominant_leg:   Optional[str] = None   # 'L' / 'R' / 'Both'
    dominant_hand:  Optional[str] = None   # 'L' / 'R'
    trainer:        Optional[str] = None
    purpose:        Optional[str] = None
    notes:          Optional[str] = None
    created_at:     Optional[str] = None
    updated_at:     Optional[str] = None
    # Convenience: list of Injury (loaded on demand)
    injuries:       list = field(default_factory=list)

    @classmethod
    def new(cls, name: str, weight_kg: float, height_cm: float, **kw) -> "Subject":
        now = now_iso()
        return cls(
            id=str(uuid.uuid4())[:8],
            name=name, weight_kg=weight_kg, height_cm=height_cm,
            created_at=now, updated_at=now, **kw,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["injuries"] = [asdict(i) for i in self.injuries]
        return d


@dataclass
class Injury:
    subject_id: str
    description: str
    date: Optional[str] = None             # ISO date
    id: Optional[int] = None


@dataclass
class Session:
    id: str
    subject_id: str
    test_type: str
    session_date: str
    status: str                            # 'recorded' | 'analyzed' | 'completed'
    duration_s:   Optional[float] = None
    options_json: Optional[str]   = None
    session_dir:  Optional[str]   = None
    trainer:      Optional[str]   = None
    notes:        Optional[str]   = None

    @classmethod
    def new(cls, subject_id: str, test_type: str, **kw) -> "Session":
        # Default status to "recorded" but let caller override via kwargs.
        kw.setdefault("status", "recorded")
        return cls(
            id=str(uuid.uuid4())[:12],
            subject_id=subject_id, test_type=test_type,
            session_date=now_iso(), **kw,
        )

    def options(self) -> dict:
        return json.loads(self.options_json) if self.options_json else {}

    @staticmethod
    def encode_options(options: dict) -> str:
        return json.dumps(options, ensure_ascii=False)


@dataclass
class Calibration:
    calibrated_at: str
    scale_8: list                          # 8-element list of floats
    bw_subject_kg:    Optional[float] = None
    report_json_path: Optional[str]   = None
    notes:            Optional[str]   = None
    id:               Optional[int]   = None


@dataclass
class SessionMetricsRow:
    """Cached key metrics for one session (row in session_metrics table)."""
    session_id:   str
    subject_id:   str
    test_type:    str
    session_date: str
    metrics:      dict
    variant:      Optional[str] = None


@dataclass
class TestPreset:
    name: str
    protocol: list                         # list of {test, options} dicts
    description: Optional[str] = None
    created_at:  Optional[str] = None
    id:          Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# CRUD for subjects
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_subject(row: sqlite3.Row) -> Subject:
    return Subject(
        id=row["id"], name=row["name"],
        weight_kg=float(row["weight_kg"]),
        height_cm=float(row["height_cm"]),
        birthdate=row["birthdate"], gender=row["gender"],
        dominant_leg=row["dominant_leg"], dominant_hand=row["dominant_hand"],
        trainer=row["trainer"], purpose=row["purpose"],
        notes=row["notes"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def create_subject(subject: Subject, injuries: Optional[list[Injury]] = None) -> Subject:
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO subjects
               (id, name, birthdate, gender, weight_kg, height_cm,
                dominant_leg, dominant_hand, trainer, purpose, notes,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (subject.id, subject.name, subject.birthdate, subject.gender,
             subject.weight_kg, subject.height_cm,
             subject.dominant_leg, subject.dominant_hand,
             subject.trainer, subject.purpose, subject.notes,
             subject.created_at, subject.updated_at),
        )
        if injuries:
            for inj in injuries:
                conn.execute(
                    "INSERT INTO injuries (subject_id, description, date) VALUES (?,?,?)",
                    (subject.id, inj.description, inj.date),
                )
        conn.commit()
    finally:
        conn.close()
    return subject


def update_subject(subject: Subject, injuries: Optional[list[Injury]] = None) -> Subject:
    subject.updated_at = now_iso()
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE subjects SET
                 name=?, birthdate=?, gender=?, weight_kg=?, height_cm=?,
                 dominant_leg=?, dominant_hand=?, trainer=?, purpose=?,
                 notes=?, updated_at=?
               WHERE id=?""",
            (subject.name, subject.birthdate, subject.gender,
             subject.weight_kg, subject.height_cm,
             subject.dominant_leg, subject.dominant_hand,
             subject.trainer, subject.purpose, subject.notes,
             subject.updated_at, subject.id),
        )
        if injuries is not None:
            # replace all injuries atomically
            conn.execute("DELETE FROM injuries WHERE subject_id=?", (subject.id,))
            for inj in injuries:
                conn.execute(
                    "INSERT INTO injuries (subject_id, description, date) VALUES (?,?,?)",
                    (subject.id, inj.description, inj.date),
                )
        conn.commit()
    finally:
        conn.close()
    return subject


def delete_subject(subject_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM subjects WHERE id=?", (subject_id,))
        conn.commit()
    finally:
        conn.close()


def get_subject(subject_id: str, with_injuries: bool = True) -> Optional[Subject]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM subjects WHERE id=?", (subject_id,)
        ).fetchone()
        if row is None:
            return None
        subj = _row_to_subject(row)
        if with_injuries:
            subj.injuries = [
                Injury(subject_id=r["subject_id"], description=r["description"],
                       date=r["date"], id=r["id"])
                for r in conn.execute(
                    "SELECT * FROM injuries WHERE subject_id=?", (subject_id,)
                ).fetchall()
            ]
        return subj
    finally:
        conn.close()


def list_subjects(search: Optional[str] = None) -> list[Subject]:
    conn = get_connection()
    try:
        if search:
            rows = conn.execute(
                "SELECT * FROM subjects WHERE name LIKE ? OR id LIKE ? "
                "ORDER BY updated_at DESC",
                (f"%{search}%", f"%{search}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM subjects ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_subject(r) for r in rows]
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# CRUD for sessions
# ─────────────────────────────────────────────────────────────────────────────

def create_session(s: Session) -> Session:
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO sessions
               (id, subject_id, test_type, session_date, duration_s,
                options_json, session_dir, status, trainer, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (s.id, s.subject_id, s.test_type, s.session_date, s.duration_s,
             s.options_json, s.session_dir, s.status, s.trainer, s.notes),
        )
        conn.commit()
    finally:
        conn.close()
    return s


def update_session_status(session_id: str, status: str,
                          session_dir: Optional[str] = None) -> None:
    conn = get_connection()
    try:
        if session_dir is not None:
            conn.execute(
                "UPDATE sessions SET status=?, session_dir=? WHERE id=?",
                (status, session_dir, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET status=? WHERE id=?", (status, session_id),
            )
        conn.commit()
    finally:
        conn.close()


def list_sessions(subject_id: Optional[str] = None,
                  test_type: Optional[str] = None,
                  limit: int = 200) -> list[Session]:
    conn = get_connection()
    try:
        sql = "SELECT * FROM sessions WHERE 1=1"
        args: list = []
        if subject_id:
            sql += " AND subject_id=?"
            args.append(subject_id)
        if test_type:
            sql += " AND test_type=?"
            args.append(test_type)
        sql += " ORDER BY session_date DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(sql, args).fetchall()
        return [
            Session(
                id=r["id"], subject_id=r["subject_id"],
                test_type=r["test_type"], session_date=r["session_date"],
                duration_s=r["duration_s"], options_json=r["options_json"],
                session_dir=r["session_dir"], status=r["status"],
                trainer=r["trainer"], notes=r["notes"],
            )
            for r in rows
        ]
    finally:
        conn.close()


def delete_session(session_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Calibration history
# ─────────────────────────────────────────────────────────────────────────────

def record_calibration(scale_8: list[float], bw_subject_kg: Optional[float] = None,
                       report_json_path: Optional[str] = None,
                       notes: Optional[str] = None) -> Calibration:
    cal = Calibration(
        calibrated_at=now_iso(),
        scale_8=list(scale_8),
        bw_subject_kg=bw_subject_kg,
        report_json_path=report_json_path,
        notes=notes,
    )
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO calibrations
               (calibrated_at, scale_8_json, bw_subject_kg, report_json_path, notes)
               VALUES (?,?,?,?,?)""",
            (cal.calibrated_at, json.dumps(cal.scale_8),
             cal.bw_subject_kg, cal.report_json_path, cal.notes),
        )
        cal.id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    return cal


def latest_calibration() -> Optional[Calibration]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM calibrations ORDER BY calibrated_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return Calibration(
            id=row["id"],
            calibrated_at=row["calibrated_at"],
            scale_8=json.loads(row["scale_8_json"]),
            bw_subject_kg=row["bw_subject_kg"],
            report_json_path=row["report_json_path"],
            notes=row["notes"],
        )
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────

def save_preset(preset: TestPreset) -> TestPreset:
    preset.created_at = preset.created_at or now_iso()
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT OR REPLACE INTO test_presets
               (name, description, protocol_json, created_at)
               VALUES (?,?,?,?)""",
            (preset.name, preset.description,
             json.dumps(preset.protocol, ensure_ascii=False),
             preset.created_at),
        )
        preset.id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    return preset


def list_presets() -> list[TestPreset]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM test_presets ORDER BY name ASC"
        ).fetchall()
        return [
            TestPreset(
                id=r["id"], name=r["name"], description=r["description"],
                protocol=json.loads(r["protocol_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]
    finally:
        conn.close()


def delete_preset(preset_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM test_presets WHERE id=?", (preset_id,))
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Session metrics cache (history / trend queries)
# ─────────────────────────────────────────────────────────────────────────────

def upsert_session_metrics(row: SessionMetricsRow) -> None:
    """Insert or replace the cached metrics for one session."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO session_metrics
               (session_id, subject_id, test_type, variant,
                session_date, metrics_json)
               VALUES (?,?,?,?,?,?)""",
            (row.session_id, row.subject_id, row.test_type, row.variant,
             row.session_date, json.dumps(row.metrics, ensure_ascii=False,
                                          default=str)),
        )
        conn.commit()
    finally:
        conn.close()


def list_session_metrics(subject_id: str, test_type: str,
                         variant: Optional[str] = None,
                         since_date: Optional[str] = None,
                         limit: Optional[int] = None
                         ) -> list[SessionMetricsRow]:
    """Fetch cached metrics for a subject×test, newest first.

    ``variant`` filters by stance/vision/trigger if set.
    ``since_date`` (ISO) filters by ``session_date >= since_date``.
    ``limit`` caps the number of rows (None = all).
    """
    conn = get_connection()
    try:
        sql = ("SELECT * FROM session_metrics "
               "WHERE subject_id=? AND test_type=?")
        args: list = [subject_id, test_type]
        if variant is not None:
            sql += " AND variant=?"
            args.append(variant)
        if since_date:
            sql += " AND session_date >= ?"
            args.append(since_date)
        sql += " ORDER BY session_date DESC"
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            args.append(int(limit))
        rows = conn.execute(sql, args).fetchall()
        return [
            SessionMetricsRow(
                session_id=r["session_id"],
                subject_id=r["subject_id"],
                test_type=r["test_type"],
                variant=r["variant"],
                session_date=r["session_date"],
                metrics=json.loads(r["metrics_json"] or "{}"),
            )
            for r in rows
        ]
    finally:
        conn.close()


def delete_session_metrics(session_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM session_metrics WHERE session_id=?", (session_id,))
        conn.commit()
    finally:
        conn.close()
