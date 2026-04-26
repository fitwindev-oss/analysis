"""
SQLite schema + initialisation for the biomech app.

Tables:
  subjects         — subject profile
  injuries         — 1:N per subject
  sessions         — recorded test sessions
  session_metrics  — cached key metrics per session for fast history queries
  calibrations     — DAQ scale calibration history
  test_presets     — saved test option bundles

Data lives in data/biomech.db (see config.DATA_DIR).

All timestamps are stored as ISO-8601 strings in KST (Asia/Seoul).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import config


KST = timezone(timedelta(hours=9))


def now_iso() -> str:
    """Current time as ISO-8601 string in KST."""
    return datetime.now(KST).isoformat(timespec="seconds")


DB_PATH = config.DATA_DIR / "biomech.db"


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS subjects (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    birthdate       TEXT,
    gender          TEXT,
    weight_kg       REAL NOT NULL,
    height_cm       REAL NOT NULL,
    dominant_leg    TEXT,
    dominant_hand   TEXT,
    trainer         TEXT,
    purpose         TEXT,
    notes           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS injuries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id      TEXT NOT NULL,
    description     TEXT NOT NULL,
    date            TEXT,
    FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    subject_id      TEXT NOT NULL,
    test_type       TEXT NOT NULL,
    session_date    TEXT NOT NULL,
    duration_s      REAL,
    options_json    TEXT,
    session_dir     TEXT,
    status          TEXT NOT NULL,
    trainer         TEXT,
    notes           TEXT,
    FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_subject ON sessions(subject_id);
CREATE INDEX IF NOT EXISTS idx_sessions_test    ON sessions(test_type);
CREATE INDEX IF NOT EXISTS idx_sessions_date    ON sessions(session_date);

-- Cached key metrics per session. Populated by AnalysisWorker on every
-- analysis completion so history queries (subject × test × variant) can
-- be answered with a single indexed scan instead of re-reading result.json
-- files from disk.
CREATE TABLE IF NOT EXISTS session_metrics (
    session_id      TEXT PRIMARY KEY,
    subject_id      TEXT NOT NULL,
    test_type       TEXT NOT NULL,
    variant         TEXT,                 -- stance / vision / trigger, may be null
    session_date    TEXT NOT NULL,
    metrics_json    TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sm_subject_test
    ON session_metrics(subject_id, test_type, session_date DESC);
CREATE INDEX IF NOT EXISTS idx_sm_variant
    ON session_metrics(subject_id, test_type, variant, session_date DESC);

CREATE TABLE IF NOT EXISTS calibrations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    calibrated_at   TEXT NOT NULL,
    scale_8_json    TEXT NOT NULL,
    bw_subject_kg   REAL,
    report_json_path TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS test_presets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT UNIQUE NOT NULL,
    description     TEXT,
    protocol_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys + row factory."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialise(db_path: Optional[Path] = None) -> Path:
    """Create the DB file and tables if they do not yet exist. Returns path."""
    path = db_path or DB_PATH
    conn = get_connection(path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return path
