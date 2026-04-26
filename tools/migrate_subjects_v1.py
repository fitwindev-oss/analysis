"""
Phase V1 migration: ensure every subject has gender and birthdate.

These two fields are now REQUIRED to compute 1RM grades — the
strength-norms tables in src/analysis/strength_norms.py do a
``sex × age × bw`` lookup and gracefully refuse to grade subjects
with either missing.

Run this once after pulling V1. Subjects missing values are listed,
then prompted one at a time. Ctrl+C aborts without saving.

Usage:
    python tools/migrate_subjects_v1.py
    python tools/migrate_subjects_v1.py --dry-run     # only list, no prompt
    python tools/migrate_subjects_v1.py --db <path>   # alternate DB

Exit codes:
    0 = nothing to migrate, or migration completed
    1 = aborted by user (Ctrl+C)
    2 = bad arguments / DB not found
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Make project root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_DB = Path("data/biomech.db")


def find_incomplete(conn: sqlite3.Connection) -> list[tuple]:
    """Return rows from `subjects` missing gender or birthdate.

    Treats both NULL and empty-string as "missing" so we don't trust
    the DB-level NOT NULL alone (some import paths have written '').
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, gender, birthdate FROM subjects "
        "WHERE gender IS NULL OR gender = '' "
        "OR birthdate IS NULL OR birthdate = '' "
        "ORDER BY name"
    )
    return cur.fetchall()


def prompt_gender(name: str, current: str | None) -> str:
    while True:
        v = input(f"  gender for {name} (M/F) "
                  f"[current: {current!r}]: ").strip().upper()
        if v in ("M", "F"):
            return v
        print("    invalid — please enter M or F")


def prompt_birthdate(name: str, current: str | None) -> str:
    while True:
        v = input(f"  birthdate for {name} (YYYY-MM-DD) "
                  f"[current: {current!r}]: ").strip()
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            print("    invalid — please enter as YYYY-MM-DD "
                  "(e.g. 1990-05-21)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="V1 migration: fill missing gender/birthdate on subjects.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List incomplete subjects without prompting")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB,
                    help=f"SQLite DB path (default: {DEFAULT_DB})")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(args.db))
    incomplete = find_incomplete(conn)

    if not incomplete:
        print("All subjects have gender + birthdate. Nothing to migrate.")
        conn.close()
        return 0

    print(f"Found {len(incomplete)} subject(s) with missing fields:")
    for sid, name, gender, birth in incomplete:
        print(f"  - {name} (id={sid}): "
              f"gender={gender!r}, birthdate={birth!r}")

    if args.dry_run:
        conn.close()
        return 0

    print()
    print("Filling missing fields. Press Ctrl+C to abort without saving.")
    print()

    updates: list[tuple[str, str, str]] = []
    try:
        for sid, name, gender, birth in incomplete:
            print(f"--- {name} (id={sid}) ---")
            new_gender = (gender if gender
                          else prompt_gender(name, gender))
            new_birth = (birth if birth
                         else prompt_birthdate(name, birth))
            updates.append((new_gender, new_birth, sid))
    except KeyboardInterrupt:
        print("\naborted by user — no changes saved")
        conn.close()
        return 1

    cur = conn.cursor()
    cur.executemany(
        "UPDATE subjects SET gender = ?, birthdate = ?, "
        "updated_at = datetime('now') WHERE id = ?",
        updates,
    )
    conn.commit()
    conn.close()

    print(f"\nOK Updated {len(updates)} subject(s).")
    print("Re-run with --dry-run to verify all subjects are now complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
