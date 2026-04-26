"""
Backfill the session_metrics cache from existing result.json files.

Run once after adding the session_metrics table; subsequent sessions get
their metrics populated automatically by the AnalysisWorker finish hook.

Usage:
    python scripts/backfill_session_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import models, schema
from src.reports.key_metrics import extract_key_metrics, variant_from_meta


def main() -> None:
    schema.initialise()
    sessions = models.list_sessions(limit=10000)
    print(f"scanning {len(sessions)} sessions …")
    n_ok = 0; n_skip = 0; n_err = 0
    for s in sessions:
        sdir = s.session_dir and Path(s.session_dir)
        if not sdir or not sdir.exists():
            n_skip += 1; continue
        result_json = sdir / "result.json"
        if not result_json.exists():
            n_skip += 1; continue
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
            result = payload.get("result") or {}
            if not result or payload.get("error"):
                n_skip += 1; continue
            metrics = extract_key_metrics(s.test_type, result)
            if not metrics:
                n_skip += 1; continue
            # Load session.json for variant info
            meta = {}
            sj = sdir / "session.json"
            if sj.exists():
                try:
                    meta = json.loads(sj.read_text(encoding="utf-8"))
                except Exception:
                    pass
            row = models.SessionMetricsRow(
                session_id=s.id, subject_id=s.subject_id,
                test_type=s.test_type,
                variant=variant_from_meta(s.test_type, meta),
                session_date=s.session_date,
                metrics=metrics,
            )
            models.upsert_session_metrics(row)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [!] {s.id}: {type(e).__name__} {e}")
    print(f"\ndone. populated={n_ok}  skipped={n_skip}  errors={n_err}")


if __name__ == "__main__":
    main()
