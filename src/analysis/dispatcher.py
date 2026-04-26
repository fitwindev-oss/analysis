"""
Analysis dispatcher — routes a session folder to the correct analyzer and
persists the result as `result.json` alongside the raw recording.

Public entry point:
    result = analyze_session(session_dir, test_type=None)

If test_type is None, it's read from session.json. The returned dict is the
same content that was written to result.json.

Contract for all analyzers:
    - Input: session directory path
    - Output: a dataclass with .to_dict()
    - Failures raise exceptions; the dispatcher catches + records them.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional


# Lazy-import each analyzer so a single broken module doesn't break the
# whole dispatcher. Each entry takes a session_dir and returns a dataclass.
_ANALYZERS: dict[str, Callable[[Any], Any]] = {}


def _register_lazy():
    """Populate _ANALYZERS on first use (avoids heavy import at module load)."""
    if _ANALYZERS:
        return
    from .balance        import analyze_balance_file
    from .cmj            import analyze_cmj_file
    from .squat          import analyze_squat_file
    from .encoder        import analyze_encoder_file
    from .reaction       import analyze_reaction_file
    from .proprio        import analyze_proprio_file
    from .free_exercise  import analyze_free_exercise_file
    from .strength_3lift import analyze_strength_3lift_file
    from .cognitive_reaction import analyze_cognitive_reaction_file

    _ANALYZERS.update({
        "balance_eo":     analyze_balance_file,
        "balance_ec":     analyze_balance_file,
        "cmj":            analyze_cmj_file,
        # Phase V4 — Squat Jump uses the same takeoff/landing detection
        # pipeline as CMJ. The biomechanical difference (no
        # counter-movement) doesn't change the force-time signature
        # the analyzer cares about (quiet → flight → landing).
        "sj":             analyze_cmj_file,
        "squat":          analyze_squat_file,
        "overhead_squat": analyze_squat_file,
        "encoder":        analyze_encoder_file,
        "reaction":            analyze_reaction_file,
        "cognitive_reaction":  analyze_cognitive_reaction_file,
        "proprio":             analyze_proprio_file,
        "free_exercise":       analyze_free_exercise_file,
        "strength_3lift":      analyze_strength_3lift_file,
    })


def _to_dict(obj: Any) -> Any:
    """Recursive dataclass → dict conversion."""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def supported_tests() -> list[str]:
    _register_lazy()
    return sorted(_ANALYZERS.keys())


def analyze_session(session_dir: str | Path,
                    test_type: Optional[str] = None,
                    write_result: bool = True) -> dict:
    """Run the appropriate analyzer on a session directory.

    Returns a result dict with this shape on success:
        {
            "test": "balance_eo",
            "analyzed_at": "2026-04-22T15:30:12+09:00",
            "duration_analysis_s": 0.83,
            "result": { <analyzer-specific payload> },
            "error": None,
        }
    On failure:
        {
            "test": "...", "analyzed_at": "...", "duration_analysis_s": N,
            "result": None,
            "error": "<exception message>",
            "traceback": "...",
        }
    """
    _register_lazy()
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"session_dir does not exist: {session_dir}")

    # Resolve test_type from session.json if not given
    if test_type is None:
        meta_path = session_dir / "session.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"cannot infer test_type — {meta_path} missing")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        test_type = meta.get("test")
        if not test_type:
            raise RuntimeError("session.json missing 'test' field")

    fn = _ANALYZERS.get(test_type)
    if fn is None:
        raise RuntimeError(f"no analyzer registered for test: {test_type!r}")

    import datetime as _dt
    t0 = time.monotonic()
    payload: dict = {
        "test":                 test_type,
        "analyzed_at":          _dt.datetime.now().astimezone().isoformat(
                                   timespec="seconds"),
        "duration_analysis_s":  0.0,
        "result":               None,
        "error":                None,
    }
    try:
        res = fn(session_dir)
        payload["result"] = _to_dict(res)
    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"
        payload["traceback"] = traceback.format_exc()
    payload["duration_analysis_s"] = round(time.monotonic() - t0, 3)

    if write_result:
        out = session_dir / "result.json"
        out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    return payload


def read_result(session_dir: str | Path) -> Optional[dict]:
    """Return the cached result.json payload, or None if absent/broken."""
    p = Path(session_dir) / "result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
