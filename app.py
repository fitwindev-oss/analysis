"""
Biomech MoCap desktop application — new 3-tab entry point.

Launch:
    python app.py

Contains:
    Subjects tab   — subject profile CRUD
    Measure tab    — test selection + recording (Phase 2+)
    Reports tab    — session list + report viewer (Phase 4+)

The legacy live-preview window remains available via main.py for debugging.
"""
from __future__ import annotations

import site
import sys
from pathlib import Path

# Ensure repo root on PYTHONPATH regardless of launch cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure user site-packages is on sys.path. When the interpreter is invoked
# from certain contexts (IDE launchers, some shells, Task Scheduler) the
# automatic user-site append can be skipped even though ENABLE_USER_SITE
# is True. We add every plausible user-site directory so that packages
# installed via `pip install --user` (e.g. mediapipe) are visible.
def _candidate_user_sites() -> list[Path]:
    out: list[Path] = []
    us = getattr(site, "USER_SITE", None)
    if us:
        out.append(Path(us))
    try:
        out.extend(Path(p) for p in site.getusersitepackages().split(";")
                   if p.strip())
    except Exception:
        pass
    # Windows AppData/Roaming fallback
    import os as _os
    appdata = _os.environ.get("APPDATA")
    if appdata:
        vers = f"Python{sys.version_info.major}{sys.version_info.minor}"
        out.append(Path(appdata) / "Python" / vers / "site-packages")
    # Dedupe while preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq

for _p in _candidate_user_sites():
    ps = str(_p)
    if _p.exists() and ps not in sys.path:
        sys.path.append(ps)

# Force stdout/stderr to UTF-8 so Korean + dash characters don't crash
# on cp949 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")     # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")     # type: ignore[attr-defined]
except Exception:
    pass

from PyQt6.QtWidgets import QApplication

import config
from src.ui.app_window import AppWindow


# Branded theme (FITWIN palette + Pretendard + dark QSS) lives in
# src/ui/theme.py. Apply it via ``apply_theme(app)`` in main() below.
# The legacy DARK_QSS inline string was removed in the Phase R rebrand.


def _check_mediapipe_available() -> tuple[bool, str]:
    """Diagnostic import check so pose errors are surfaced up-front."""
    try:
        import mediapipe as _mp    # noqa: F401
        return True, ""
    except Exception as e:
        # Search likely install locations so the user can see exactly where
        # mediapipe lives (or not) vs. where we're looking.
        candidates = _candidate_user_sites() + [
            Path(sys.prefix) / "Lib" / "site-packages",
        ]
        found_at: list[str] = []
        for c in candidates:
            mp_dir = c / "mediapipe"
            if mp_dir.exists():
                found_at.append(str(mp_dir))
        hint_lines = [
            f"sys.executable = {sys.executable}",
            f"ENABLE_USER_SITE = {site.ENABLE_USER_SITE}",
            "sys.path:",
            *[f"  {p}" for p in sys.path],
            "mediapipe dir search:",
            *[f"  FOUND: {f}" for f in found_at],
        ]
        if not found_at:
            hint_lines.append("  (not found in any candidate location)")
        hint_lines.append(
            f"install with:  \"{sys.executable}\" -m pip install "
            f"\"mediapipe>=0.10,<0.11\""
        )
        return False, f"{type(e).__name__}: {e}\n" + "\n".join(hint_lines)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_TITLE)
    app.setApplicationVersion(config.APP_VERSION)

    # FITWIN brand theme — loads QSS, registers Pretendard, tunes
    # pyqtgraph + matplotlib palettes in one call.
    from src.ui.theme import apply_theme
    theme_summary = apply_theme(app)
    print(f"[theme] qss_loaded={theme_summary['qss_loaded']} "
          f"pretendard={theme_summary['pretendard']!r}", flush=True)
    for msg in theme_summary.get("fallbacks", []):
        print(f"[theme] fallback: {msg}", flush=True)

    # Early mediapipe sanity check — prints a clear message if the module is
    # unreachable from this interpreter. Pose features stay disabled until
    # the user fixes the install / relaunches.
    mp_ok, mp_err = _check_mediapipe_available()
    if not mp_ok:
        print(f"[pose] MediaPipe unavailable:\n{mp_err}", flush=True)
    else:
        print(f"[pose] MediaPipe OK", flush=True)

    # One-time camera probe — unreachable devices (eg. flaky USB cams) are
    # pruned from config.CAMERAS so downstream code sees only live cameras.
    from src.capture.camera_detector import detect_available_cameras
    declared = list(config.CAMERAS)
    available, probe_results = detect_available_cameras(declared)
    config.CAMERAS = available
    print(f"[cam] {len(available)}/{len(declared)} cameras available "
          f"({', '.join(c['id'] for c in available)})", flush=True)

    window = AppWindow(camera_probe_results=probe_results)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
