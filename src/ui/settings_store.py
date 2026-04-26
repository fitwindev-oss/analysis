"""
QSettings-backed persistence for window geometry + splitter sizes.

Centralised so each widget that wants to save/restore size only adds
two lines (one in __init__ to call ``restore_splitter`` and one in
the close path to call ``save_splitter``).

Storage keys live under the FITWIN organisation namespace so a future
sibling app sees a clean settings space.
"""
from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtCore import QSettings, QByteArray
from PyQt6.QtWidgets import QSplitter, QMainWindow

ORG  = "FITWIN"
APP  = "biomech-mocap"


def _settings() -> QSettings:
    """Singleton-ish accessor — QSettings is cheap to construct."""
    return QSettings(ORG, APP)


# ── Window geometry ──────────────────────────────────────────────────

def save_window(window: QMainWindow, key: str = "main") -> None:
    s = _settings()
    s.setValue(f"window/{key}/geometry", window.saveGeometry())
    s.setValue(f"window/{key}/state",    window.saveState())
    s.setValue(f"window/{key}/maximised", window.isMaximized())


def restore_window(window: QMainWindow, key: str = "main") -> None:
    """Restore window geometry + maximised flag if previously saved.
    No-op on first run (everything stays at the values set in __init__)."""
    s = _settings()
    geo = s.value(f"window/{key}/geometry")
    if isinstance(geo, QByteArray) and not geo.isEmpty():
        window.restoreGeometry(geo)
    state = s.value(f"window/{key}/state")
    if isinstance(state, QByteArray) and not state.isEmpty():
        window.restoreState(state)
    if str(s.value(f"window/{key}/maximised", "false")).lower() == "true":
        window.showMaximized()


# ── Splitter sizes ───────────────────────────────────────────────────

def save_splitter(splitter: QSplitter, key: str) -> None:
    """Persist splitter sizes under ``splitter/<key>``. Uses
    ``saveState()`` so collapsed/expanded states are preserved too."""
    if splitter is None:
        return
    _settings().setValue(f"splitter/{key}", splitter.saveState())


def restore_splitter(splitter: QSplitter, key: str,
                      default_sizes: Optional[Iterable[int]] = None) -> bool:
    """Restore splitter sizes if previously saved. Returns True on
    success, False if no value was stored. When False, the caller's
    own ``setSizes(default_sizes)`` should run as fallback."""
    if splitter is None:
        return False
    state = _settings().value(f"splitter/{key}")
    if isinstance(state, QByteArray) and not state.isEmpty():
        ok = splitter.restoreState(state)
        if ok:
            return True
    if default_sizes is not None:
        splitter.setSizes(list(default_sizes))
    return False
