"""
PlaybackController — single clock that drives synchronized replay.

Everything time-related in the replay panel refers to a single scalar,
``current_t_s`` — seconds since force recording started. Subwidgets
(video player, force timeline, CoP trail) subscribe to ``time_changed``
and update their display when the clock ticks.

Design notes (multi-rate):
    Force samples at ~100 Hz, video frames at 30/60 fps. We never iterate
    "index-by-index" across these streams. Instead, the playback clock is
    continuous (float seconds), and each subwidget looks up its own
    native-rate sample by timestamp. This keeps both streams at full
    fidelity and avoids resampling artefacts.

Clock model — wall-clock anchored (Phase U3):
    The tick handler does NOT advance ``current_t_s`` by a fixed
    "1/TICK_HZ × speed" step. Doing that would couple playback rate to
    the slot's processing time: when the per-tick work (video decode +
    multiple pyqtgraph repaints) takes longer than the timer interval,
    QTimer waits for the slot to finish before re-firing, so wall time
    races ahead of the clock and the user sees a "slow-motion" 1× that
    is actually ~0.4× when the work-cost per tick exceeds 33 ms.

    Instead, ``play()`` snapshots a wall-time anchor (``_wall_anchor``)
    + a clock anchor (``_t_anchor``), and every tick re-derives the
    current clock position from the elapsed wall time. Heavy ticks
    cause frame *skips* (lookups are stateless and cheap), but never
    clock drift. ``seek()`` and ``set_speed()`` reset the anchors so
    the wall-clock relationship stays correct across user actions.
"""
from __future__ import annotations

import time
from typing import Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


TICK_HZ = 30   # display tick frequency


class PlaybackController(QObject):
    time_changed  = pyqtSignal(float)   # current_t_s
    state_changed = pyqtSignal(bool)    # is_playing
    speed_changed = pyqtSignal(float)   # playback speed multiplier

    def __init__(self, duration_s: float = 0.0, parent=None):
        super().__init__(parent)
        self._duration_s = float(max(0.0, duration_s))
        self._current_t_s = 0.0
        self._speed = 1.0
        self._is_playing = False
        # Wall-clock anchors — set on play()/seek()/set_speed() so the
        # tick handler can derive ``current_t_s`` from real elapsed time
        # rather than from a fixed per-tick step.
        self._wall_anchor: float = 0.0
        self._t_anchor: float = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / TICK_HZ))
        self._timer.timeout.connect(self._on_tick)

    # ── public API ─────────────────────────────────────────────────────────
    @property
    def duration_s(self) -> float: return self._duration_s

    @property
    def current_t_s(self) -> float: return self._current_t_s

    @property
    def is_playing(self) -> bool: return self._is_playing

    @property
    def speed(self) -> float: return self._speed

    def set_duration(self, duration_s: float) -> None:
        self._duration_s = max(0.0, float(duration_s))
        if self._current_t_s > self._duration_s:
            self.seek(self._duration_s)

    def seek(self, t_s: float) -> None:
        t = max(0.0, min(self._duration_s, float(t_s)))
        if t == self._current_t_s:
            # Even a no-op seek must re-anchor while playing, otherwise
            # the wall-time delta accumulated since play() will keep
            # advancing past this position next tick.
            if self._is_playing:
                self._wall_anchor = time.monotonic()
                self._t_anchor = t
            return
        self._current_t_s = t
        # Re-anchor so wall-clock elapsed time is measured from this
        # new position, not from where the user was before the seek.
        if self._is_playing:
            self._wall_anchor = time.monotonic()
            self._t_anchor = t
        self.time_changed.emit(t)

    def play(self) -> None:
        if self._is_playing:
            return
        # Restart from 0 if at end
        if self._current_t_s >= self._duration_s - 1e-3:
            self.seek(0.0)
        self._is_playing = True
        # Anchor the wall clock so the tick handler can compute
        # next_t = t_anchor + (now - wall_anchor) × speed.
        self._wall_anchor = time.monotonic()
        self._t_anchor = self._current_t_s
        self._timer.start()
        self.state_changed.emit(True)

    def pause(self) -> None:
        if not self._is_playing:
            return
        self._is_playing = False
        self._timer.stop()
        self.state_changed.emit(False)

    def toggle(self) -> None:
        (self.pause if self._is_playing else self.play)()

    def set_speed(self, speed: float) -> None:
        # If we're playing, capture the clock position implied by the
        # *old* speed before the change, then re-anchor so the new speed
        # takes effect from "now" without a position jump.
        if self._is_playing:
            elapsed = time.monotonic() - self._wall_anchor
            self._current_t_s = max(0.0, min(
                self._duration_s,
                self._t_anchor + elapsed * self._speed))
            self._wall_anchor = time.monotonic()
            self._t_anchor = self._current_t_s
        self._speed = max(0.05, float(speed))
        self.speed_changed.emit(self._speed)

    # ── internals ──────────────────────────────────────────────────────────
    def _on_tick(self) -> None:
        # Wall-clock anchored advancement: the clock now matches real
        # elapsed time × speed, regardless of how long the previous
        # tick's ``time_changed`` handlers took. Heavy work just causes
        # frame skips on the next lookup — never slow-motion drift.
        elapsed = time.monotonic() - self._wall_anchor
        next_t = self._t_anchor + elapsed * self._speed
        if next_t >= self._duration_s:
            self.seek(self._duration_s)
            self.pause()
            return
        # Use _emit_seek instead of seek() so the anchor isn't reset on
        # every tick (that would no-op the wall-clock catch-up).
        if next_t == self._current_t_s:
            return
        self._current_t_s = next_t
        self.time_changed.emit(next_t)
