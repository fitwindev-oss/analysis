"""
QThread wrapper around SessionRecorder.

The recorder's callbacks run on this QThread. Signals are fired with
Qt's AutoConnection (queued when the receiver lives on the GUI thread),
so the UI can safely consume them without extra locking.

Signals:
    camera_frame(cam_id: str, bgr: np.ndarray, frame_idx: int, t_ns: int)
    daq_frame(DaqFrame)
    state_changed(RecorderState)
    log_message(str)
    finished(result: dict)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.capture.daq_reader import DaqFrame
from src.capture.session_recorder import (
    RecorderConfig, RecorderState, SessionRecorder,
)


class RecordWorker(QThread):
    camera_frame  = pyqtSignal(str, object, int, int)  # cam_id, bgr, frame_idx, t_ns
    daq_frame     = pyqtSignal(object)                 # DaqFrame
    state_changed = pyqtSignal(object)                 # RecorderState
    log_message   = pyqtSignal(str)
    finished_ok   = pyqtSignal(dict)                   # result summary

    def __init__(self, cfg: RecorderConfig, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._recorder: Optional[SessionRecorder] = None

    # ── public (main-thread) API ────────────────────────────────────────────
    def cancel(self) -> None:
        if self._recorder is not None:
            self._recorder.cancel()

    def manual_reaction(self, response_type: str) -> None:
        if self._recorder is not None:
            self._recorder.manual_reaction(response_type)

    def manual_random(self) -> None:
        if self._recorder is not None:
            self._recorder.manual_random()

    # ── Multi-set strength assessment controls (Phase V1-E) ─────────────────
    # Forward GUI button presses to the recorder's same-named methods.
    # All four are no-ops when no recording is active OR when the active
    # test is not ``strength_3lift``.

    def end_set(self) -> None:
        if self._recorder is not None:
            self._recorder.end_set()

    def pause_rest(self) -> None:
        if self._recorder is not None:
            self._recorder.pause_rest()

    def resume_rest(self) -> None:
        if self._recorder is not None:
            self._recorder.resume_rest()

    def skip_rest(self) -> None:
        if self._recorder is not None:
            self._recorder.skip_rest()

    def end_session(self) -> None:
        if self._recorder is not None:
            self._recorder.end_session()

    # ── QThread entry point ─────────────────────────────────────────────────
    def run(self) -> None:
        rec = SessionRecorder(self._cfg)
        self._recorder = rec
        rec.set_callbacks(
            on_camera_frame=self._on_camera,
            on_daq_frame=self._on_daq,
            on_state=self._on_state,
            on_log=self._on_log,
        )
        try:
            result = rec.run()
        except Exception as e:
            self.log_message.emit(f"[worker] recorder error: {e}")
            result = {"error": str(e), "cancelled": True}
        self.finished_ok.emit(result)

    # ── bridge recorder callbacks → Qt signals ──────────────────────────────
    def _on_camera(self, cam_id: str, bgr: np.ndarray,
                   frame_idx: int, t_ns: int) -> None:
        # Copy — bgr buffer comes from a subprocess queue and may be reused
        self.camera_frame.emit(cam_id, bgr.copy(), frame_idx, t_ns)

    def _on_daq(self, fr: DaqFrame) -> None:
        self.daq_frame.emit(fr)

    def _on_state(self, st: RecorderState) -> None:
        self.state_changed.emit(st)

    def _on_log(self, msg: str) -> None:
        self.log_message.emit(msg)
