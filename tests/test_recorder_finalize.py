"""
Unit tests for SessionRecorder._finalize ordering + DAQ tail trim
(Phase U3-2).

These exercise the close-window protocol that prevents the DAQ
callback from leaking samples past the camera's last frame:

  1. ``_record_ready.clear()`` at the top of ``_finalize`` causes the
     DAQ callback's gate (``if self._record_ready.is_set()``) to fail,
     so samples arriving after that instant are dropped at the gate.

  2. ``_trim_daq_tail()`` is the post-hoc defense — frames that were
     mid-flight in the callback (had passed the gate but not yet
     appended) are removed by wall-time threshold.

  3. ``MultiCameraCapture.signal_stop()`` + ``wait_join()`` split lets
     the recorder issue stop signals to camera + DAQ back-to-back
     without serialising on the camera's join wait.

Run from project root:
    python -m pytest tests/test_recorder_finalize.py -v

Or directly:
    python tests/test_recorder_finalize.py
"""
from __future__ import annotations

import sys
import threading
import time
import multiprocessing as mp
from pathlib import Path
from unittest.mock import MagicMock

# Make project root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.capture.daq_reader import DaqFrame
from src.capture.session_recorder import SessionRecorder, RecorderConfig
from src.capture.camera_worker import MultiCameraCapture


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def make_recorder() -> SessionRecorder:
    """Build a SessionRecorder without touching real hardware.

    We never call ``run()``, so no threads or processes are spawned.
    Only the in-memory state matters for these tests.
    """
    cfg = RecorderConfig(
        test="cmj",
        duration_s=10.0,
        subject_kg=80.0,
    )
    return SessionRecorder(cfg)


def make_fake_frame(t_wall: float) -> DaqFrame:
    return DaqFrame(
        t_ns=int(t_wall * 1e9),
        t_wall=t_wall,
        forces_n=np.zeros(8, dtype=np.float64),
        enc1_mm=0.0,
        enc2_mm=0.0,
    )


def install_daq_callback(rec: SessionRecorder) -> callable:
    """Replicate the closure inside SessionRecorder._setup_daq that the
    real DAQ thread invokes per sample. We can't reach the original
    closure (it's local), so re-create a behaviourally identical one.
    """
    def _on_daq(fr: DaqFrame) -> None:
        rec._daq_latest_frame = fr
        if rec._record_ready.is_set():
            with rec._daq_lock:
                rec._daq_frames.append(fr)
    return _on_daq


# ────────────────────────────────────────────────────────────────────────────
# T1. _record_ready gate
# ────────────────────────────────────────────────────────────────────────────

def test_record_ready_open_appends_frames():
    """When _record_ready is set, the DAQ callback appends frames."""
    rec = make_recorder()
    cb = install_daq_callback(rec)
    rec._record_ready.set()
    cb(make_fake_frame(100.0))
    cb(make_fake_frame(100.01))
    assert len(rec._daq_frames) == 2


def test_record_ready_closed_blocks_appends():
    """When _record_ready is cleared, the DAQ callback drops frames."""
    rec = make_recorder()
    cb = install_daq_callback(rec)
    rec._record_ready.set()
    cb(make_fake_frame(100.0))
    rec._record_ready.clear()    # close the window
    cb(make_fake_frame(100.05))   # should be ignored
    cb(make_fake_frame(100.10))   # should be ignored
    assert len(rec._daq_frames) == 1
    assert rec._daq_frames[0].t_wall == 100.0


def test_record_ready_clear_atomic_with_stop_signal():
    """Clearing _record_ready takes effect for any callback invocation
    that starts after the clear() call returns. (CPython single-flag
    atomicity guarantee.)"""
    rec = make_recorder()
    cb = install_daq_callback(rec)
    rec._record_ready.set()
    # Simulate ~100 callbacks straddling the clear() call
    cb(make_fake_frame(0.0))
    rec._record_ready.clear()
    for i in range(100):
        cb(make_fake_frame(i + 1.0))
    assert len(rec._daq_frames) == 1


# ────────────────────────────────────────────────────────────────────────────
# T2. _trim_daq_tail
# ────────────────────────────────────────────────────────────────────────────

def test_trim_drops_frames_past_record_end():
    """Frames whose t_wall exceeds _rec_end_wall are removed."""
    rec = make_recorder()
    rec._rec_end_wall = 100.0
    rec._daq_frames = [
        make_fake_frame(99.0),
        make_fake_frame(99.5),
        make_fake_frame(100.0),     # boundary — kept (≤)
        make_fake_frame(100.005),    # past — drop
        make_fake_frame(100.10),     # past — drop
    ]
    rec._trim_daq_tail()
    assert len(rec._daq_frames) == 3
    assert all(fr.t_wall <= 100.0 for fr in rec._daq_frames)


def test_trim_no_op_when_record_end_none():
    """If _rec_end_wall was never set, trim is a no-op."""
    rec = make_recorder()
    rec._rec_end_wall = None
    rec._daq_frames = [make_fake_frame(t) for t in [10.0, 20.0, 30.0]]
    rec._trim_daq_tail()
    assert len(rec._daq_frames) == 3


def test_trim_no_op_when_no_frames():
    """Empty _daq_frames is left as-is."""
    rec = make_recorder()
    rec._rec_end_wall = 100.0
    rec._daq_frames = []
    rec._trim_daq_tail()  # should not raise
    assert rec._daq_frames == []


def test_trim_keeps_all_when_within_window():
    """If every frame is ≤ record_end, none are dropped."""
    rec = make_recorder()
    rec._rec_end_wall = 100.0
    rec._daq_frames = [make_fake_frame(t) for t in [50.0, 75.0, 99.999]]
    rec._trim_daq_tail()
    assert len(rec._daq_frames) == 3


# ────────────────────────────────────────────────────────────────────────────
# T3. MultiCameraCapture signal/join split
# ────────────────────────────────────────────────────────────────────────────

def test_camera_signal_stop_is_non_blocking():
    """signal_stop sets the event and returns immediately."""
    cap = MultiCameraCapture(cameras=[])
    t0 = time.monotonic()
    cap.signal_stop()
    elapsed = time.monotonic() - t0
    # No-op event-set should be sub-millisecond. Generous bound for CI.
    assert elapsed < 0.05
    assert cap._stop.is_set()


def test_camera_wait_join_idempotent_no_workers():
    """wait_join with no spawned processes returns immediately."""
    cap = MultiCameraCapture(cameras=[])
    cap.signal_stop()
    t0 = time.monotonic()
    cap.wait_join(timeout=2.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05
    assert cap._procs == []


def test_camera_legacy_stop_still_works():
    """The original stop() entry point is preserved for callers that
    don't need the split form."""
    cap = MultiCameraCapture(cameras=[])
    # Should not raise, should set stop and clear procs
    cap.stop(timeout=0.5)
    assert cap._stop.is_set()
    assert cap._procs == []


# ────────────────────────────────────────────────────────────────────────────
# T4. Combined close-window simulation
# ────────────────────────────────────────────────────────────────────────────

def test_callback_during_simulated_join_window_is_blocked():
    """Reproduce the original bug end-to-end:

    - ``_record_ready`` is open
    - We simulate the recorder declaring "done" by setting ``_rec_end_wall``
      and clearing ``_record_ready``
    - Then a background thread fires "DAQ callbacks" for 200 ms
      (simulating the join window) — none should make it past the gate
    """
    rec = make_recorder()
    cb = install_daq_callback(rec)
    rec._record_ready.set()

    # Pre-close: 5 frames during recording
    for i in range(5):
        cb(make_fake_frame(100.0 + i * 0.01))

    # Close the window
    rec._rec_end_wall = 100.05
    rec._record_ready.clear()

    # Background "DAQ thread" continues to fire post-close. None should
    # be appended.
    stop = threading.Event()
    def daq_loop():
        i = 0
        while not stop.is_set():
            cb(make_fake_frame(100.10 + i * 0.01))
            i += 1
            time.sleep(0.005)
    t = threading.Thread(target=daq_loop, daemon=True)
    t.start()
    time.sleep(0.20)   # 200 ms of "join wait"
    stop.set()
    t.join(timeout=1.0)

    # Only the 5 pre-close frames remain
    assert len(rec._daq_frames) == 5
    assert all(fr.t_wall <= 100.05 for fr in rec._daq_frames)


def test_post_trim_catches_mid_flight_callback():
    """Even with the gate, simulate a frame that slipped through (the
    callback had already passed the is_set() check). The post-hoc
    trim must catch it."""
    rec = make_recorder()
    rec._record_ready.set()
    rec._daq_frames = [
        make_fake_frame(99.0),
        make_fake_frame(99.5),
        make_fake_frame(100.0),
    ]
    # Simulate close-window
    rec._rec_end_wall = 100.0
    rec._record_ready.clear()
    # Simulate "callback had passed the gate but appends now" — this
    # bypasses the gate entirely (worst case)
    with rec._daq_lock:
        rec._daq_frames.append(make_fake_frame(100.30))
    assert len(rec._daq_frames) == 4    # before trim
    rec._trim_daq_tail()
    assert len(rec._daq_frames) == 3    # after trim
    assert max(fr.t_wall for fr in rec._daq_frames) <= 100.0


# ────────────────────────────────────────────────────────────────────────────
# T5. Metadata
# ────────────────────────────────────────────────────────────────────────────

def test_metadata_includes_record_end_fields():
    """_build_metadata exposes record_end_wall_s + record_end_monotonic_ns
    so replay can use them as the slider's right edge."""
    rec = make_recorder()
    rec._rec_start_ns = 1_000_000_000
    rec._rec_start_wall = 1234567890.0
    rec._rec_end_ns = 11_000_000_000
    rec._rec_end_wall = 1234567900.0
    meta = rec._build_metadata(cancelled=False)
    assert "record_end_wall_s" in meta
    assert "record_end_monotonic_ns" in meta
    assert meta["record_end_wall_s"] == 1234567900.0
    assert meta["record_end_monotonic_ns"] == 11_000_000_000
    # Original fields preserved
    assert meta["record_start_wall_s"] == 1234567890.0


def test_metadata_record_end_none_before_finalize():
    """A recorder that never reached _finalize has record_end = None."""
    rec = make_recorder()
    rec._rec_start_wall = 1234567890.0
    meta = rec._build_metadata(cancelled=True)
    assert meta["record_end_wall_s"] is None
    assert meta["record_end_monotonic_ns"] is None


# ────────────────────────────────────────────────────────────────────────────
# Direct runner (pytest-free)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # No pytest? Run all tests by introspection.
    fns = [v for k, v in dict(globals()).items()
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"=== {failed}/{len(fns)} tests failed ===")
        sys.exit(1)
    else:
        print(f"=== All {len(fns)} tests passed ===")
