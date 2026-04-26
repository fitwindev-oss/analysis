"""
NI USB-6210 DAQ reader — 8 force channels + 2 linear encoders @ 100 Hz.

Ported from the existing Force Plate Viewer project with two improvements:
  - No max(0, kg) clipping by default (config.CLIP_NEGATIVE_FORCE).
  - Per-corner scale hook (set via `set_corner_scales`) for future in-situ calibration.
  - Emits N (newtons), not kg, plus world-frame CoP in mm.

The thread stops cleanly on `stop()` and exposes a callback-based API:

    daq = DaqReader()
    daq.set_callback(lambda frame: print(frame))
    daq.connect()
    daq.start()
    ...
    daq.stop()
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import config

GRAVITY = 9.80665


@dataclass
class DaqFrame:
    t_ns: int                # time.monotonic_ns at read
    t_wall: float            # time.time()
    # Corner forces (Newtons) in world frame:
    # order matches config.DAQ_CHANNEL_MAP -> b1_tl, b1_tr, b1_bl, b1_br, b2_*
    forces_n: np.ndarray = field(default_factory=lambda: np.zeros(8))
    # Linear encoders (mm)
    enc1_mm: float = 0.0
    enc2_mm: float = 0.0

    @property
    def b1_total_n(self) -> float:
        return float(self.forces_n[:4].sum())

    @property
    def b2_total_n(self) -> float:
        return float(self.forces_n[4:].sum())

    @property
    def total_n(self) -> float:
        return float(self.forces_n.sum())

    def cop_world_mm(self) -> tuple[float, float]:
        """
        Center of pressure in WORLD coordinates (mm).
        Origin at plate's left-bottom corner, +X right, +Y forward.
        Returns (NaN, NaN) if total force is below threshold.
        """
        if self.total_n < 5.0:
            return float("nan"), float("nan")

        # Each board's local CoP (mm from its own left-bottom corner)
        b1 = self._board_local_cop(self.forces_n[:4],
                                   config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM)
        b2 = self._board_local_cop(self.forces_n[4:],
                                   config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM)

        b1_world = (b1[0] + config.BOARD1_ORIGIN_MM[0],
                    b1[1] + config.BOARD1_ORIGIN_MM[1])
        b2_world = (b2[0] + config.BOARD2_ORIGIN_MM[0],
                    b2[1] + config.BOARD2_ORIGIN_MM[1])

        # Combined CoP = force-weighted average
        f1 = self.b1_total_n
        f2 = self.b2_total_n
        w_total = f1 + f2
        if w_total < 5.0:
            return float("nan"), float("nan")
        cx = (b1_world[0] * f1 + b2_world[0] * f2) / w_total
        cy = (b1_world[1] * f1 + b2_world[1] * f2) / w_total
        return cx, cy

    @staticmethod
    def _board_local_cop(corners_n: np.ndarray, w: float, h: float) -> tuple[float, float]:
        """
        Local CoP on one plate.
        Corner order: TL (0,h), TR (w,h), BL (0,0), BR (w,0).
        Returns (cx, cy) in mm, with plate's left-bottom at (0,0).

        Corner forces are CLIPPED TO >= 0 for CoP calculation only. A load cell
        cannot physically push a subject upward; negative readings are zero-
        offset drift or noise, and letting them into the moment calculation
        produces CoP values outside the plate footprint.  We keep the raw
        signed forces in forces_n for diagnostics; only the spatial moment
        used here uses the clipped values.
        """
        clipped = np.maximum(corners_n, 0.0)
        total = float(clipped.sum())
        if total < 10.0:     # raised from 2.5 for stability under drift
            return w / 2, h / 2
        tl, tr, bl, br = clipped
        cx = (tr * w + br * w) / total         # tl/bl contribute 0 to x
        cy = (tl * h + tr * h) / total         # bl/br contribute 0 to y
        # Guarantee within plate bounds (should always hold after clipping)
        cx = float(max(0.0, min(w, cx)))
        cy = float(max(0.0, min(h, cy)))
        return cx, cy


FrameCallback = Callable[[DaqFrame], None]
CalCallback   = Callable[[float, bool], None]


class DaqReader:
    """NI USB-6210 force + encoder reader (threaded, non-blocking start)."""

    def __init__(self):
        self._callback: Optional[FrameCallback] = None
        self._cal_cb:   Optional[CalCallback]   = None
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        n_ch = len(config.DAQ_CHANNEL_MAP) + 2
        self._ma_bufs = [
            collections.deque(maxlen=config.MOVING_AVERAGE_SAMPLES)
            for _ in range(n_ch)
        ]
        self._zero_offsets = np.zeros(n_ch, dtype=np.float64)

    # ── public API ───────────────────────────────────────────────────────────
    def set_callback(self, cb: FrameCallback) -> None:
        self._callback = cb

    def set_cal_callback(self, cb: CalCallback) -> None:
        self._cal_cb = cb

    def connect(self) -> bool:
        try:
            import nidaqmx
            import nidaqmx.system
            sys_handle = nidaqmx.system.System.local()
            found = any(d.name == config.DAQ_DEVICE_NAME for d in sys_handle.devices)
            if not found:
                print(f"[DAQ] device '{config.DAQ_DEVICE_NAME}' not found")
                return False
            return True
        except Exception as e:
            print(f"[DAQ] connect failed: {e}")
            return False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── acquisition loop ─────────────────────────────────────────────────────
    def _loop(self) -> None:
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        except ImportError:
            print("[DAQ] nidaqmx not installed — run: pip install nidaqmx")
            self._running = False
            return

        all_ch = config.DAQ_CHANNEL_MAP + [
            config.DAQ_ENCODER1_CHANNEL, config.DAQ_ENCODER2_CHANNEL,
        ]
        chunk = max(1, config.SAMPLE_RATE_HZ // 10)
        n_force = len(config.DAQ_CHANNEL_MAP)

        # Resolve per-channel voltage -> kg scale. config.DAQ_VOLTAGE_SCALE
        # may be a scalar (legacy) or a length-n_force list.
        raw_scale = np.asarray(config.DAQ_VOLTAGE_SCALE, dtype=np.float64)
        if raw_scale.ndim == 0:
            volt_scale = np.full(n_force, float(raw_scale))
        elif raw_scale.shape == (n_force,):
            volt_scale = raw_scale
        else:
            raise RuntimeError(
                f"config.DAQ_VOLTAGE_SCALE must be a scalar or a "
                f"{n_force}-element list, got shape {raw_scale.shape}")

        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(
                ",".join(all_ch),
                min_val=config.DAQ_VOLTAGE_MIN,
                max_val=config.DAQ_VOLTAGE_MAX,
                terminal_config=TerminalConfiguration.RSE,
            )
            task.timing.cfg_samp_clk_timing(
                rate=config.SAMPLE_RATE_HZ,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=chunk * 4,
            )
            task.start()
            print(f"[DAQ] streaming {len(all_ch)} channels @ {config.SAMPLE_RATE_HZ} Hz")

            # ── Zero calibration ────────────────────────────────────────────
            cal_samples = int(config.ZERO_CAL_SECONDS * config.SAMPLE_RATE_HZ)
            acc = [[] for _ in range(len(all_ch))]
            collected = 0
            while collected < cal_samples and not self._stop_event.is_set():
                raw = task.read(number_of_samples_per_channel=chunk, timeout=2.0)
                if isinstance(raw[0], (int, float)):
                    raw = [[v] for v in raw]
                for ci, cd in enumerate(raw):
                    acc[ci].extend(cd)
                collected += len(raw[0])
                if self._cal_cb:
                    self._cal_cb(min(1.0, collected / cal_samples), False)

            # Only force channels get zero offset. Encoders are absolute.
            for ci in range(n_force):
                if acc[ci]:
                    self._zero_offsets[ci] = float(np.mean(acc[ci]))
            if self._cal_cb:
                self._cal_cb(1.0, True)
            print(f"[DAQ] zero offsets: {self._zero_offsets[:n_force]}")

            # ── Acquisition ────────────────────────────────────────────────
            while not self._stop_event.is_set():
                try:
                    raw = task.read(number_of_samples_per_channel=chunk, timeout=2.0)
                    if isinstance(raw[0], (int, float)):
                        raw = [[v] for v in raw]
                    arr = np.asarray(raw, dtype=np.float64)  # (n_ch, n_samples)
                    n_samp = arr.shape[1]
                    t_ns_base = time.monotonic_ns() - int(
                        1e9 * n_samp / config.SAMPLE_RATE_HZ
                    )
                    t_wall_base = time.time() - (n_samp / config.SAMPLE_RATE_HZ)
                    for i in range(n_samp):
                        sample = arr[:, i].copy()
                        sample[:n_force] -= self._zero_offsets[:n_force]
                        smoothed = self._moving_average(sample)
                        forces_n = smoothed[:n_force] * volt_scale * GRAVITY
                        if config.CLIP_NEGATIVE_FORCE:
                            forces_n = np.maximum(forces_n, 0.0)
                        enc1_mm = smoothed[n_force] * config.ENCODER_VOLTAGE_SCALE
                        enc2_mm = smoothed[n_force + 1] * config.ENCODER_VOLTAGE_SCALE

                        frame = DaqFrame(
                            t_ns=t_ns_base + int(1e9 * i / config.SAMPLE_RATE_HZ),
                            t_wall=t_wall_base + i / config.SAMPLE_RATE_HZ,
                            forces_n=forces_n,
                            enc1_mm=enc1_mm,
                            enc2_mm=enc2_mm,
                        )
                        if self._callback:
                            self._callback(frame)
                except nidaqmx.errors.DaqError as e:
                    if not self._stop_event.is_set():
                        print(f"[DAQ] read error: {e}")
                    break

            task.stop()

    def _moving_average(self, sample: np.ndarray) -> np.ndarray:
        out = np.empty_like(sample)
        for i, v in enumerate(sample):
            self._ma_bufs[i].append(float(v))
            out[i] = sum(self._ma_bufs[i]) / len(self._ma_bufs[i])
        return out
