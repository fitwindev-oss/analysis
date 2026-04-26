"""
Offline TDMS reader — compatible with the existing Force Plate Viewer format.

    Group:  "Force Data"
    Channels (kg):  Board1_TL, Board1_TR, Board1_BL, Board1_BR,
                    Board2_TL, Board2_TR, Board2_BL, Board2_BR,
                    Board1_Total, Board2_Total, COP_X, COP_Y,
                    Encoder1_mm, Encoder2_mm
    Also:  timestamp

This module converts loaded kg-scale values into Newtons and provides a
world-frame CoP in mm, matching the realtime DaqFrame API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

import config
from src.capture.daq_reader import DaqFrame, GRAVITY


class TdmsReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, np.ndarray] | None = None

    def load(self) -> dict[str, np.ndarray]:
        if self._data is not None:
            return self._data
        try:
            from nptdms import TdmsFile
        except ImportError as e:
            raise RuntimeError("nptdms not installed: pip install nptdms") from e

        tdms = TdmsFile.read(str(self.path))
        if config.TDMS_GROUP_NAME not in [g.name for g in tdms.groups()]:
            raise ValueError(
                f"TDMS group '{config.TDMS_GROUP_NAME}' not found in {self.path}"
            )

        group = tdms[config.TDMS_GROUP_NAME]
        result: dict[str, np.ndarray] = {ch.name: ch[:] for ch in group.channels()}

        # Synthesize timestamp if absent
        if "timestamp" not in result and result:
            n = len(next(iter(result.values())))
            result["timestamp"] = np.linspace(
                0, n / config.SAMPLE_RATE_HZ, n, endpoint=False
            )
        self._data = result
        return result

    def metadata(self) -> dict:
        try:
            from nptdms import TdmsFile
        except ImportError:
            return {}
        tdms = TdmsFile.read(str(self.path))
        return dict(tdms.properties)

    def iter_frames(self) -> Iterator[DaqFrame]:
        """Yield DaqFrame objects compatible with the realtime pipeline."""
        d = self.load()
        n = len(d["timestamp"])
        corners = np.stack([
            d["Board1_TL"], d["Board1_TR"], d["Board1_BL"], d["Board1_BR"],
            d["Board2_TL"], d["Board2_TR"], d["Board2_BL"], d["Board2_BR"],
        ], axis=1)  # (N, 8) in kg

        forces_n_all = corners * GRAVITY  # kg -> N

        # TDMS stores wall-clock seconds as float. We map:
        #   t_wall = d['timestamp']
        #   t_ns   = int(t_wall * 1e9)  (used for relative syncing downstream)
        t_wall = d["timestamp"].astype(np.float64)
        enc1 = d.get("Encoder1_mm", np.zeros(n))
        enc2 = d.get("Encoder2_mm", np.zeros(n))

        for i in range(n):
            yield DaqFrame(
                t_ns=int(t_wall[i] * 1e9),
                t_wall=float(t_wall[i]),
                forces_n=forces_n_all[i],
                enc1_mm=float(enc1[i]),
                enc2_mm=float(enc2[i]),
            )

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return convenient batched arrays (N samples)."""
        d = self.load()
        corners_kg = np.stack([
            d["Board1_TL"], d["Board1_TR"], d["Board1_BL"], d["Board1_BR"],
            d["Board2_TL"], d["Board2_TR"], d["Board2_BL"], d["Board2_BR"],
        ], axis=1)
        forces_n = corners_kg * GRAVITY
        return {
            "t_wall":    d["timestamp"].astype(np.float64),
            "forces_n":  forces_n,
            "b1_total_n": forces_n[:, :4].sum(axis=1),
            "b2_total_n": forces_n[:, 4:].sum(axis=1),
            "total_n":   forces_n.sum(axis=1),
            "encoder1_mm": d.get("Encoder1_mm", np.zeros(len(d["timestamp"]))),
            "encoder2_mm": d.get("Encoder2_mm", np.zeros(len(d["timestamp"]))),
            "cop_x_norm": d.get("COP_X", np.zeros(len(d["timestamp"]))),
            "cop_y_norm": d.get("COP_Y", np.zeros(len(d["timestamp"]))),
        }
