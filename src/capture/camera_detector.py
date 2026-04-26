"""
Camera availability detector.

Run once at app startup to filter `config.CAMERAS` down to physically
reachable devices. Each declared camera idx is probed with
`cv2.VideoCapture(idx, CAP_DSHOW)` and must return at least one frame
within a short time budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import cv2


@dataclass
class CameraProbeResult:
    id:        str
    index:     int
    label:     str
    available: bool
    reason:    str = ""


def probe_camera(entry: dict, timeout_s: float = 0.7) -> CameraProbeResult:
    """Try to open one camera and read a single frame.

    Succeeds only when the device opens AND a frame is received within
    `timeout_s`. Always releases the handle before returning.
    """
    cam_id = entry.get("id", "?")
    idx    = int(entry.get("index", -1))
    label  = entry.get("label", "")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        try: cap.release()
        except Exception: pass
        return CameraProbeResult(cam_id, idx, label, False,
                                 reason="open_failed")
    t_end = time.monotonic() + float(timeout_s)
    ok = False
    while time.monotonic() < t_end:
        got, _ = cap.read()
        if got:
            ok = True
            break
        time.sleep(0.02)
    try: cap.release()
    except Exception: pass
    if not ok:
        return CameraProbeResult(cam_id, idx, label, False,
                                 reason="read_timeout")
    return CameraProbeResult(cam_id, idx, label, True)


def detect_available_cameras(cameras: Iterable[dict],
                             timeout_s: float = 0.7) -> tuple[
                                 list[dict], list[CameraProbeResult]]:
    """Probe each entry, return (available_subset, full_probe_results).

    The available subset preserves the original `config.CAMERAS` dict shape
    so existing code (``MultiCameraCapture`` etc.) can swap it in verbatim.
    """
    results: list[CameraProbeResult] = []
    for entry in cameras:
        results.append(probe_camera(entry, timeout_s=timeout_s))
    available = [
        {"id": r.id, "index": r.index, "label": r.label}
        for r in results if r.available
    ]
    return available, results
