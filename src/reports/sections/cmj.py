"""CMJ charts section — force-time curve with BW line + peak + phase markers.

The analyzer's result.json stores only summary scalars, not the phase
boundary timestamps. For a cleaner report we recompute the takeoff and
landing times here by thresholding vGRF < 30 N from the raw forces.csv.
Cheap (a few ms) and makes the chart self-contained.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np

from src.reports.base import ReportContext, ReportSection
from src.reports.charts import make_cmj_force_time, png_data_uri
from src.reports.fonts import pdf_font_family


class CmjChartsSection(ReportSection):

    def _prep(self, ctx: ReportContext):
        try:
            from src.analysis.common import load_force_session
            force = load_force_session(ctx.session_dir)
        except Exception:
            return None
        r = ctx.result or {}
        bw_n = r.get("bw_n")
        vgrf = force.total
        t = force.t_s

        # Preferred: use takeoff/landing that the analyzer already computed.
        take_t = r.get("t_takeoff_s")
        land_t = r.get("t_landing_s")

        # Fallback for older sessions without those keys — use expected
        # flight duration (flight_time_s) to find the CORRECT <30N gap
        # (not just the longest one; if the subject stepped off the plate
        # after landing, that gap is longer than the real flight phase).
        if (take_t is None or land_t is None):
            below = vgrf < 30.0
            if below.any():
                idx = np.flatnonzero(below)
                if len(idx) > 2:
                    gaps   = np.flatnonzero(np.diff(idx) > 1)
                    starts = np.r_[idx[0], idx[gaps + 1]] if len(gaps) \
                        else np.array([idx[0]])
                    ends   = np.r_[idx[gaps], idx[-1]] if len(gaps) \
                        else np.array([idx[-1]])
                    fs = 1.0 / float(np.mean(np.diff(t))) \
                        if len(t) > 1 else 100.0
                    lens_s = (ends - starts) / fs
                    expected = r.get("flight_time_s")
                    if expected and expected > 0:
                        # Pick the run whose length is closest to the
                        # analyzer-reported flight time (within 60% tolerance
                        # so we don't latch onto an obviously wrong gap).
                        diffs = np.abs(lens_s - float(expected))
                        best = int(np.argmin(diffs))
                        if diffs[best] <= 0.6 * float(expected):
                            take_t = float(t[starts[best]])
                            land_t = float(t[min(ends[best] + 1, len(t) - 1)])
                    if take_t is None:
                        # Last resort: longest < 30N run of modest length
                        best = int(np.argmax(lens_s))
                        if lens_s[best] > 0.05:
                            take_t = float(t[starts[best]])
                            land_t = float(t[min(ends[best] + 1, len(t) - 1)])

        # Peak PROPULSION force: restrict to pre-takeoff so the landing
        # impact spike is never picked.
        peak_t = None
        if take_t is not None:
            mask = t < take_t
            if mask.any():
                peak_idx = int(np.argmax(np.where(mask, vgrf, -np.inf)))
                peak_t = float(t[peak_idx])
        return {"t": t, "vgrf": vgrf, "bw_n": bw_n,
                "takeoff_t": take_t, "landing_t": land_t, "peak_t": peak_t}

    def _png(self, ctx: ReportContext) -> bytes | None:
        p = self._prep(ctx)
        if p is None:
            return None
        return make_cmj_force_time(
            p["t"], p["vgrf"], bw_n=p["bw_n"],
            takeoff_t=p["takeoff_t"], landing_t=p["landing_t"],
            peak_t=p["peak_t"],
        )

    def to_html(self, ctx: ReportContext) -> str:
        png = self._png(ctx)
        if png is None:
            return ""
        return (
            "<h2>힘-시간 곡선</h2>"
            f"<img class='chart' src='{png_data_uri(png)}' "
            f"style='max-width:100%; height:auto;'>"
        )

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        png = self._png(ctx)
        if png is None:
            return []
        from reportlab.platypus import Image, Paragraph, Spacer, KeepTogether
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm
        family = pdf_font_family()
        h2 = ParagraphStyle("h2", fontName=family, fontSize=12,
                             textColor=HexColor("#1976D2"),
                             spaceBefore=10, spaceAfter=6)
        return [KeepTogether([
            Paragraph("힘-시간 곡선", h2),
            Image(BytesIO(png), width=170*mm, height=72*mm),
            Spacer(1, 6),
        ])]
