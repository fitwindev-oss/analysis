"""
Matplotlib chart generators — return PNG bytes for embedding in both
HTML (base64 data-URI) and PDF (reportlab ``Image`` from BytesIO).

Every plotting function:
  - sets up Korean fonts on first call (fonts.setup_korean_fonts)
  - renders onto a figure at ``dpi`` (default 150 = print-friendly)
  - returns ``bytes`` (PNG)
  - closes its figure to free memory (matters when 20+ charts per report)
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Iterable, Optional, Sequence

import numpy as np

from src.reports.fonts import setup_korean_fonts
from src.reports.palette import (
    ANGLE_COLORS, BOARD1_COLOR, BOARD2_COLOR, COORD_COLORS,
    HISTORY_DOT, HISTORY_LINE, NORM_BAND_FILL, NORM_BAND_LINE,
    STATUS_CAUTION, STATUS_OK, STATUS_WARNING, TOTAL_COLOR,
)


def _fig_to_png_bytes(fig, dpi: int = 220) -> bytes:
    """Render a matplotlib figure to PNG bytes.

    DPI default bumped to 220 (from 150) so the source PNG carries
    ~2× pixel density. Qt's QTextBrowser scales embedded images down
    to fit the current viewport width — at 150 DPI, narrow windows
    cause Korean glyphs (already small in 8-10 pt legends) to alias
    into unreadable boxes. With 220 DPI, the subsampling has enough
    source detail to keep Hangul strokes crisp even when the viewer
    is resized down.

    Trade-off: ~2× base64 payload size. Still fine for embedded
    reports (chart stays under ~200 KB per image).
    """
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def png_data_uri(png_bytes: bytes) -> str:
    """Convert PNG bytes to a ``data:image/png;base64,...`` URI for HTML ``<img>``."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
# Common charts
# ─────────────────────────────────────────────────────────────────────────────

def make_history_trend(values: Sequence[float],
                       dates: Sequence[str],
                       metric_label: str,
                       unit: str = "",
                       norm_range: Optional[tuple[float, float]] = None,
                       width_in: float = 7.5,
                       height_in: float = 2.6) -> bytes:
    """Line + dots chart of a metric across sessions, with optional
    shaded normal-range band.

    ``values`` and ``dates`` must align element-wise (oldest → newest).
    ``norm_range`` = (low, high); draws a green band between them.
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width_in, height_in),
                           facecolor="white")
    ax.set_facecolor("white")
    x = np.arange(len(values))

    if norm_range is not None and all(v is not None for v in norm_range):
        lo, hi = float(norm_range[0]), float(norm_range[1])
        ax.axhspan(lo, hi, color=NORM_BAND_FILL, alpha=0.7,
                   label=f"정상 범위 ({lo:.0f}–{hi:.0f}{unit})")
        ax.axhline(lo, color=NORM_BAND_LINE, linewidth=0.8, linestyle="--")
        ax.axhline(hi, color=NORM_BAND_LINE, linewidth=0.8, linestyle="--")

    # Filter out None values while preserving x positions
    vs = [np.nan if v is None else float(v) for v in values]
    ax.plot(x, vs, color=HISTORY_LINE, linewidth=2, marker="o",
            markerfacecolor=HISTORY_DOT, markeredgecolor="white",
            markersize=7, label=metric_label)

    # Annotate latest value
    if len(vs) > 0 and not np.isnan(vs[-1]):
        ax.annotate(f"{vs[-1]:.1f}{unit}",
                    xy=(x[-1], vs[-1]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    color=HISTORY_DOT)

    ax.set_title(f"{metric_label} 추이", fontsize=12, pad=10)
    ax.set_ylabel(unit or "value")
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def make_status_bar(label: str, value: float, unit: str,
                    thresholds: Optional[tuple[float, float, float]] = None,
                    width_in: float = 3.8, height_in: float = 0.7) -> bytes:
    """Horizontal gauge: single value with ok/caution/warning bands.

    ``thresholds`` = (ok_hi, caution_hi, max). Zones:
        [0, ok_hi]          → green
        (ok_hi, caution_hi] → amber
        (caution_hi, max]   → red
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    if thresholds is None:
        hi = max(abs(value) * 1.4, 1.0)
        thresholds = (hi * 0.6, hi * 0.8, hi)
    ok, cau, mx = thresholds

    ax.barh([0], [ok],       color=STATUS_OK,       edgecolor="none")
    ax.barh([0], [cau - ok], left=[ok],  color=STATUS_CAUTION, edgecolor="none")
    ax.barh([0], [mx - cau], left=[cau], color=STATUS_WARNING, edgecolor="none")
    ax.axvline(value, color="black", linewidth=2.5)
    ax.text(value, 0.4, f"{value:.2f}{unit}", ha="center", fontsize=10,
            fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, mx)
    ax.set_title(label, fontsize=10, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Balance
# ─────────────────────────────────────────────────────────────────────────────

def make_stabilogram(cop_x: np.ndarray, cop_y: np.ndarray,
                     plate_w_mm: float = 558.0, plate_h_mm: float = 432.0,
                     board_w_mm: float = 279.0, board_h_mm: float = 432.0,
                     board1_origin: tuple[float, float] = (0.0, 0.0),
                     board2_origin: tuple[float, float] = (279.0, 0.0),
                     width_in: float = 5.0,
                     height_in: float = 4.0) -> bytes:
    """2D CoP trajectory with plate outline + 95% ellipse + mean marker."""
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_facecolor("white")
    # Plate outlines
    for (x0, y0), color, label in [
        (board1_origin, BOARD1_COLOR, "Board1"),
        (board2_origin, BOARD2_COLOR, "Board2"),
    ]:
        rect = mpatches.Rectangle((x0, y0), board_w_mm, board_h_mm,
                                   fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
    # Valid CoP only
    mask = (~np.isnan(cop_x)) & (~np.isnan(cop_y))
    x = cop_x[mask]; y = cop_y[mask]
    if len(x) > 3:
        ax.plot(x, y, color="#CE93D8", linewidth=1.2, alpha=0.85,
                label="CoP 경로")
        # 95% confidence ellipse (Mahalanobis 2.447 for 2D 95%)
        mx_, my_ = float(x.mean()), float(y.mean())
        cov = np.cov(x, y)
        w, v = np.linalg.eigh(cov)
        order = w.argsort()[::-1]
        w, v = w[order], v[:, order]
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
        width_e  = 2 * 2.447 * np.sqrt(max(w[0], 1e-9))
        height_e = 2 * 2.447 * np.sqrt(max(w[1], 1e-9))
        ell = mpatches.Ellipse((mx_, my_), width_e, height_e, angle=angle,
                                fill=False, edgecolor="#2E7D32", linewidth=2,
                                linestyle="-", label="95% 타원")
        ax.add_patch(ell)
        ax.plot([mx_], [my_], marker="+", color="#2E7D32", markersize=14,
                markeredgewidth=2.5, label="평균")
    ax.set_xlim(-10, plate_w_mm + 10)
    ax.set_ylim(-10, plate_h_mm + 10)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm, 좌 → 우)")
    ax.set_ylabel("Y (mm, 후 → 전)")
    ax.set_title("Stabilogram (CoP 경로)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def make_cop_timeseries(t: np.ndarray, cop_x: np.ndarray, cop_y: np.ndarray,
                        analysis_window: Optional[tuple[float, float]] = None,
                        width_in: float = 7.5,
                        height_in: float = 3.2) -> bytes:
    """Two stacked subplots: CoP ML (X) and AP (Y) vs time."""
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(width_in, height_in), sharex=True, facecolor="white")
    for ax, y, label, color in [
        (ax1, cop_x, "ML (좌우, mm)", BOARD1_COLOR),
        (ax2, cop_y, "AP (전후, mm)", BOARD2_COLOR),
    ]:
        ax.set_facecolor("white")
        mask = ~np.isnan(y)
        ax.plot(t[mask], y[mask], color=color, linewidth=1.2)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.4)
        if analysis_window is not None:
            ax.axvspan(*analysis_window, alpha=0.1, color=TOTAL_COLOR,
                        label="분석 구간" if ax is ax1 else None)
        if mask.any():
            mean_y = float(y[mask].mean())
            ax.axhline(mean_y, color="#888", linewidth=0.6, linestyle="--")
    ax2.set_xlabel("시간 (s)")
    ax1.set_title("CoP 시계열 (ML · AP)", fontsize=12)
    if analysis_window is not None:
        ax1.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CMJ
# ─────────────────────────────────────────────────────────────────────────────

def make_cmj_force_time(t: np.ndarray, vgrf: np.ndarray,
                        bw_n: Optional[float] = None,
                        takeoff_t: Optional[float] = None,
                        landing_t: Optional[float] = None,
                        peak_t: Optional[float] = None,
                        width_in: float = 7.5,
                        height_in: float = 3.2) -> bytes:
    """CMJ vGRF over time with BW line, takeoff/landing markers, peak."""
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(t, vgrf, color=TOTAL_COLOR, linewidth=1.4, label="vGRF")
    if bw_n is not None:
        ax.axhline(bw_n, color="#888", linewidth=0.8, linestyle="--",
                    label=f"BW ({bw_n:.0f} N)")
    # Takeoff / landing markers
    if takeoff_t is not None:
        ax.axvline(takeoff_t, color=STATUS_OK, linewidth=1.5,
                    linestyle="-.", label="Takeoff")
    if landing_t is not None:
        ax.axvline(landing_t, color=STATUS_WARNING, linewidth=1.5,
                    linestyle="-.", label="Landing")
    if peak_t is not None:
        peak_idx = int(np.argmin(np.abs(t - peak_t)))
        ax.plot([peak_t], [vgrf[peak_idx]], marker="*",
                color="#B71C1C", markersize=14, label="Peak Force")
    # Flight phase shading
    if takeoff_t is not None and landing_t is not None:
        ax.axvspan(takeoff_t, landing_t, alpha=0.12, color=STATUS_OK,
                    label="체공")
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("vGRF (N)")
    ax.set_title("CMJ 힘-시간 곡선", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=10, ncol=2)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Squat precision (Phase S1d) — patent 2 §4
# ─────────────────────────────────────────────────────────────────────────────

def make_squat_cop_overlay(t_arr: np.ndarray,
                           cop_x: np.ndarray, cop_y: np.ndarray,
                           reps: list[dict],
                           width_in: float = 5.0,
                           height_in: float = 4.0) -> bytes:
    """Overlay all reps' CoP trails (centred per-rep on foot midpoint)
    with the mean trajectory highlighted. Helps visualise consistency
    — tight cluster = high CMC, scattered = low CMC.

    X axis = ML deviation (mm), Y = AP deviation (mm). Zero = rep-mean
    foot center.
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_facecolor("white")

    all_x = []
    all_y = []
    for rep in reps:
        ts = rep.get("t_start_s")
        te = rep.get("t_end_s")
        if ts is None or te is None:
            continue
        i_s = int(np.searchsorted(t_arr, ts))
        i_e = int(np.searchsorted(t_arr, te))
        if i_e <= i_s + 2:
            continue
        xs = cop_x[i_s:i_e + 1]
        ys = cop_y[i_s:i_e + 1]
        mask = (~np.isnan(xs)) & (~np.isnan(ys))
        if mask.sum() < 3:
            continue
        xv = xs[mask] - xs[mask].mean()
        yv = ys[mask] - ys[mask].mean()
        ax.plot(xv, yv, alpha=0.35, linewidth=1.0, color="#1565C0")
        all_x.append(xv)
        all_y.append(yv)

    # Mean trajectory (resample each to common length, average)
    if len(all_x) >= 2:
        N = 101
        x_stack = np.stack([np.interp(
            np.linspace(0, 1, N),
            np.linspace(0, 1, len(xv)), xv) for xv in all_x])
        y_stack = np.stack([np.interp(
            np.linspace(0, 1, N),
            np.linspace(0, 1, len(yv)), yv) for yv in all_y])
        ax.plot(x_stack.mean(0), y_stack.mean(0),
                color="#D32F2F", linewidth=2.2,
                label=f"평균 경로 ({len(all_x)} rep)")
        ax.legend(loc="best", fontsize=10, framealpha=0.9)

    ax.axhline(0, color="#BBB", linewidth=0.5)
    ax.axvline(0, color="#BBB", linewidth=0.5)
    ax.set_xlabel("ML 편차 (mm)")
    ax.set_ylabel("AP 편차 (mm)")
    ax.set_title("CoP 궤적 일관성 (발 중앙 기준)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def make_rfd_intervals_bar(rfd_per_rep: list[dict],
                            width_in: float = 6.0,
                            height_in: float = 3.0) -> bytes:
    """Bar chart of mean RFD across reps at 20/40/60/80/100 ms from
    concentric force onset. Each bar labeled with N/s magnitude.
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_facecolor("white")

    intervals = [20, 40, 60, 80, 100]
    means: list[float] = []
    for ms in intervals:
        vals: list[float] = []
        for r in rfd_per_rep:
            v = (r or {}).get(str(ms)) or (r or {}).get(ms)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        means.append(float(np.mean(vals)) if vals else 0.0)

    bars = ax.bar([f"0-{ms}" for ms in intervals], means,
                   color="#5F8A00", edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(means) * 0.02 if means else 0,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RFD (N/s)")
    ax.set_xlabel("힘 발현 onset 이후 구간 (ms)")
    ax.set_title("구간별 힘 생성 속도 (RFD)", fontsize=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Squat / Encoder — rep bars + rep markers
# ─────────────────────────────────────────────────────────────────────────────

def make_force_time_with_reps(t: np.ndarray, vgrf: np.ndarray,
                              reps: list[dict],
                              bw_n: Optional[float] = None,
                              width_in: float = 7.5,
                              height_in: float = 2.8) -> bytes:
    """vGRF over time with vertical markers at each rep's start/bottom/end."""
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(t, vgrf, color=TOTAL_COLOR, linewidth=1.2, label="vGRF")
    if bw_n is not None:
        ax.axhline(bw_n, color="#888", linewidth=0.8, linestyle="--",
                    label=f"BW ({bw_n:.0f} N)")
    for i, rep in enumerate(reps):
        ts = rep.get("t_start_s");  tb = rep.get("t_bottom_s")
        te = rep.get("t_end_s")
        if ts is not None and te is not None:
            ax.axvspan(ts, te, alpha=0.08, color=STATUS_OK)
            ax.axvline(ts, color=STATUS_OK,  linewidth=0.8, linestyle=":")
            if tb is not None:
                ax.axvline(tb, color=STATUS_WARNING, linewidth=0.8,
                            linestyle=":")
            ax.axvline(te, color=STATUS_OK,  linewidth=0.8, linestyle=":")
            if te > ts:
                ax.text((ts + te) / 2, ax.get_ylim()[1] * 0.92,
                        f"#{i+1}", ha="center", fontsize=8, color="#555")
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("vGRF (N)")
    ax.set_title(f"반복 {len(reps)} 회 힘-시간", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def make_rep_metric_bars(values: list[Optional[float]],
                         metric_label: str, unit: str = "",
                         colors: Optional[list[str]] = None,
                         width_in: float = 7.5,
                         height_in: float = 2.4) -> bytes:
    """Per-rep metric bars (one bar per rep). Missing values shown as gray."""
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_facecolor("white")
    x = np.arange(1, len(values) + 1)
    y = np.array([np.nan if v is None else float(v) for v in values])
    cols = colors if colors is not None \
        else [TOTAL_COLOR if not np.isnan(v) else "#BDBDBD" for v in y]
    bars = ax.bar(x, np.nan_to_num(y, nan=0.0), color=cols, edgecolor="white")
    for xi, vi in zip(x, y):
        if not np.isnan(vi):
            ax.text(xi, vi, f"{vi:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i}" for i in x])
    ax.set_ylabel(f"{metric_label}{' (' + unit + ')' if unit else ''}")
    ax.set_title(f"반복별 {metric_label}", fontsize=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# VBT zone colors (strength → speed)
VBT_ZONES = [
    (0.00, 0.45, "#E53935", "Max Strength"),   # red
    (0.45, 0.75, "#FB8C00", "Strength-Speed"),  # orange
    (0.75, 1.00, "#FDD835", "Power"),           # yellow
    (1.00, 1.30, "#43A047", "Speed-Strength"),  # green
    (1.30, 3.00, "#1E88E5", "Speed"),           # blue
]


def _zone_color(mcv: Optional[float]) -> str:
    if mcv is None or np.isnan(mcv):
        return "#BDBDBD"
    for lo, hi, c, _ in VBT_ZONES:
        if lo <= mcv < hi:
            return c
    return "#BDBDBD"


def make_vbt_velocity_bars(mcv_values: list[Optional[float]],
                           width_in: float = 7.5,
                           height_in: float = 2.8) -> bytes:
    """Per-rep velocity bars, colored by VBT zone."""
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_facecolor("white")
    x = np.arange(1, len(mcv_values) + 1)
    y = np.array([np.nan if v is None else float(v) for v in mcv_values])
    cols = [_zone_color(v) for v in y]
    ax.bar(x, np.nan_to_num(y, nan=0.0), color=cols, edgecolor="white")
    for xi, vi in zip(x, y):
        if not np.isnan(vi):
            ax.text(xi, vi, f"{vi:.2f}",
                    ha="center", va="bottom", fontsize=8)
    # Zone lines
    for lo, hi, c, label in VBT_ZONES[:-1]:
        ax.axhline(hi, color=c, linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i}" for i in x])
    ax.set_ylabel("MCV (m/s)")
    ax.set_title("반복별 평균 concentric velocity (VBT zone 색상)", fontsize=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Reaction
# ─────────────────────────────────────────────────────────────────────────────

def make_rt_histogram(rt_values_ms: list[float],
                      width_in: float = 6.5,
                      height_in: float = 2.6) -> bytes:
    """Distribution of reaction times, mean ± SD line."""
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_facecolor("white")
    arr = np.array([r for r in rt_values_ms
                    if r is not None and not np.isnan(r)], dtype=float)
    if len(arr) > 0:
        n_bins = max(4, min(20, int(len(arr) ** 0.7)))
        ax.hist(arr, bins=n_bins, color=TOTAL_COLOR, edgecolor="white")
        mu = float(arr.mean()); sd = float(arr.std())
        ax.axvline(mu, color="#2E7D32", linewidth=2,
                    label=f"평균 {mu:.0f} ms")
        ax.axvspan(mu - sd, mu + sd, alpha=0.15, color="#2E7D32",
                    label=f"±1 SD ({sd:.0f} ms)")
    ax.set_xlabel("RT (ms)")
    ax.set_ylabel("trials")
    ax.set_title("반응 시간 분포", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Proprio
# ─────────────────────────────────────────────────────────────────────────────

def make_proprio_scatter(trials: list[dict],
                         width_in: float = 5.0,
                         height_in: float = 4.8) -> bytes:
    """Target-vs-reproduction scatter in mm, with error vectors."""
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_facecolor("white")
    for t in trials:
        tgt = t.get("target_xy_mm");    rep = t.get("reproduction_xy_mm")
        if not tgt or not rep:
            continue
        tx, ty = float(tgt[0]), float(tgt[1])
        rx, ry = float(rep[0]), float(rep[1])
        ax.annotate("", xy=(rx, ry), xytext=(tx, ty),
                     arrowprops=dict(arrowstyle="->", color="#888",
                                     lw=0.8, alpha=0.7))
        ax.plot([tx], [ty], marker="x", color="#1976D2", markersize=10,
                 markeredgewidth=2)
        ax.plot([rx], [ry], marker="o", color="#E53935", markersize=8,
                 markerfacecolor="#E53935", alpha=0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("목표 (×) vs 재현 (●) — 오차 벡터", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Strength 3-lift (Phase V1-G)
# ─────────────────────────────────────────────────────────────────────────────
def make_strength_grade_band(thresholds_kg: dict, one_rm_kg: float,
                              exercise_label: str = "",
                              width_in: float = 7.5,
                              height_in: float = 1.6) -> bytes:
    """Horizontal grade-band visualisation for a strength_3lift result.

    Renders the five population zones (Beginner / Novice / Intermediate /
    Advanced / Elite) as adjacent coloured bars with the subject's 1RM
    marked above as a vertical line + value badge. Below-beginner is
    rendered as a hatched red prefix (warning zone), and the bar
    extends a bit past elite so subjects who exceed the elite threshold
    still get a visible marker.

    ``thresholds_kg`` must be the dict returned by
    ``strength_norms.grade_1rm`` (keys: beginner, novice, intermediate,
    advanced, elite — all in kg).
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    beg  = float(thresholds_kg["beginner"])
    nov  = float(thresholds_kg["novice"])
    inter = float(thresholds_kg["intermediate"])
    adv  = float(thresholds_kg["advanced"])
    eli  = float(thresholds_kg["elite"])
    val  = float(one_rm_kg)

    # X axis: from 60% of beginner (warning zone start) to max(elite,
    # subject 1RM) + 10% headroom so the marker is always visible.
    x_lo = 0.60 * beg
    x_hi = max(eli, val) * 1.05

    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")

    # Below-beginner warning zone (60-100% of beginner) — hatched red.
    ax.barh([0], [beg - x_lo], left=[x_lo],
            color="#FFCDD2", edgecolor="#C62828",
            hatch="///", linewidth=0.4, height=0.55)
    # Five zones, blue-green-yellow-orange-red-ish progression.
    zone_colors = ["#EF5350",  # Beginner → 5등급 위험 (red)
                   "#FFA726",  # Novice   → 4등급 나쁨 (orange)
                   "#FFEE58",  # Intermed → 3등급 보통 (yellow)
                   "#9CCC65",  # Advanced → 2등급 좋음 (light green)
                   "#26A69A"]  # Elite    → 1등급 엘리트 (teal)
    zone_labels = ["Beginner", "Novice", "Intermediate", "Advanced", "Elite"]
    zone_kr     = ["위험\n(5등급)", "나쁨\n(4등급)", "보통\n(3등급)",
                   "좋음\n(2등급)", "엘리트\n(1등급)"]
    edges = [beg, nov, inter, adv, eli]
    starts = [beg, nov, inter, adv]
    for i in range(5):
        start = starts[i] if i < 4 else adv
        end = edges[i + 1] if i < 4 else x_hi
        if i == 4:
            # Elite zone extends to x_hi
            start = adv
            end = x_hi
        else:
            start = starts[i]
            end = edges[i]
        # Skip negative-width edge cases (shouldn't happen with monotonic thresholds).
        if end <= start:
            continue
        ax.barh([0], [end - start], left=[start],
                color=zone_colors[i], edgecolor="white",
                linewidth=0.8, height=0.55)
        # Zone label (small, centered)
        midx = (start + end) / 2
        ax.text(midx, -0.55, zone_kr[i], ha="center", va="top",
                fontsize=8, color="#424242")
        # Threshold tick value above the bar
        if i < 4:
            ax.text(end, 0.35, f"{end:.0f}", ha="center", va="bottom",
                    fontsize=8, color="#666666")

    # Subject's 1RM marker — vertical line + value badge.
    ax.axvline(val, color="#212121", linewidth=2.2, ymin=0.0, ymax=0.95)
    badge_color = (
        "#26A69A" if val >= eli  else
        "#9CCC65" if val >= adv  else
        "#FBC02D" if val >= inter else
        "#FFA726" if val >= nov  else
        "#EF5350" if val >= beg  else
        "#B71C1C"
    )
    ax.scatter([val], [0.45], s=160, marker="v",
               color=badge_color, edgecolor="black", linewidth=1.2,
               zorder=5)
    ax.text(val, 0.85, f"{val:.1f} kg",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color="#212121",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor=badge_color,
                      linewidth=1.4))

    title_parts = ["1RM 등급"]
    if exercise_label:
        title_parts.append(f"({exercise_label})")
    ax.set_title("  ".join(title_parts), fontsize=11, loc="left",
                 color="#1976D2")
    ax.set_yticks([])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-1.0, 1.4)
    ax.set_xlabel("1RM (kg)", fontsize=9, color="#666666")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def make_recovery_set_bars(set_indices: list[int],
                            set_values: list[float],
                            variable_label: str = "평균 파워 (W)",
                            fi_pct: Optional[float] = None,
                            pds_pct: Optional[float] = None,
                            width_in: float = 7.5,
                            height_in: float = 2.4) -> bytes:
    """Per-set bar chart of the recovery primary variable (mean power).

    Used by the V2 ATP-PCr recovery subsection. Annotates each bar with
    its value and overlays FI/PDS reference text in the corner.
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt

    n = len(set_values)
    if n == 0:
        fig, ax = plt.subplots(figsize=(width_in, height_in),
                               facecolor="white")
        ax.text(0.5, 0.5, "회복 지표 데이터 없음", ha="center", va="center",
                fontsize=14, color="#999")
        ax.set_xticks([]); ax.set_yticks([])
        return _fig_to_png_bytes(fig)

    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    xs = list(range(1, n + 1))
    # Gradient: first set bright (best), tail bars muted.
    bar_colors = []
    for i, v in enumerate(set_values):
        # Per-bar tint relative to set 1
        if i == 0:
            bar_colors.append("#1976D2")
        else:
            ratio = (v / set_values[0]) if set_values[0] > 0 else 0
            ratio = max(0.0, min(1.0, ratio))
            # Blue at 1.0 → orange at 0.0
            bar_colors.append(
                "#1976D2" if ratio >= 0.85
                else "#42A5F5" if ratio >= 0.70
                else "#FFB74D" if ratio >= 0.55
                else "#EF5350"
            )

    bars = ax.bar(xs, set_values, color=bar_colors, edgecolor="white",
                  linewidth=1.2)
    for x, v in zip(xs, set_values):
        ax.text(x, v + max(set_values) * 0.02, f"{v:.0f}",
                ha="center", va="bottom", fontsize=10, color="#212121",
                fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"세트 {i + 1}" for i in set_indices],
                       fontsize=9)
    ax.set_ylabel(variable_label, fontsize=10)
    title = "세트별 수행 변화"
    if fi_pct is not None and pds_pct is not None:
        title += f"   (FI = {fi_pct:.1f}%, PDS = {pds_pct:.1f}%)"
    ax.set_title(title, fontsize=11, loc="left", color="#1976D2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def make_fiber_tendency_slider(tendency: float,
                                label_ko: str = "",
                                width_in: float = 6.5,
                                height_in: float = 1.4) -> bytes:
    """Endurance ↔ Power tendency horizontal slider (V2).

    ``tendency`` ∈ [-1, +1]:
        -1 = pure endurance (Type 1 / 지근 dominant)
         0 = balanced
        +1 = pure power     (Type 2 / 속근 dominant)
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt

    t = max(-1.0, min(1.0, float(tendency)))
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")

    # Gradient bar — green (endurance) → grey (mid) → orange (power).
    n_seg = 100
    x = np.linspace(-1, 1, n_seg)
    for i in range(n_seg - 1):
        v = (x[i] + x[i + 1]) / 2
        # Color interpolation
        if v < 0:
            # Green to grey
            r = 0.55 + 0.31 * (-v)
            g = 0.78 + 0.07 * (1 + v)
            b = 0.40 + 0.40 * (-v)
        else:
            # Grey to orange
            r = 0.86 + 0.14 * v
            g = 0.85 - 0.18 * v
            b = 0.80 - 0.65 * v
        ax.fill_betweenx([0, 1], x[i], x[i + 1],
                          color=(r, g, b), edgecolor="none")

    # Marker
    ax.axvline(t, color="#212121", linewidth=2.5, ymin=0.0, ymax=1.0)
    badge_color = ("#26A69A" if t < -0.3 else
                   "#FFA726" if t > 0.3 else "#9E9E9E")
    ax.scatter([t], [0.5], s=240, marker="o",
               color=badge_color, edgecolor="black", linewidth=1.5,
               zorder=5)

    if label_ko:
        ax.text(t, 1.25, label_ko,
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#212121",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor=badge_color,
                          linewidth=1.5))

    ax.text(-1.0, -0.35, "지구력\n(Type 1)", ha="left", va="top",
            fontsize=9, color="#26A69A", fontweight="bold")
    ax.text(0, -0.35, "균형형", ha="center", va="top",
            fontsize=9, color="#666666")
    ax.text(1.0, -0.35, "파워\n(Type 2)", ha="right", va="top",
            fontsize=9, color="#FFA726", fontweight="bold")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.0, 1.7)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    ax.set_title("근섬유 성향 (PDS 기반 추정)",
                 fontsize=10, loc="left", color="#1976D2")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Composite strength — body diagram + per-region bars (Phase V3)
# ─────────────────────────────────────────────────────────────────────────────
# Grade → fill colour. Mirrors the strength_3lift band chart so all
# strength visualisations share one palette.
_COMPOSITE_GRADE_COLORS: dict[int, str] = {
    1: "#26A69A",
    2: "#9CCC65",
    3: "#FBC02D",
    4: "#FFA726",
    5: "#EF5350",
    6: "#C62828",
    7: "#7F0000",
}
_UNMEASURED_COLOR = "#E0E0E0"


def make_body_strength_diagram(region_grades: dict,
                                region_one_rm: dict,
                                width_in: float = 5.5,
                                height_in: float = 7.0) -> bytes:
    """Frontal anatomical-zone diagram with per-region grade fill.

    ``region_grades`` is a dict ``{region: grade(1-7)}``. Missing
    regions are rendered grey with "—" label. ``region_one_rm`` carries
    the corresponding 1RM kg values for annotation; missing entries
    fall back to no annotation.

    Layout (matplotlib patches; not anatomically photorealistic but
    cleanly readable):

        ┌──────────┐
        │  head    │
        ├──────────┤
        │ shoulder │   shoulder zone
        │ chest    │   (3-row torso)
        │ back     │
        │whole_body│
        ├─┬──────┬─┤
        │L│      │R│   biceps + triceps (per arm)
        │ │      │ │
        ├─┴──────┴─┤
        │   legs   │
        └──────────┘
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    def _zone_color(region: str) -> str:
        g = region_grades.get(region)
        if g is None or not (1 <= int(g) <= 7):
            return _UNMEASURED_COLOR
        return _COMPOSITE_GRADE_COLORS.get(int(g), _UNMEASURED_COLOR)

    def _zone_text(region: str, region_label: str) -> str:
        g = region_grades.get(region)
        rm = region_one_rm.get(region)
        if g is None:
            return f"{region_label}\n—"
        rm_str = f"{rm:.0f} kg" if rm else "—"
        return f"{region_label}\n{g}등급 · {rm_str}"

    # Head — neutral grey (decorative).
    ax.add_patch(mpatches.Circle((5, 13), 0.85,
                                  facecolor="#BDBDBD", edgecolor="black",
                                  linewidth=1.0))

    # Shoulder band (single horizontal zone).
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.5, 11.0), 7.0, 1.2, boxstyle="round,pad=0.04",
        facecolor=_zone_color("shoulder"), edgecolor="black", linewidth=1.0))
    ax.text(5, 11.6, _zone_text("shoulder", "어깨"),
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Torso — chest above, back below (3 zones split vertically).
    ax.add_patch(mpatches.FancyBboxPatch(
        (2.0, 9.3), 6.0, 1.5, boxstyle="round,pad=0.04",
        facecolor=_zone_color("chest"), edgecolor="black", linewidth=1.0))
    ax.text(5, 10.05, _zone_text("chest", "가슴"),
            ha="center", va="center", fontsize=10, fontweight="bold")

    ax.add_patch(mpatches.FancyBboxPatch(
        (2.0, 7.6), 6.0, 1.5, boxstyle="round,pad=0.04",
        facecolor=_zone_color("back"), edgecolor="black", linewidth=1.0))
    ax.text(5, 8.35, _zone_text("back", "등"),
            ha="center", va="center", fontsize=10, fontweight="bold")

    ax.add_patch(mpatches.FancyBboxPatch(
        (2.0, 5.9), 6.0, 1.5, boxstyle="round,pad=0.04",
        facecolor=_zone_color("whole_body"), edgecolor="black", linewidth=1.0))
    ax.text(5, 6.65, _zone_text("whole_body", "전신"),
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Arms — biceps (front of upper arm) above triceps (back).
    # Left arm
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 8.6), 1.5, 1.4, boxstyle="round,pad=0.04",
        facecolor=_zone_color("biceps"), edgecolor="black", linewidth=1.0))
    ax.text(0.95, 9.3, _zone_text("biceps", "이두"),
            ha="center", va="center", fontsize=8)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 7.0), 1.5, 1.4, boxstyle="round,pad=0.04",
        facecolor=_zone_color("triceps"), edgecolor="black", linewidth=1.0))
    ax.text(0.95, 7.7, _zone_text("triceps", "삼두"),
            ha="center", va="center", fontsize=8)
    # Right arm
    ax.add_patch(mpatches.FancyBboxPatch(
        (8.3, 8.6), 1.5, 1.4, boxstyle="round,pad=0.04",
        facecolor=_zone_color("biceps"), edgecolor="black", linewidth=1.0))
    ax.text(9.05, 9.3, _zone_text("biceps", "이두"),
            ha="center", va="center", fontsize=8)
    ax.add_patch(mpatches.FancyBboxPatch(
        (8.3, 7.0), 1.5, 1.4, boxstyle="round,pad=0.04",
        facecolor=_zone_color("triceps"), edgecolor="black", linewidth=1.0))
    ax.text(9.05, 7.7, _zone_text("triceps", "삼두"),
            ha="center", va="center", fontsize=8)

    # Legs — split L/R for visual fullness, single zone semantically.
    ax.add_patch(mpatches.FancyBboxPatch(
        (2.0, 1.0), 2.5, 4.4, boxstyle="round,pad=0.04",
        facecolor=_zone_color("legs"), edgecolor="black", linewidth=1.0))
    ax.add_patch(mpatches.FancyBboxPatch(
        (5.5, 1.0), 2.5, 4.4, boxstyle="round,pad=0.04",
        facecolor=_zone_color("legs"), edgecolor="black", linewidth=1.0))
    ax.text(5, 3.2, _zone_text("legs", "하체"),
            ha="center", va="center", fontsize=11, fontweight="bold")

    return _fig_to_png_bytes(fig)


def make_strength_per_region_bars(regions: list[dict],
                                    width_in: float = 7.5,
                                    height_in: float = 3.5) -> bytes:
    """Horizontal bar chart, one bar per measured region.

    ``regions`` is a list of dicts with keys: ``region_label``,
    ``one_rm_kg``, ``grade``. Missing regions can be passed with
    ``measured=False`` — they render at 0 with grey colour.

    Bars use the composite grade colour palette so the diagram and
    bars are visually consistent.
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt

    if not regions:
        fig, ax = plt.subplots(figsize=(width_in, height_in),
                               facecolor="white")
        ax.text(0.5, 0.5, "측정 데이터 없음", ha="center", va="center",
                fontsize=14, color="#999")
        ax.set_xticks([]); ax.set_yticks([])
        return _fig_to_png_bytes(fig)

    labels  = [r["region_label"] for r in regions]
    values  = [float(r.get("one_rm_kg") or 0.0) for r in regions]
    grades  = [int(r.get("grade") or 0) for r in regions]
    colors  = [_COMPOSITE_GRADE_COLORS.get(g, _UNMEASURED_COLOR)
               for g in grades]

    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    ys = list(range(len(labels)))
    bars = ax.barh(ys, values, color=colors, edgecolor="white",
                    linewidth=1.2)
    for y, v, g, lbl in zip(ys, values, grades, labels):
        rm_str = f"{v:.0f} kg" if v > 0 else "—"
        gr_str = f"{g}등급" if g else "(미측정)"
        ax.text(v + max(values + [1]) * 0.02, y,
                f"{rm_str}   {gr_str}",
                va="center", fontsize=10, color="#212121",
                fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("추정 1RM (kg)", fontsize=10)
    ax.set_title("부위별 1RM × 등급", fontsize=11, loc="left",
                 color="#1976D2")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def make_strength_per_set_bars(per_set: list[dict],
                                width_in: float = 7.5,
                                height_in: float = 2.5) -> bytes:
    """Per-set vertical bar chart of estimated 1RM with reliability tint.

    ``per_set`` is the list of StrengthSetResult dicts (after to_dict).
    Warmup sets are rendered with reduced opacity + diagonal hatch so
    they're visually distinct from working sets.
    """
    setup_korean_fonts()
    import matplotlib.pyplot as plt
    if not per_set:
        # Empty placeholder
        fig, ax = plt.subplots(figsize=(width_in, height_in),
                               facecolor="white")
        ax.text(0.5, 0.5, "측정 데이터 없음", ha="center", va="center",
                fontsize=14, color="#999")
        ax.set_xticks([]); ax.set_yticks([])
        return _fig_to_png_bytes(fig)

    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    xs = list(range(1, len(per_set) + 1))
    rel_color = {
        "excellent":  "#1B5E20",
        "high":       "#2E7D32",
        "medium":     "#558B2F",
        "low":        "#F9A825",
        "unreliable": "#C62828",
    }
    bar_colors = [rel_color.get(s.get("reliability", "unreliable"), "#9E9E9E")
                  for s in per_set]
    one_rms = [float(s.get("one_rm_kg") or 0.0) for s in per_set]
    warmups = [bool(s.get("warmup", False)) for s in per_set]
    bars = ax.bar(xs, one_rms, color=bar_colors, edgecolor="white",
                  linewidth=1.2)
    # Tint warmup bars (lower alpha + hatch).
    for b, wu in zip(bars, warmups):
        if wu:
            b.set_alpha(0.45)
            b.set_hatch("///")
    # Annotate each bar with reps × load.
    for x, s, val in zip(xs, per_set, one_rms):
        reps = s.get("n_reps", 0)
        load = s.get("load_kg", 0)
        ax.text(x, val + max(one_rms) * 0.02 if val > 0 else 0.5,
                f"{reps}회\n@ {load:.0f}kg",
                ha="center", va="bottom", fontsize=9, color="#424242")

    ax.set_xticks(xs)
    ax.set_xticklabels([
        f"세트 {i}" + (" (워밍업)" if w else "")
        for i, w in zip(xs, warmups)
    ], fontsize=9)
    ax.set_ylabel("추정 1RM (kg)", fontsize=10)
    ax.set_title("세트별 1RM 추정", fontsize=11, loc="left",
                 color="#1976D2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)
