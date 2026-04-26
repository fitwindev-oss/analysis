"""
Extract the compact key-metrics payload cached in the session_metrics DB.

For each test type, only a handful of headline metrics are pulled from the
full result.json so trend queries stay fast and the DB row stays small.
Adding a new metric here = it becomes available as a history line in
the report's HistorySection.
"""
from __future__ import annotations

from typing import Any


# Ordered list of (key, label, unit) per test type.
# The ORDER matters — the top ~3 are shown as default history trend lines;
# others are queryable via the UI's metric selector.
KEY_METRICS: dict[str, list[tuple[str, str, str]]] = {
    "balance_eo": [
        ("mean_velocity_mm_s",  "평균 이동 속도",       "mm/s"),
        ("ellipse95_area_mm2",  "95% 타원 면적",        "mm²"),
        ("path_length_mm",      "경로 길이",            "mm"),
        ("rms_ml_mm",           "RMS ML",               "mm"),
        ("rms_ap_mm",           "RMS AP",               "mm"),
    ],
    "balance_ec": [
        ("mean_velocity_mm_s",  "평균 이동 속도",       "mm/s"),
        ("ellipse95_area_mm2",  "95% 타원 면적",        "mm²"),
        ("path_length_mm",      "경로 길이",            "mm"),
        ("rms_ml_mm",           "RMS ML",               "mm"),
        ("rms_ap_mm",           "RMS AP",               "mm"),
    ],
    "cmj": [
        ("jump_height_m_impulse", "점프 높이 (임펄스)", "m"),
        ("peak_force_bw",         "최고 Force (BW)",    "×BW"),
        ("peak_power_w",          "Peak Power",         "W"),
        ("peak_rfd_n_s",          "Peak RFD",           "N/s"),
        ("flight_time_s",         "체공 시간",          "s"),
        ("takeoff_velocity_m_s",  "이륙 속도",          "m/s"),
    ],
    "squat": [
        ("mean_peak_vgrf_bw",        "평균 peak vGRF",      "×BW"),
        ("mean_wba_pct",             "평균 WBA",            "%"),
        ("n_reps",                   "반복 횟수",           ""),
        # Phase S1d precision metrics (patent 2 §4)
        ("cmc_ap",                   "CMC (AP)",            ""),
        ("cmc_ml",                   "CMC (ML)",            ""),
        ("mean_rmse_ap_mm",          "평균 RMSE (AP)",      "mm"),
        ("mean_rmse_ml_mm",          "평균 RMSE (ML)",      "mm"),
        ("mean_tempo_ratio",         "평균 Tempo (E:C)",    ""),
        ("mean_impulse_asym_ecc_pct","하강 좌우 비대칭",    "%"),
        ("mean_impulse_asym_con_pct","상승 좌우 비대칭",    "%"),
        ("mean_peak_rfd_n_s",        "평균 peak RFD",       "N/s"),
        ("mean_vrt_ms",              "평균 VRT",            "ms"),
    ],
    "overhead_squat": [
        ("mean_peak_vgrf_bw",        "평균 peak vGRF",      "×BW"),
        ("mean_wba_pct",             "평균 WBA",            "%"),
        ("n_reps",                   "반복 횟수",           ""),
        ("cmc_ap",                   "CMC (AP)",            ""),
        ("cmc_ml",                   "CMC (ML)",            ""),
        ("mean_rmse_ap_mm",          "평균 RMSE (AP)",      "mm"),
        ("mean_rmse_ml_mm",          "평균 RMSE (ML)",      "mm"),
        ("mean_tempo_ratio",         "평균 Tempo (E:C)",    ""),
        ("mean_impulse_asym_ecc_pct","하강 좌우 비대칭",    "%"),
        ("mean_impulse_asym_con_pct","상승 좌우 비대칭",    "%"),
        ("mean_peak_rfd_n_s",        "평균 peak RFD",       "N/s"),
        ("mean_vrt_ms",              "평균 VRT",            "ms"),
    ],
    "encoder": [
        ("mean_mcv_m_s",        "평균 MCV",             "m/s"),
        ("mean_rom_mm",         "평균 ROM",             "mm"),
        ("n_reps",              "반복 횟수",            ""),
    ],
    "reaction": [
        ("mean_rt_ms",                    "평균 RT",          "ms"),
        ("mean_peak_displacement_mm",     "평균 peak 변위",   "mm"),
        ("mean_recovery_time_s",          "평균 회복 시간",    "s"),
        ("n_trials",                      "trial 수",          ""),
    ],
    "proprio": [
        ("mean_absolute_error_mm",  "평균 절대 오차",   "mm"),
        ("constant_error_mm",       "일정 오차 (CE)",  "mm"),
        ("variable_error_mm",       "가변 오차 (VE)",  "mm"),
    ],
    "free_exercise": [
        ("mean_con_vel_m_s",    "평균 MCV",         "m/s"),
        ("peak_con_vel_m_s",    "peak 속도",        "m/s"),
        ("mean_rom_mm",         "평균 ROM",         "mm"),
        ("mean_con_power_w",    "평균 power",       "W"),
        ("peak_con_power_w",    "peak power",       "W"),
        ("n_reps",              "반복 횟수",        ""),
        ("load_kg",             "하중",             "kg"),
    ],
}


def extract_key_metrics(test_type: str, result: dict) -> dict:
    """Return a compact dict of metric_key → numeric value.

    Only scalar int/float entries are kept. Missing keys are omitted
    (not stored as None) so the DB payload stays minimal.
    """
    if not result:
        return {}
    keys = [k for (k, _, _) in KEY_METRICS.get(test_type, [])]
    out: dict[str, float] = {}
    for k in keys:
        v = result.get(k)
        # Derived aggregates for list-bearing tests
        if v is None:
            if k == "n_reps":
                v = len(result.get("reps") or [])
            elif k == "n_trials":
                v = len(result.get("trials") or [])
            elif k == "mean_mcv_m_s" and result.get("reps"):
                vals = [r.get("mean_con_vel_m_s") for r in result["reps"]
                        if r.get("mean_con_vel_m_s") is not None]
                v = (sum(vals) / len(vals)) if vals else None
            elif k == "mean_rom_mm" and result.get("reps"):
                vals = [r.get("rom_mm") for r in result["reps"]
                        if r.get("rom_mm") is not None]
                v = (sum(vals) / len(vals)) if vals else None
            elif k == "mean_recovery_time_s" and result.get("trials"):
                vals = [t.get("recovery_time_s") for t in result["trials"]
                        if t.get("recovery_time_s") is not None]
                v = (sum(vals) / len(vals)) if vals else None
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def key_metric_labels(test_type: str) -> list[tuple[str, str, str]]:
    """Public accessor — (key, label, unit) tuples, ordered."""
    return list(KEY_METRICS.get(test_type, []))


def variant_from_meta(test_type: str, meta: dict) -> str | None:
    """Return the variant string for session_metrics.variant column."""
    if not meta:
        return None
    if test_type in ("balance_eo", "balance_ec"):
        return meta.get("stance") or "two"
    if test_type == "reaction":
        return meta.get("reaction_trigger")
    if test_type == "free_exercise":
        # Group history by exercise name so "back squat @ 60 kg" trends
        # stay separate from "push-up bodyweight" trends.
        name = meta.get("exercise_name")
        return str(name).strip().lower() if name else None
    return None
