"""
nexoguard.features.decelerations
Deceleration, acceleration, uterine contraction morphometry, and synchrony lag Δt.
"""

from typing import Any, Dict
import numpy as np


def analyze_morphometry_and_synchrony(
    fhr_clean: np.ndarray,
    baseline: np.ndarray,
    uc_clean: np.ndarray,
    fs: float = 4.0
) -> Dict[str, Any]:
    """
    Extracts acceleration count, deceleration count, deceleration depth, area,
    uterine contraction count, and contraction-to-deceleration synchrony lag Δt:
    - Acceleration: FHR >= baseline + 15 bpm for >= 15 s
    - Deceleration: FHR <= baseline - 15 bpm for >= 15 s
    - Contraction: UC > max(35, mean + std) for >= 30 s
    - Synchrony Lag Δt = (t_nadir(FHR) - t_peak(UC)) / fs (in seconds)
    """
    fhr = np.asarray(fhr_clean, dtype=float)
    bl = np.asarray(baseline, dtype=float)
    uc = np.asarray(uc_clean, dtype=float)

    if len(fhr) == 0 or len(bl) == 0:
        return {
            "accel_count": 0,
            "decel_count": 0,
            "decel_max_depth_bpm": 0.0,
            "decel_total_area": 0.0,
            "contraction_count": 0,
            "decel_uc_lag_sec": 0.0,
        }

    diff = fhr - bl
    is_accel = diff >= 15.0
    is_decel = diff <= -15.0

    min_event_samples = int(fs * 15)  # 15 seconds (60 samples at 4 Hz)
    accel_count = int(np.sum(is_accel) // min_event_samples)
    decel_count = int(np.sum(is_decel) // min_event_samples)

    decel_depth = float(np.max(np.abs(diff[is_decel]))) if np.any(is_decel) else 0.0
    decel_area = float(np.sum(np.abs(diff[is_decel]))) if np.any(is_decel) else 0.0

    # Contraction detection: UC > max(35, mean + std)
    if len(uc) > 0:
        uc_mean = float(np.mean(uc))
        uc_std = float(np.std(uc))
        uc_thresh = max(35.0, uc_mean + uc_std)
        is_contracting = uc > uc_thresh
        contraction_count = int(np.sum(is_contracting) // int(fs * 30))
    else:
        is_contracting = np.array([], dtype=bool)
        contraction_count = 0

    # Synchrony Lag Δt: time of Decel nadir minus time of UC peak
    decel_uc_lag = 0.0
    if np.any(is_decel) and np.any(is_contracting):
        uc_peak_idx = int(np.argmax(uc))
        fhr_nadir_idx = int(np.argmin(fhr))
        decel_uc_lag = float((fhr_nadir_idx - uc_peak_idx) / fs)

    return {
        "accel_count": accel_count,
        "decel_count": decel_count,
        "decel_max_depth_bpm": round(decel_depth, 2),
        "decel_total_area": round(decel_area, 2),
        "contraction_count": contraction_count,
        "decel_uc_lag_sec": round(decel_uc_lag, 2)
    }
