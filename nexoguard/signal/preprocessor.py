"""
nexoguard.signal.preprocessor
Physiological artifact rejection, dropout tracking, gap interpolation, and digital filtering for FHR.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter


def calculate_signal_loss_pct(
    fhr_raw: np.ndarray,
    min_bpm: float = 50.0,
    max_bpm: float = 240.0
) -> float:
    """
    Calculates percentage of Doppler dropout / physiologically invalid samples.
    Samples with values outside [min_bpm, max_bpm] or <= 0.0 or NaN are treated as signal loss.
    """
    arr = np.asarray(fhr_raw, dtype=float)
    if len(arr) == 0:
        return 100.0
    invalid_mask = np.isnan(arr) | (arr < min_bpm) | (arr > max_bpm) | (arr <= 0.0)
    return float(np.sum(invalid_mask) / len(arr) * 100.0)


def clean_fhr_signal(
    fhr_raw: np.ndarray,
    fs: float = 4.0,
    min_bpm: float = 50.0,
    max_bpm: float = 240.0,
    max_interp_sec: float = 15.0,
    default_median: float = 135.0
) -> np.ndarray:
    """
    Multi-stage DSP cleaning pipeline for raw 4 Hz FHR telemetry:
    1. Replaces non-physiological outliers (< 50 or > 240 bpm, <= 0) with NaN.
    2. Interpolates short dropouts (<= 15 seconds / 60 samples).
    3. Imputes residual gaps with median baseline (or default_median).
    4. Applies 3-point median filter to remove impulse spikes without blunting deceleration troughs.
    5. Applies Savitzky-Golay polynomial smoothing (window 11, polyorder 2).
    """
    fhr = np.array(fhr_raw, dtype=float).copy()
    if len(fhr) == 0:
        raise ValueError("Cannot clean empty FHR signal: Fetal telemetry buffer is empty.")

    # 1. Outlier rejection
    invalid_mask = np.isnan(fhr) | (fhr < min_bpm) | (fhr > max_bpm) | (fhr <= 0.0)
    fhr[invalid_mask] = np.nan

    # 2. Linear gap interpolation for short dropouts
    max_gap_samples = int(fs * max_interp_sec)
    s = pd.Series(fhr)
    fhr_interp = s.interpolate(method="linear", limit=max_gap_samples).to_numpy()

    # 3. Residual imputation
    valid_samples = fhr_interp[~np.isnan(fhr_interp)]
    fill_val = float(np.median(valid_samples)) if len(valid_samples) > 0 else default_median
    fhr_filled = np.nan_to_num(fhr_interp, nan=fill_val)

    # 4. 3-point median filter
    if len(fhr_filled) >= 3:
        try:
            fhr_med = medfilt(fhr_filled, kernel_size=3)
        except Exception:
            fhr_med = fhr_filled
    else:
        fhr_med = fhr_filled

    # 5. Savitzky-Golay smoothing
    savgol_win = 11
    if len(fhr_med) >= savgol_win:
        try:
            fhr_clean = savgol_filter(fhr_med, window_length=savgol_win, polyorder=2)
        except Exception:
            fhr_clean = fhr_med
    else:
        fhr_clean = fhr_med

    # Physiological clamp on final smoothed signal
    return np.clip(fhr_clean, min_bpm, max_bpm)
