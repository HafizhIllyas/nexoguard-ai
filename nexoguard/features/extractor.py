"""
nexoguard.features.extractor
Feature extraction engine complying with PROJECT.md Interface Contracts.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from nexoguard.signal.preprocessor import clean_fhr_signal, calculate_signal_loss_pct
from nexoguard.signal.windowing import slice_signal_windows
from nexoguard.features.baseline import calculate_nexoprima_ema
from nexoguard.features.variability import calculate_stv_ms, calculate_ltv_bpm
from nexoguard.features.decelerations import analyze_morphometry_and_synchrony

CONTRACT_FEATURE_COLS = [
    "window_id",
    "baseline_bpm",
    "stv_ms",
    "ltv_bpm",
    "accel_count",
    "decel_count",
    "decel_max_depth_bpm",
    "decel_total_area",
    "contraction_count",
    "decel_uc_lag_sec",
    "signal_loss_pct",
    "mhr_fhr_cross_corr"
]


def check_mhr_confusion(fhr_win: np.ndarray, mhr_win: Optional[np.ndarray] = None) -> float:
    """
    Computes Pearson cross-correlation between FHR and maternal heart rate channel.
    Returns 0.15 baseline default when maternal channel is not supplied.
    """
    if mhr_win is None or len(mhr_win) == 0:
        return 0.15
    f = np.nan_to_num(np.asarray(fhr_win, dtype=float), nan=130.0)
    m = np.nan_to_num(np.asarray(mhr_win, dtype=float), nan=80.0)
    if np.std(f) == 0 or np.std(m) == 0:
        return 0.0
    corr = float(np.corrcoef(f, m)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def extract_features_from_window(
    fhr_win: np.ndarray,
    uc_win: np.ndarray,
    win_id: str = "W00",
    patient_id: str = "Unknown",
    fs: float = 4.0,
    mhr_win: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Extracts all clinical features and quality indices from a single 10-minute window.
    """
    loss_pct = calculate_signal_loss_pct(fhr_win)

    # Flag windows with excessive telemetry loss
    is_uninterpretable = loss_pct >= 80.0
    quality_flag = "UNINTERPRETABLE" if is_uninterpretable else "ADEQUATE"

    fhr_clean = clean_fhr_signal(fhr_win, fs=fs)
    uc_clean = np.nan_to_num(np.array(uc_win, dtype=float), nan=10.0)

    ema_baseline = calculate_nexoprima_ema(fhr_clean)
    baseline_bpm = float(np.mean(ema_baseline)) if len(ema_baseline) > 0 else 135.0

    stv_ms = calculate_stv_ms(fhr_clean, fs=fs)
    ltv_bpm = calculate_ltv_bpm(fhr_clean, fs=fs)

    morph = analyze_morphometry_and_synchrony(fhr_clean, ema_baseline, uc_clean, fs=fs)
    mhr_corr = check_mhr_confusion(fhr_clean, mhr_win)

    return {
        "window_id": win_id,
        "patient_id": str(patient_id),
        "baseline_bpm": round(baseline_bpm, 2),
        "stv_ms": round(stv_ms, 2),
        "ltv_bpm": round(ltv_bpm, 2),
        "accel_count": morph["accel_count"],
        "decel_count": morph["decel_count"],
        "decel_max_depth_bpm": morph["decel_max_depth_bpm"],
        "decel_total_area": morph["decel_total_area"],
        "contraction_count": morph["contraction_count"],
        "decel_uc_lag_sec": morph["decel_uc_lag_sec"],
        "signal_loss_pct": round(loss_pct, 2),
        "mhr_fhr_cross_corr": round(mhr_corr, 2),
        "quality_flag": quality_flag
    }


def extract_features_from_signal(
    fhr_raw: np.ndarray,
    uc_raw: np.ndarray,
    fs: float = 4.0,
    window_len_sec: int = 600,
    step_sec: int = 300,
    patient_id: str = "Unknown"
) -> pd.DataFrame:
    """
    Public Shared Contract (PROJECT.md:77-88):
    extract_features_from_signal(fhr_raw: np.ndarray, uc_raw: np.ndarray,
                                fs: float = 4.0, window_len_sec: int = 600,
                                step_sec: int = 300) -> pd.DataFrame

    Inputs:
      - fhr_raw: 1D array of float/int, raw FHR in bpm
      - uc_raw: 1D array of float/int, raw UC in relative units
      - fs: sampling rate in Hz (default 4.0)

    Outputs:
      - pd.DataFrame where each row represents a window with columns:
        ['window_id', 'baseline_bpm', 'stv_ms', 'ltv_bpm', 'accel_count', 'decel_count',
         'decel_max_depth_bpm', 'decel_total_area', 'contraction_count', 'decel_uc_lag_sec',
         'signal_loss_pct', 'mhr_fhr_cross_corr']
    """
    fhr_arr = np.asarray(fhr_raw, dtype=float)
    uc_arr = np.asarray(uc_raw, dtype=float)

    if len(fhr_arr) == 0 or len(uc_arr) == 0:
        raise ValueError("Invalid CTG signal: telemetry buffer is empty. Minimum window duration is 600 seconds.")

    min_samples = int(fs * window_len_sec)
    if len(fhr_arr) < min_samples or len(uc_arr) < min_samples:
        raise ValueError(
            f"Invalid CTG signal duration: recording has {len(fhr_arr)} samples ({len(fhr_arr)/fs:.1f} s), "
            f"which is less than the required 10-minute window ({min_samples} samples)."
        )

    windows = slice_signal_windows(
        fhr=fhr_arr,
        uc=uc_arr,
        fs=fs,
        window_sec=window_len_sec,
        step_sec=step_sec,
        patient_id=patient_id
    )

    feature_rows = []
    for win in windows:
        row = extract_features_from_window(
            fhr_win=win.fhr,
            uc_win=win.uc,
            win_id=win.window_id,
            patient_id=win.patient_id,
            fs=fs
        )
        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)

    # Ensure required columns are present in contract order
    for col in CONTRACT_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df
