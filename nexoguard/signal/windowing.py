"""
nexoguard.signal.windowing
Standard 10-minute sliding window generator with 50% overlap for intrapartum CTG analysis.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class SignalWindow:
    window_id: str
    patient_id: str
    win_idx: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    fhr: np.ndarray
    uc: np.ndarray


def slice_signal_windows(
    fhr: np.ndarray,
    uc: np.ndarray,
    fs: float = 4.0,
    window_sec: int = 600,
    step_sec: int = 300,
    patient_id: str = "Unknown"
) -> List[SignalWindow]:
    """
    Slices multi-channel time series into 10-minute sliding windows (2400 samples at 4 Hz)
    with 5-minute step size (1200 samples = 50% overlap).
    """
    win_len = int(fs * window_sec)   # 2400 samples
    step_len = int(fs * step_sec)    # 1200 samples
    total_samples = min(len(fhr), len(uc))

    windows: List[SignalWindow] = []
    win_idx = 0

    if total_samples < win_len:
        # If signal is shorter than window length, return empty list or handle gracefully
        return windows

    for start in range(0, total_samples - win_len + 1, step_len):
        end = start + win_len
        w_id = f"{patient_id}_W{win_idx:02d}"
        windows.append(
            SignalWindow(
                window_id=w_id,
                patient_id=str(patient_id),
                win_idx=win_idx,
                start_sample=start,
                end_sample=end,
                start_sec=float(start / fs),
                end_sec=float(end / fs),
                fhr=np.asarray(fhr[start:end], dtype=float),
                uc=np.asarray(uc[start:end], dtype=float)
            )
        )
        win_idx += 1

    return windows
