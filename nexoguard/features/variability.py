"""
nexoguard.features.variability
Short-Term Variability (STV in ms) and Long-Term Variability (LTV in bpm) computation.
"""

import numpy as np


def calculate_stv_ms(
    fhr_clean: np.ndarray,
    fs: float = 4.0,
    method: str = "beat_diff"
) -> float:
    """
    Calculates Short-Term Variability (STV in ms).
    - 'beat_diff': Mean absolute difference of pulse intervals across consecutive beats.
      Strictly compatible with features.csv benchmark values.
    - 'dawes_redman': 1/16-minute epoch mean pulse interval differences (classic cCTG).
    """
    fhr = np.clip(np.asarray(fhr_clean, dtype=float), 50.0, 240.0)
    if len(fhr) < 2:
        return 0.0

    intervals_ms = 60000.0 / fhr

    if method == "beat_diff":
        diffs = np.abs(np.diff(intervals_ms))
        return float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    # Classic Dawes-Redman: 1/16 minute (3.75 s) epochs = 15 samples at 4 Hz
    epoch_len = max(1, int(fs * 3.75))
    num_epochs = len(intervals_ms) // epoch_len
    if num_epochs < 2:
        diffs = np.abs(np.diff(intervals_ms))
        return float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    epoch_means = [
        np.mean(intervals_ms[i * epoch_len : (i + 1) * epoch_len])
        for i in range(num_epochs)
    ]
    epoch_diffs = np.abs(np.diff(epoch_means))
    return float(np.mean(epoch_diffs)) if len(epoch_diffs) > 0 else 0.0


def calculate_ltv_bpm(
    fhr_clean: np.ndarray,
    fs: float = 4.0,
    block_sec: float = 60.0
) -> float:
    """
    Calculates Long-Term Variability (LTV in bpm) as mean peak-to-peak amplitude range
    across 1-minute blocks (FIGO 2015 criteria: normal 5-25 bpm).
    """
    fhr = np.asarray(fhr_clean, dtype=float)
    if len(fhr) == 0:
        return 0.0

    block_size = int(fs * block_sec)
    if len(fhr) < block_size:
        return float(np.ptp(fhr))

    num_blocks = len(fhr) // block_size
    ltv_blocks = [
        float(np.ptp(fhr[i * block_size : (i + 1) * block_size]))
        for i in range(num_blocks)
    ]
    return float(np.mean(ltv_blocks)) if ltv_blocks else float(np.ptp(fhr))
