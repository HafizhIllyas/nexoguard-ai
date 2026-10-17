"""
nexoguard.features.baseline
NexoPrima Exponential Moving Average (EMA) dynamic baseline detector with deceleration dampening.
"""

import numpy as np


def calculate_nexoprima_ema(
    fhr_clean: np.ndarray,
    alpha: float = 0.85,
    decel_dampen: bool = False
) -> np.ndarray:
    """
    Computes dynamic baseline via recursive Exponential Moving Average:
    BL_0 = median(FHR_{0..min(N, 240)})
    BL_t = alpha * BL_{t-1} + (1 - alpha) * FHR_t

    With deceleration dampening:
    If FHR drops below BL_{t-1} - 10 bpm, alpha is increased to 0.98 to prevent
    the baseline from tracking into deep deceleration troughs.
    """
    fhr = np.asarray(fhr_clean, dtype=float)
    n = len(fhr)
    if n == 0:
        return np.array([], dtype=float)

    baseline = np.zeros(n, dtype=float)
    init_window = min(240, n)
    baseline[0] = float(np.median(fhr[:init_window]))

    for t in range(1, n):
        curr_fhr = fhr[t]
        prev_bl = baseline[t - 1]

        if decel_dampen and (curr_fhr < prev_bl - 10.0):
            # Dampen baseline fall during deep deceleration
            a = 0.98
        else:
            a = alpha

        baseline[t] = a * prev_bl + (1.0 - a) * curr_fhr

    return baseline
