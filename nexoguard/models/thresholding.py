"""
nexoguard.models.thresholding
Calibrated decision thresholding and continuous composite risk scoring.
"""

from typing import Tuple
import numpy as np


def apply_calibrated_thresholds(
    probs: np.ndarray,
    tau_cat3: float = 0.25,
    tau_cat2: float = 0.40,
    tau_cat3_susp: float = 0.10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies calibrated clinical decision thresholding to multi-class probabilities:
    - If P(Cat III) >= tau_cat3 (0.25): Triage as Category III (Pathological)
    - Else if P(Cat II) >= tau_cat2 (0.40) or P(Cat III) >= tau_cat3_susp (0.10): Triage as Category II
    - Else: Triage as Category I (Normal)

    Returns:
    - pred_classes: 1D array of integers (0, 1, 2)
    - risk_scores: 1D array of continuous risk scores in [0.0, 1.0]
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    n_samples = probs.shape[0]
    pred_classes = np.zeros(n_samples, dtype=int)
    risk_scores = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        # probs has columns [p0, p1, p2]
        p1 = probs[i, 0] if probs.shape[1] > 0 else 0.0
        p2 = probs[i, 1] if probs.shape[1] > 1 else 0.0
        p3 = probs[i, 2] if probs.shape[1] > 2 else 0.0

        # Composite continuous risk score: P(Cat III) + 0.35 * P(Cat II)
        risk = min(1.0, float(p3 + 0.35 * p2))
        risk_scores[i] = risk

        if p3 >= tau_cat3:
            pred_classes[i] = 2  # Category III
        elif (p2 >= tau_cat2) or (p3 >= tau_cat3_susp):
            pred_classes[i] = 1  # Category II
        else:
            pred_classes[i] = 0  # Category I

    return pred_classes, risk_scores
