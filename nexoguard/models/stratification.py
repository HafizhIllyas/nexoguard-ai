"""
nexoguard.models.stratification
Patient-level cross-validation generator ensuring ZERO data leakage.
"""

from typing import Generator, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def get_patient_stratified_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    patient_col: str = "patient_id",
    label_col: str = "label",
    seed: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generates patient-stratified train/validation splits ensuring ZERO window leakage.
    Each patient's windows belong strictly to train or val, never both.
    """
    if patient_col not in df.columns:
        raise ValueError(f"Patient grouping column '{patient_col}' not found in DataFrame.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    groups = df[patient_col].to_numpy()
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col])

    num_groups = len(np.unique(groups))
    effective_splits = min(n_splits, num_groups)

    sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    return sgkf.split(X, y, groups)
