"""
nexoguard.models.train
Model training and optimization with asymmetric cost weighting and out-of-fold evaluation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from nexoguard.models.stratification import get_patient_stratified_folds

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

FEATURE_COLS = [
    "baseline_bpm", "stv_ms", "ltv_bpm", "accel_count", "decel_count",
    "decel_max_depth_bpm", "decel_total_area", "contraction_count",
    "decel_uc_lag_sec", "mhr_fhr_cross_corr", "signal_loss_pct"
]


def compute_asymmetric_sample_weights(
    y: np.ndarray,
    w_cat1: float = 1.0,
    w_cat2: float = 2.0,
    w_cat3: float = 10.0
) -> np.ndarray:
    """
    Generates sample weights penalizing Category III false negatives heavily.
    """
    weight_map = {0: w_cat1, 1: w_cat2, 2: w_cat3}
    return np.array([weight_map.get(int(lbl), 1.0) for lbl in y], dtype=float)


def train_model(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains multi-class XGBoost model with 5-fold StratifiedGroupKFold out-of-fold
    evaluation, fits final production model, and fits TreeSHAP explainer.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X = df[feature_cols]
    y = df["label"].to_numpy().astype(int)
    sample_weights = compute_asymmetric_sample_weights(y)

    oof_probs = np.zeros((len(df), 3))
    oof_preds = np.zeros(len(df), dtype=int)

    splits = get_patient_stratified_folds(df, n_splits=n_splits, seed=random_state)

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        w_train = sample_weights[train_idx]
        X_val = X.iloc[val_idx]

        clf = XGBClassifier(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
            random_state=random_state + fold,
            eval_metric="mlogloss"
        )
        clf.fit(X_train, y_train, sample_weight=w_train)

        probs = clf.predict_proba(X_val)
        oof_probs[val_idx] = probs
        oof_preds[val_idx] = np.argmax(probs, axis=1)

    # Train final production model on full cohort
    final_clf = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.06,
        objective="multi:softprob",
        num_class=3,
        random_state=random_state
    )
    final_clf.fit(X, y, sample_weight=sample_weights)

    # Fit TreeSHAP explainer if available
    explainer = None
    if _SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(final_clf)
        except Exception:
            explainer = None

    return {
        "model": final_clf,
        "explainer": explainer,
        "feature_cols": feature_cols,
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "y_true": y
    }
