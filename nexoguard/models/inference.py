"""
nexoguard.models.inference
Public model inference contract complying with PROJECT.md Interface Contracts.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd

from nexoguard.models.serialization import load_model_bundle
from nexoguard.models.thresholding import apply_calibrated_thresholds

FEATURE_COLS = [
    "baseline_bpm", "stv_ms", "ltv_bpm", "accel_count", "decel_count",
    "decel_max_depth_bpm", "decel_total_area", "contraction_count",
    "decel_uc_lag_sec", "mhr_fhr_cross_corr", "signal_loss_pct"
]

CATEGORY_MAP = {
    0: "Category I",
    1: "Category II",
    2: "Category III"
}


def predict_risk(
    features_df: pd.DataFrame,
    model_path: Optional[Union[str, Path]] = None,
    tau_cat3: float = 0.25
) -> pd.DataFrame:
    """
    Public Shared Contract (PROJECT.md:89-97):
    predict_risk(features_df: pd.DataFrame, model_path: Optional[str] = None) -> pd.DataFrame

    Inputs:
      - features_df: DataFrame matching feature contract
    Outputs:
      - pd.DataFrame with original features plus:
        ['prob_cat_i', 'prob_cat_ii', 'prob_cat_iii', 'pred_category_raw', 'risk_score']
      - Values: prob_cat_* in [0.0, 1.0] summing to 1.0;
        pred_category_raw in {'Category I', 'Category II', 'Category III'}
    """
    if features_df.empty:
        raise ValueError("Cannot predict risk on empty features DataFrame.")

    model_bundle = load_model_bundle(model_path)
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model = model_bundle["model"]
        cols = model_bundle.get("feature_cols", FEATURE_COLS)
    else:
        model = model_bundle
        cols = FEATURE_COLS

    missing_cols = [c for c in cols if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Features DataFrame missing required columns: {missing_cols}")

    X = features_df[cols].copy()
    probs = model.predict_proba(X)

    # Normalize probabilities to ensure they sum to 1.0
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, probs / row_sums, np.array([1.0, 0.0, 0.0]))

    pred_classes, risk_scores = apply_calibrated_thresholds(probs, tau_cat3=tau_cat3)

    out_df = features_df.copy()
    out_df["prob_cat_i"] = np.round(probs[:, 0], 4)
    out_df["prob_cat_ii"] = np.round(probs[:, 1], 4)
    out_df["prob_cat_iii"] = np.round(probs[:, 2], 4)
    out_df["pred_class_idx"] = pred_classes
    out_df["pred_category_raw"] = [CATEGORY_MAP.get(c, "Category I") for c in pred_classes]
    out_df["risk_score"] = np.round(risk_scores, 4)

    return out_df
