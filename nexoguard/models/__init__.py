"""
nexoguard.models
Predictive modeling, patient-stratified split, calibrated risk scoring, and serialization.
"""

from nexoguard.models.stratification import get_patient_stratified_folds
from nexoguard.models.thresholding import apply_calibrated_thresholds
from nexoguard.models.train import train_model, compute_asymmetric_sample_weights, FEATURE_COLS
from nexoguard.models.inference import predict_risk, CATEGORY_MAP
from nexoguard.models.serialization import save_model_bundle, load_model_bundle

__all__ = [
    "get_patient_stratified_folds",
    "apply_calibrated_thresholds",
    "train_model",
    "compute_asymmetric_sample_weights",
    "predict_risk",
    "save_model_bundle",
    "load_model_bundle",
    "FEATURE_COLS",
    "CATEGORY_MAP",
]
