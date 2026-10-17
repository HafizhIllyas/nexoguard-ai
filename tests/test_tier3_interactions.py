"""
tests/test_tier3_interactions.py
Tier 3: Component interactions, pipeline end-to-end integration, and anti-leakage verification.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from nexoguard.signal.reader import read_wfdb_record
from nexoguard.features.extractor import extract_features_from_signal, CONTRACT_FEATURE_COLS
from nexoguard.models.stratification import get_patient_stratified_folds
from nexoguard.models.thresholding import apply_calibrated_thresholds
from nexoguard.models.inference import predict_risk
from nexoguard.guardrails.clinical_rules import apply_safety_guardrails
from nexoguard.models.serialization import save_model_bundle, load_model_bundle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class TestTier3Interactions(unittest.TestCase):

    def test_stratification_zero_patient_leakage(self):
        """Verify StratifiedGroupKFold strictly prevents patient window overlap across all folds."""
        features_csv = DATA_DIR / "features.csv"
        if not features_csv.is_file():
            self.skipTest("features.csv not found")

        df = pd.read_csv(features_csv)
        folds = list(get_patient_stratified_folds(df, n_splits=5, seed=42))
        self.assertGreaterEqual(len(folds), 2)

        patients = df["patient_id"].values
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_pts = set(patients[train_idx])
            val_pts = set(patients[val_idx])
            overlap = train_pts.intersection(val_pts)
            self.assertEqual(len(overlap), 0, f"Data leakage in fold {fold_idx}: {overlap}")

    def test_full_pipeline_signal_to_guardrails(self):
        """Verify end-to-end flow: Raw CTG -> Extracted Features -> Inference -> Guardrails."""
        rec_path = DATA_DIR / "1001.dat"
        if not rec_path.is_file():
            self.skipTest("1001.dat not found")

        # 1. Ingest
        rec = read_wfdb_record("1001", data_dir=DATA_DIR)

        # 2. Extract features from signal
        feats_df = extract_features_from_signal(
            rec.fhr, rec.uc, fs=rec.fs, window_len_sec=600, step_sec=300, patient_id="1001"
        )
        self.assertGreater(len(feats_df), 0)

        # 3. Model inference
        model_file = MODELS_DIR / "model_xgb.pkl"
        preds_df = predict_risk(feats_df, model_path=model_file if model_file.is_file() else None)
        self.assertEqual(len(preds_df), len(feats_df))
        self.assertIn("prob_cat_i", preds_df.columns)
        self.assertIn("pred_category_raw", preds_df.columns)

        # 4. Apply safety guardrails
        guarded_df = apply_safety_guardrails(preds_df, feats_df)
        self.assertEqual(len(guarded_df), len(feats_df))
        self.assertIn("final_category", guarded_df.columns)
        self.assertIn("guardrail_triggered", guarded_df.columns)
        self.assertIn("clinical_alert_level", guarded_df.columns)

    def test_calibrated_thresholding_vs_argmax(self):
        """Verify calibrated decision rule P(Cat III) >= 0.25 flags borderline hypoxic risk."""
        # Ambiguous probability distribution where argmax would choose Category II (0.45)
        # but elevated P(Cat III) is 0.30 (>= 0.25)
        probs = np.array([[0.25, 0.45, 0.30]])
        classes, risks = apply_calibrated_thresholds(probs, tau_cat3=0.25)
        self.assertEqual(classes[0], 2)  # Category III
        self.assertGreater(risks[0], 0.30)

    def test_model_serialization_roundtrip(self):
        """Verify save and load bundle preserves model metadata and schema."""
        test_bundle = {
            "model": "dummy_estimator_placeholder",
            "feature_cols": CONTRACT_FEATURE_COLS,
            "metrics": {"cat3_recall": 0.96}
        }
        tmp_path = MODELS_DIR / "test_model_roundtrip.pkl"
        try:
            saved_path = save_model_bundle(test_bundle, tmp_path)
            self.assertTrue(Path(saved_path).is_file())
            loaded = load_model_bundle(tmp_path)
            self.assertEqual(loaded["feature_cols"], CONTRACT_FEATURE_COLS)
            self.assertAlmostEqual(loaded["metrics"]["cat3_recall"], 0.96)
        finally:
            if tmp_path.is_file():
                tmp_path.unlink()
            if tmp_path.with_suffix(".json").is_file():
                tmp_path.with_suffix(".json").unlink()


if __name__ == "__main__":
    unittest.main()
