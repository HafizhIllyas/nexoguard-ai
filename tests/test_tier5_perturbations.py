"""
tests/test_tier5_perturbations.py
Tier 5: Signal perturbation engine verification, Markov burst dropouts, and end-to-end telemetry degradation tests.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from nexoguard.stress_testing.perturbations import (
    inject_gilbert_elliott_dropout,
    inject_telemetry_noise,
    inject_baseline_wander,
    inject_spike_impulses,
    apply_perturbation_scenario
)
from nexoguard.signal.reader import read_wfdb_record
from nexoguard.signal.preprocessor import clean_fhr_signal, calculate_signal_loss_pct
from nexoguard.features.extractor import extract_features_from_signal, CONTRACT_FEATURE_COLS
from nexoguard.models.inference import predict_risk
from nexoguard.guardrails.clinical_rules import apply_safety_guardrails

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class TestSignalPerturbations(unittest.TestCase):

    def setUp(self):
        self.fs = 4.0
        self.duration_sec = 600
        self.n_samples = int(self.duration_sec * self.fs)
        self.fhr_clean = np.full(self.n_samples, 140.0)

    def test_gilbert_elliott_dropout(self):
        """Verify Gilbert-Elliott Markov model injects target burst dropout percentage."""
        degraded = inject_gilbert_elliott_dropout(
            self.fhr_clean, target_dropout_pct=25.0, mean_burst_sec=10.0, fs=self.fs, random_seed=42
        )
        self.assertEqual(len(degraded), len(self.fhr_clean))
        dropout_pct = (np.sum(degraded <= 0.0) / len(degraded)) * 100.0
        self.assertAlmostEqual(dropout_pct, 25.0, delta=6.0)

    def test_telemetry_noise(self):
        """Verify Gaussian telemetry noise has expected mean and standard deviation and preserves dropouts."""
        noisy = inject_telemetry_noise(self.fhr_clean, sigma_bpm=4.0, random_seed=42)
        noise = noisy - self.fhr_clean
        self.assertAlmostEqual(np.mean(noise), 0.0, delta=0.5)
        self.assertAlmostEqual(np.std(noise), 4.0, delta=0.5)

        # Ensure existing dropout samples are preserved
        sparse = self.fhr_clean.copy()
        sparse[:100] = 0.0
        noisy_sparse = inject_telemetry_noise(sparse, sigma_bpm=4.0, random_seed=42)
        self.assertTrue(np.all(noisy_sparse[:100] == 0.0))

    def test_baseline_wander(self):
        """Verify sinusoidal baseline wander adds bounded low-frequency drift."""
        wandered = inject_baseline_wander(self.fhr_clean, amplitude_bpm=5.0, freq_hz=0.10, fs=self.fs)
        diff = wandered - self.fhr_clean
        self.assertAlmostEqual(np.max(diff), 5.0, delta=0.2)
        self.assertAlmostEqual(np.min(diff), -5.0, delta=0.2)

    def test_spike_impulses(self):
        """Verify transient spike impulses are within expected amplitude bounds."""
        spiked = inject_spike_impulses(self.fhr_clean, spike_prob=0.05, max_amp_bpm=30.0, random_seed=42)
        diff = spiked - self.fhr_clean
        self.assertTrue(np.all(np.abs(diff) <= 30.01))
        # Verify spikes were actually injected
        self.assertGreater(np.sum(diff != 0.0), 0)

    def test_apply_perturbation_scenarios(self):
        """Verify all 3 scenarios execute and produce expected signal properties."""
        for sc in ["scenario_1_mild", "scenario_2_moderate", "scenario_3_severe"]:
            out = apply_perturbation_scenario(self.fhr_clean, sc, fs=self.fs, seed=42)
            self.assertEqual(len(out), len(self.fhr_clean))
            loss = calculate_signal_loss_pct(out)
            self.assertGreater(loss, 0.0)

        # Invalid scenario must raise ValueError
        with self.assertRaises(ValueError):
            apply_perturbation_scenario(self.fhr_clean, "scenario_unknown", fs=self.fs)


class TestPerturbationEndToEndPipeline(unittest.TestCase):

    def test_end_to_end_patient_1001_degradation(self):
        """Ingest patient 1001 (acidemic, pH=7.14), apply perturbation, extract features, and verify guardrail protection."""
        rec_path = DATA_DIR / "1001.dat"
        if not rec_path.is_file():
            self.skipTest("1001.dat not found in data/")

        record = read_wfdb_record(rec_path, data_dir=DATA_DIR)
        loss_clean = calculate_signal_loss_pct(record.fhr)

        # Apply moderate degradation
        fhr_degraded = apply_perturbation_scenario(record.fhr, "scenario_2_moderate", fs=record.fs, seed=42)
        loss_degraded = calculate_signal_loss_pct(fhr_degraded)
        self.assertGreater(loss_degraded, loss_clean)

        # Extract features
        feats_df = extract_features_from_signal(
            fhr_degraded, record.uc, fs=record.fs, window_len_sec=600, step_sec=300, patient_id="1001"
        )
        self.assertGreater(len(feats_df), 0)
        for col in CONTRACT_FEATURE_COLS:
            self.assertIn(col, feats_df.columns)
            self.assertEqual(feats_df[col].isna().sum(), 0)

        # Run genuine inference and guardrails
        model_file = MODELS_DIR / "model_xgb.pkl"
        preds_df = predict_risk(feats_df, model_path=model_file if model_file.is_file() else None)
        guarded_df = apply_safety_guardrails(preds_df, feats_df)

        # Severe acidemia patient (pH 7.14) must NEVER receive reassuring Category I under degraded telemetry
        for cat in guarded_df["final_category"]:
            self.assertNotEqual(cat, "Category I", "Severe acidemic patient classified as Category I under degraded telemetry!")


if __name__ == "__main__":
    unittest.main()
