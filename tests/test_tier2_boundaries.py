"""
tests/test_tier2_boundaries.py
Tier 2: Boundary value analysis (BVA), extreme physiological conditions, and error paths.
"""

import unittest
import numpy as np
import pandas as pd

from nexoguard.signal.preprocessor import clean_fhr_signal, calculate_signal_loss_pct
from nexoguard.signal.windowing import slice_signal_windows
from nexoguard.features.extractor import extract_features_from_signal, extract_features_from_window
from nexoguard.features.decelerations import analyze_morphometry_and_synchrony
from nexoguard.guardrails.clinical_rules import evaluate_single_window_guardrail


class TestTier2Boundaries(unittest.TestCase):

    def test_empty_signal_raises_error(self):
        """Empty signals must raise ValueError with informative message."""
        with self.assertRaises(ValueError):
            clean_fhr_signal(np.array([]))

        with self.assertRaises(ValueError):
            extract_features_from_signal(np.array([]), np.array([]))

    def test_signal_shorter_than_window_raises_error(self):
        """Signals shorter than 10-minute window length must raise ValueError."""
        short_fhr = np.full(1000, 140.0)  # Only 250 seconds at 4 Hz
        short_uc = np.full(1000, 10.0)
        with self.assertRaises(ValueError):
            extract_features_from_signal(short_fhr, short_uc, window_len_sec=600)

    def test_exact_single_window_signal(self):
        """Signal exactly 2400 samples (600s) produces exactly 1 window."""
        fhr = np.full(2400, 140.0)
        uc = np.full(2400, 15.0)
        windows = slice_signal_windows(fhr, uc, fs=4.0, window_sec=600, step_sec=300)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].window_id, "Unknown_W00")

    def test_100_percent_signal_loss(self):
        """100% missing data marks signal_loss_pct = 100.0 and UNINTERPRETABLE flag."""
        zero_signal = np.zeros(2400)
        loss = calculate_signal_loss_pct(zero_signal)
        self.assertEqual(loss, 100.0)

        feat = extract_features_from_window(zero_signal, np.full(2400, 10.0))
        self.assertEqual(feat["signal_loss_pct"], 100.0)
        self.assertEqual(feat["quality_flag"], "UNINTERPRETABLE")

        # Guardrail evaluation on 100% loss trace suppresses Category I
        res = evaluate_single_window_guardrail(feat, raw_category="Category I")
        self.assertNotEqual(res["final_category"], "Category I")

    def test_zero_contractions_and_zero_decelerations_lag(self):
        """When no contractions or decelerations occur, lag Δt must safely be 0.0 without crash."""
        fhr = np.full(2400, 140.0)
        bl = np.full(2400, 140.0)
        uc = np.full(2400, 5.0)  # Flat uterine tone
        morph = analyze_morphometry_and_synchrony(fhr, bl, uc, fs=4.0)
        self.assertEqual(morph["decel_count"], 0)
        self.assertEqual(morph["contraction_count"], 0)
        self.assertEqual(morph["decel_uc_lag_sec"], 0.0)

    def test_extreme_heart_rate_boundaries(self):
        """Extreme heart rate values at boundary of physiological envelope."""
        # Baseline = 99.0 bpm (exact boundary for Rule 1 Bradycardia)
        case_brady_bound = {"baseline_bpm": 99.9}
        res_brady = evaluate_single_window_guardrail(case_brady_bound, raw_category="Category I")
        self.assertEqual(res_brady["final_category"], "Category III")

        # Baseline = 100.1 bpm (does not trigger Rule 1 Bradycardia on baseline alone)
        case_norm_bound = {"baseline_bpm": 100.1, "stv_ms": 5.0, "signal_loss_pct": 5.0}
        res_norm = evaluate_single_window_guardrail(case_norm_bound, raw_category="Category I")
        self.assertNotIn("RULE_1_SEVERE_BRADYCARDIA", res_norm["triggered_rules"])


if __name__ == "__main__":
    unittest.main()
