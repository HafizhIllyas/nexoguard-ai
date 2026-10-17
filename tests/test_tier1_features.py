"""
tests/test_tier1_features.py
Tier 1: Biomedical feature conformance and unit algorithm tests.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from nexoguard.signal.reader import read_wfdb_record, parse_header_file
from nexoguard.signal.preprocessor import clean_fhr_signal, calculate_signal_loss_pct
from nexoguard.features.baseline import calculate_nexoprima_ema
from nexoguard.features.variability import calculate_stv_ms, calculate_ltv_bpm
from nexoguard.features.decelerations import analyze_morphometry_and_synchrony
from nexoguard.features.extractor import extract_features_from_window, CONTRACT_FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class TestTier1Features(unittest.TestCase):

    def test_wfdb_record_reader(self):
        """Verify binary 16-bit WFDB reader parses CTU-UHB records correctly."""
        rec_path = DATA_DIR / "1001.dat"
        if rec_path.is_file():
            rec = read_wfdb_record("1001", data_dir=DATA_DIR)
            self.assertEqual(rec.rec_id, "1001")
            self.assertEqual(rec.fs, 4.0)
            self.assertGreater(len(rec.fhr), 0)
            self.assertGreater(len(rec.uc), 0)
            self.assertAlmostEqual(rec.duration_sec, len(rec.fhr) / 4.0)
            self.assertIn("pH", rec.meta)
            self.assertAlmostEqual(rec.meta["pH"], 7.14)

    def test_preprocessor_clipping_and_cleaning(self):
        """Verify artifact rejection and physiological bounds [50, 240 bpm]."""
        # Array with invalid values: <=0, <50, >240, NaN
        raw = np.array([0.0, 30.0, 140.0, 145.0, 260.0, np.nan, 150.0])
        cleaned = clean_fhr_signal(raw, fs=4.0)
        self.assertEqual(len(cleaned), len(raw))
        self.assertTrue(np.all(cleaned >= 50.0))
        self.assertTrue(np.all(cleaned <= 240.0))
        self.assertFalse(np.any(np.isnan(cleaned)))

    def test_signal_loss_pct_calculation(self):
        """Verify dropout calculation on known mask."""
        # 10 samples: 2 dropouts (0 and 260)
        raw = np.array([140.0, 140.0, 0.0, 140.0, 140.0, 260.0, 140.0, 140.0, 140.0, 140.0])
        loss = calculate_signal_loss_pct(raw)
        self.assertAlmostEqual(loss, 20.0, places=1)

    def test_nexoprima_ema_baseline(self):
        """Verify EMA baseline calculation and deceleration dampening."""
        steady_fhr = np.full(1000, 140.0)
        baseline = calculate_nexoprima_ema(steady_fhr, alpha=0.85)
        self.assertEqual(len(baseline), len(steady_fhr))
        self.assertAlmostEqual(baseline[-1], 140.0, delta=0.5)

        # Deceleration trace: drop to 80 bpm
        decel_trace = np.full(500, 140.0)
        decel_trace[200:300] = 80.0
        dampened_bl = calculate_nexoprima_ema(decel_trace, alpha=0.85, decel_dampen=True)
        # Dampening prevents baseline from falling all the way to 80
        self.assertGreater(dampened_bl[250], 100.0)

    def test_variability_metrics(self):
        """Verify STV and LTV calculation on synthetic signals."""
        flat_signal = np.full(2400, 140.0)
        stv_flat = calculate_stv_ms(flat_signal, fs=4.0)
        self.assertAlmostEqual(stv_flat, 0.0, delta=1e-5)

        ltv_flat = calculate_ltv_bpm(flat_signal, fs=4.0)
        self.assertAlmostEqual(ltv_flat, 0.0, delta=1e-5)

        # Sine wave with peak-to-peak amplitude of 20 bpm
        t = np.arange(2400) / 4.0
        sine_signal = 140.0 + 10.0 * np.sin(2 * np.pi * 0.1 * t)
        ltv_sine = calculate_ltv_bpm(sine_signal, fs=4.0)
        self.assertAlmostEqual(ltv_sine, 20.0, delta=1.5)

    def test_decelerations_and_synchrony(self):
        """Verify morphometry and contraction lag calculation."""
        fhr = np.full(2400, 140.0)
        bl = np.full(2400, 140.0)
        uc = np.full(2400, 10.0)

        # Inject contraction at sample 600 (peak), duration >= 30s (140 samples)
        uc[530:670] = 50.0
        uc[600] = 55.0  # distinct contraction peak at sample 600

        # Inject deceleration at sample 720 (nadir at 720, 120 samples after UC peak = 30s lag)
        fhr[660:780] = 100.0  # -40 bpm depth for 120 samples (30s)
        fhr[720] = 95.0       # true nadir at sample 720

        morph = analyze_morphometry_and_synchrony(fhr, bl, uc, fs=4.0)
        self.assertGreaterEqual(morph["decel_count"], 1)
        self.assertGreaterEqual(morph["decel_max_depth_bpm"], 35.0)
        self.assertGreaterEqual(morph["contraction_count"], 1)
        # Lag: 720 - 600 = 120 samples / 4 Hz = 30s
        self.assertAlmostEqual(morph["decel_uc_lag_sec"], 30.0, delta=2.0)

    def test_window_features_contract(self):
        """Verify all contract feature columns are populated."""
        fhr_win = np.full(2400, 135.0)
        uc_win = np.full(2400, 12.0)
        feat = extract_features_from_window(fhr_win, uc_win, win_id="TEST_W01", patient_id="9999")
        for col in CONTRACT_FEATURE_COLS:
            self.assertIn(col, feat)
            if isinstance(feat[col], (int, float, np.floating)):
                self.assertFalse(np.isnan(feat[col]))


if __name__ == "__main__":
    unittest.main()
