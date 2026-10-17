"""
tests/test_guardrails.py
Comprehensive verification of all 6 FIGO 2015 clinical safety guardrails and the Zero False Reassurance invariant.
"""

import unittest
import numpy as np
import pandas as pd

from nexoguard.guardrails.clinical_rules import (
    evaluate_single_window_guardrail,
    apply_safety_guardrails,
    RULE_SEVERE_BRADYCARDIA,
    RULE_PROLONGED_DECEL,
    RULE_LATE_DECEL_LOW_STV,
    RULE_ABSENT_VARIABILITY,
    RULE_SINUSOIDAL_PATTERN,
    RULE_SIGNAL_INTEGRITY
)


class TestClinicalGuardrails(unittest.TestCase):

    def test_rule_1_severe_bradycardia(self):
        """Rule 1: Baseline < 100 bpm must force Category III (Pathological) and emergency alert."""
        case = {
            "window_id": "W_BRADY",
            "baseline_bpm": 92.0,
            "stv_ms": 4.5,
            "signal_loss_pct": 2.0
        }
        res = evaluate_single_window_guardrail(case, raw_category="Category I", raw_risk_score=0.10)
        self.assertEqual(res["final_category"], "Category III")
        self.assertGreaterEqual(res["final_risk_score"], 0.98)
        self.assertEqual(res["alert_level"], "EMERGENCY")
        self.assertIn(RULE_SEVERE_BRADYCARDIA, res["triggered_rules"])
        self.assertTrue(res["zero_false_reassurance_verified"])

    def test_rule_2_prolonged_deceleration(self):
        """Rule 2: Decel depth >= 30 bpm and large area must force Category III and emergency alert."""
        case = {
            "window_id": "W_PROLONGED",
            "baseline_bpm": 135.0,
            "decel_count": 1,
            "decel_max_depth_bpm": 38.0,
            "decel_total_area": 3200.0,
            "signal_loss_pct": 5.0
        }
        res = evaluate_single_window_guardrail(case, raw_category="Category I", raw_risk_score=0.15)
        self.assertEqual(res["final_category"], "Category III")
        self.assertGreaterEqual(res["final_risk_score"], 0.99)
        self.assertEqual(res["alert_level"], "EMERGENCY")
        self.assertIn(RULE_PROLONGED_DECEL, res["triggered_rules"])

    def test_rule_3_late_deceleration_low_stv(self):
        """Rule 3: Decel lag >= 20s with acidemic STV (< 3.0 ms) must force Category III."""
        case = {
            "window_id": "W_LATE",
            "baseline_bpm": 140.0,
            "stv_ms": 2.3,
            "decel_count": 2,
            "decel_uc_lag_sec": 28.0,
            "signal_loss_pct": 8.0
        }
        res = evaluate_single_window_guardrail(case, raw_category="Category I", raw_risk_score=0.20)
        self.assertEqual(res["final_category"], "Category III")
        self.assertGreaterEqual(res["final_risk_score"], 0.96)
        self.assertEqual(res["alert_level"], "PATHOLOGICAL")
        self.assertIn(RULE_LATE_DECEL_LOW_STV, res["triggered_rules"])

    def test_rule_4_absent_variability(self):
        """Rule 4: STV < 2.5 ms or LTV < 3.0 bpm without accels must force Category III."""
        case = {
            "window_id": "W_ABSENT",
            "baseline_bpm": 138.0,
            "stv_ms": 1.7,
            "ltv_bpm": 2.2,
            "accel_count": 0,
            "signal_loss_pct": 4.0
        }
        res = evaluate_single_window_guardrail(case, raw_category="Category I", raw_risk_score=0.15)
        self.assertEqual(res["final_category"], "Category III")
        self.assertGreaterEqual(res["final_risk_score"], 0.92)
        self.assertIn(RULE_ABSENT_VARIABILITY, res["triggered_rules"])

    def test_rule_5_sinusoidal_pattern(self):
        """Rule 5: Smooth sine wave (5-15 bpm amplitude, STV < 2.0 ms, no accels) must force Category III."""
        case = {
            "window_id": "W_SINUSOIDAL",
            "baseline_bpm": 130.0,
            "stv_ms": 1.4,
            "ltv_bpm": 10.0,
            "accel_count": 0,
            "decel_count": 0,
            "is_sinusoidal": True
        }
        res = evaluate_single_window_guardrail(case, raw_category="Category I", raw_risk_score=0.20)
        self.assertEqual(res["final_category"], "Category III")
        self.assertGreaterEqual(res["final_risk_score"], 0.99)
        self.assertEqual(res["alert_level"], "EMERGENCY")
        self.assertIn(RULE_SINUSOIDAL_PATTERN, res["triggered_rules"])

    def test_rule_6_telemetry_signal_loss_suppression(self):
        """Rule 6: Signal loss > 40% or MHR corr > 0.80 must strictly suppress Category I."""
        # 1. 45% signal loss
        case_dropout = {
            "window_id": "W_DROPOUT",
            "baseline_bpm": 140.0,
            "signal_loss_pct": 45.0,
            "mhr_fhr_cross_corr": 0.15
        }
        res_drop = evaluate_single_window_guardrail(case_dropout, raw_category="Category I", raw_risk_score=0.10)
        self.assertNotEqual(res_drop["final_category"], "Category I")
        self.assertEqual(res_drop["final_category"], "Category II")
        self.assertIn(RULE_SIGNAL_INTEGRITY, res_drop["triggered_rules"])
        self.assertEqual(res_drop["alert_level"], "DATA_INTEGRITY_ALERT")

        # 2. Maternal pulse lock (cross correlation > 0.80)
        case_mhr = {
            "window_id": "W_MHR",
            "baseline_bpm": 140.0,
            "signal_loss_pct": 10.0,
            "mhr_fhr_cross_corr": 0.88
        }
        res_mhr = evaluate_single_window_guardrail(case_mhr, raw_category="Category I", raw_risk_score=0.10)
        self.assertNotEqual(res_mhr["final_category"], "Category I")
        self.assertIn(RULE_SIGNAL_INTEGRITY, res_mhr["triggered_rules"])

    def test_normal_reassuring_trace_no_trigger(self):
        """Normal physiological CTG with good variability and accelerations remains Category I."""
        case_normal = {
            "window_id": "W_NORMAL",
            "baseline_bpm": 135.0,
            "stv_ms": 6.2,
            "ltv_bpm": 14.0,
            "accel_count": 2,
            "decel_count": 0,
            "signal_loss_pct": 3.0,
            "mhr_fhr_cross_corr": 0.15
        }
        res = evaluate_single_window_guardrail(case_normal, raw_category="Category I", raw_risk_score=0.08)
        self.assertEqual(res["final_category"], "Category I")
        self.assertFalse(res["guardrail_triggered"])
        self.assertEqual(res["alert_level"], "REASSURING")

    def test_batch_guardrail_dataframe_application(self):
        """Verify apply_safety_guardrails works vectorially over DataFrames."""
        feats_df = pd.DataFrame([
            {"window_id": "W1", "baseline_bpm": 90.0, "stv_ms": 4.0, "signal_loss_pct": 5.0},      # Rule 1 Brady
            {"window_id": "W2", "baseline_bpm": 140.0, "stv_ms": 6.0, "signal_loss_pct": 50.0},    # Rule 6 Loss
            {"window_id": "W3", "baseline_bpm": 135.0, "stv_ms": 6.0, "signal_loss_pct": 2.0}     # Normal
        ])
        preds_df = pd.DataFrame([
            {"window_id": "W1", "pred_category_raw": "Category I", "risk_score": 0.10},
            {"window_id": "W2", "pred_category_raw": "Category I", "risk_score": 0.10},
            {"window_id": "W3", "pred_category_raw": "Category I", "risk_score": 0.10}
        ])

        guarded_df = apply_safety_guardrails(preds_df, feats_df)
        self.assertEqual(guarded_df.iloc[0]["final_category"], "Category III")
        self.assertEqual(guarded_df.iloc[1]["final_category"], "Category II")
        self.assertEqual(guarded_df.iloc[2]["final_category"], "Category I")


if __name__ == "__main__":
    unittest.main()
