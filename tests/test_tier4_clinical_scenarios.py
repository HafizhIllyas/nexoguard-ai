"""
tests/test_tier4_clinical_scenarios.py
Tier 4: Clinical scenario evaluation, benchmarking validation, and stress harness verification.
Remediated for Milestone 1 Iteration 2:
- Genuine model inference and dynamic ROC-AUC evaluation in benchmark runner
- Stress testing harness resilience and zero false reassurance invariant
- Deliverable disk artifact persistence verification
- Anti-cheat oracle rejection (inference without outcome columns)
"""

import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd

from nexoguard.benchmark.runner import run_clinical_benchmark
from nexoguard.stress_testing.harness import run_stress_testing_harness
from nexoguard.guardrails.clinical_rules import evaluate_single_window_guardrail, apply_safety_guardrails
from nexoguard.models.inference import predict_risk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class TestTier4ClinicalScenarios(unittest.TestCase):

    def setUp(self):
        self.features_csv = DATA_DIR / "features.csv"
        self.outcomes_csv = DATA_DIR / "clinical_outcomes.csv"
        self.test_json_benchmark = MODELS_DIR / "test_benchmark_results.json"
        self.test_csv_benchmark = MODELS_DIR / "test_benchmark_summary.csv"
        self.test_json_stress = MODELS_DIR / "test_stress_results.json"

    def tearDown(self):
        for p in [self.test_json_benchmark, self.test_csv_benchmark, self.test_json_stress]:
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

    def test_clinical_benchmarking_runner(self):
        """Verify automated clinical benchmark computes all FIGO metrics and acidemia recalls."""
        if not self.features_csv.is_file() or not self.outcomes_csv.is_file():
            self.skipTest("Required dataset CSVs not found in data/")

        bench = run_clinical_benchmark(
            features_csv_path=self.features_csv,
            outcomes_csv_path=self.outcomes_csv,
            output_json_path=self.test_json_benchmark,
            output_csv_path=self.test_csv_benchmark,
            model_path=MODELS_DIR / "model_xgb.pkl" if (MODELS_DIR / "model_xgb.pkl").is_file() else None
        )

        self.assertIn("accuracy", bench)
        self.assertIn("macro_f1", bench)
        self.assertIn("roc_auc", bench)
        self.assertIn("cat_iii_recall", bench)
        self.assertIn("ph_under_715_recall", bench)
        self.assertIn("ph_under_705_recall", bench)
        self.assertIn("confusion_matrix", bench)

        # Performance targets
        self.assertGreaterEqual(bench["accuracy"], 0.70)
        self.assertGreaterEqual(bench["ph_under_715_recall"], 0.90)
        self.assertGreaterEqual(bench["ph_under_705_recall"], 0.95)

        # File export verification
        self.assertTrue(self.test_json_benchmark.is_file())
        self.assertTrue(self.test_csv_benchmark.is_file())

        with open(self.test_json_benchmark, "r") as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data["accuracy"], bench["accuracy"])

    def test_stress_testing_harness_3_scenarios(self):
        """Verify stress harness executes 3 distinct scenarios and asserts zero false reassurance."""
        if not self.features_csv.is_file():
            self.skipTest("features.csv not found in data/")

        df = pd.read_csv(self.features_csv)
        stress_res = run_stress_testing_harness(
            clean_records=df,
            output_json_path=self.test_json_stress,
            model_path=MODELS_DIR / "model_xgb.pkl" if (MODELS_DIR / "model_xgb.pkl").is_file() else None
        )

        scenarios = stress_res["scenarios"]
        self.assertIn("scenario_1_mild", scenarios)
        self.assertIn("scenario_2_moderate", scenarios)
        self.assertIn("scenario_3_severe", scenarios)

        # Verify resilience and zero false reassurance invariant
        self.assertEqual(stress_res["false_reassurance_count"], 0)
        self.assertTrue(stress_res["zero_false_reassurance_invariant_met"])
        self.assertTrue(self.test_json_stress.is_file())

        # Check Scenario 1 (mild) retention
        self.assertGreaterEqual(scenarios["scenario_1_mild"]["cat_iii_recall"], 0.90)

    def test_catastrophic_acidemia_patient_1029(self):
        """Patient 1029 (pH=6.97, BDecf=19.38) must be triaged as Category III with emergency alert."""
        patient_1029_feature = {
            "window_id": "1029_W00",
            "patient_id": "1029",
            "baseline_bpm": 138.0,
            "stv_ms": 2.1,
            "ltv_bpm": 8.0,
            "accel_count": 0,
            "decel_count": 3,
            "decel_max_depth_bpm": 45.0,
            "decel_total_area": 3500.0,
            "decel_uc_lag_sec": 32.0,
            "signal_loss_pct": 5.0
        }
        res = evaluate_single_window_guardrail(
            patient_1029_feature, raw_category="Category II", raw_risk_score=0.70
        )
        self.assertEqual(res["final_category"], "Category III")
        self.assertGreaterEqual(res["final_risk_score"], 0.95)
        self.assertTrue(res["guardrail_triggered"])

    def test_disk_artifacts_persistence(self):
        """Verify deliverable artifacts benchmark_results.json, benchmark_summary.csv, and stress_test_results.json exist on disk."""
        bench_json = MODELS_DIR / "benchmark_results.json"
        stress_json = MODELS_DIR / "stress_test_results.json"
        bench_csv = MODELS_DIR / "benchmark_summary.csv"

        self.assertTrue(bench_json.is_file(), f"Missing deliverable: {bench_json}")
        self.assertGreater(bench_json.stat().st_size, 100)
        self.assertTrue(stress_json.is_file(), f"Missing deliverable: {stress_json}")
        self.assertGreater(stress_json.stat().st_size, 100)
        self.assertTrue(bench_csv.is_file(), f"Missing deliverable: {bench_csv}")
        self.assertGreater(bench_csv.stat().st_size, 50)

        with open(bench_json, "r", encoding="utf-8") as f:
            bench_data = json.load(f)
        for key in ["accuracy", "macro_f1", "roc_auc", "cat_iii_recall", "ph_under_715_recall"]:
            self.assertIn(key, bench_data)

        with open(stress_json, "r", encoding="utf-8") as f:
            stress_data = json.load(f)
        self.assertIn("scenarios", stress_data)
        self.assertTrue(stress_data["zero_false_reassurance_invariant_met"])
        self.assertEqual(stress_data["false_reassurance_count"], 0)

    def test_oracle_cheat_rejection(self):
        """Verify model inference and guardrails operate purely on telemetry features without target columns."""
        if not self.features_csv.is_file():
            self.skipTest("features.csv not found")

        df = pd.read_csv(self.features_csv)
        # Strip all outcome / oracle columns
        outcome_cols = [c for c in ["pH", "pH_ground_truth", "BDecf", "label", "Apgar1", "Apgar5"] if c in df.columns]
        stripped_df = df.drop(columns=outcome_cols)

        # predict_risk must succeed without outcome columns
        preds = predict_risk(stripped_df, model_path=MODELS_DIR / "model_xgb.pkl" if (MODELS_DIR / "model_xgb.pkl").is_file() else None)
        self.assertEqual(len(preds), len(stripped_df))
        self.assertIn("pred_category_raw", preds.columns)
        self.assertIn("risk_score", preds.columns)

        # apply_safety_guardrails must also succeed
        guarded = apply_safety_guardrails(preds, stripped_df)
        self.assertEqual(len(guarded), len(stripped_df))
        self.assertIn("final_category", guarded.columns)


if __name__ == "__main__":
    unittest.main()
