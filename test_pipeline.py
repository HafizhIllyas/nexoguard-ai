"""
test_pipeline.py
Automated non-interactive CLI runner for the NexoGuard AI Fetal Monitoring System.
Executes the full DSLC verification suite, clinical benchmarking runner, degradation stress-testing harness,
and programmatic guardrail assertions. Exits cleanly with status code 0 upon total pass.
Remediated for Milestone 1 Iteration 2:
- Active signal perturbation verification in Stage 2
- Self-healing model training check in Stage 3
- Explicit disk artifact persistence verification in Stage 4 and Stage 5
- Zero false reassurance invariant enforcement with intermediate dropout guardrail in Stage 6
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

# Dynamic path resolution (zero hardcoding)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure project root is in sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexoguard.signal.reader import read_wfdb_record, CTGRecord
from nexoguard.signal.preprocessor import clean_fhr_signal, calculate_signal_loss_pct
from nexoguard.features.extractor import extract_features_from_signal, CONTRACT_FEATURE_COLS
from nexoguard.models.stratification import get_patient_stratified_folds
from nexoguard.models.inference import predict_risk
from nexoguard.models.train import train_model
from nexoguard.models.serialization import save_model_bundle
from nexoguard.guardrails.clinical_rules import evaluate_single_window_guardrail, apply_safety_guardrails
from nexoguard.benchmark.runner import run_clinical_benchmark
from nexoguard.stress_testing.harness import run_stress_testing_harness
from nexoguard.stress_testing.perturbations import apply_perturbation_scenario


class TerminalColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def log_header(title: str):
    print(f"\n{TerminalColors.BOLD}{TerminalColors.OKCYAN}{'='*80}{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.OKCYAN}[>] {title}{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.OKCYAN}{'='*80}{TerminalColors.ENDC}")


def log_pass(msg: str):
    print(f"  {TerminalColors.OKGREEN}[PASS]{TerminalColors.ENDC} {msg}")


def log_fail(msg: str):
    print(f"  {TerminalColors.FAIL}[FAIL]{TerminalColors.ENDC} {msg}")


def log_info(msg: str):
    print(f"  {TerminalColors.OKBLUE}[INFO]{TerminalColors.ENDC} {msg}")


# --------------------------------------------------------------------------------------
# STAGE 1: Environment & Dataset Preflight Checks
# --------------------------------------------------------------------------------------
def stage_1_preflight_checks(data_dir: Path):
    log_header("STAGE 1: Environment & Dataset Preflight Checks")
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    outcomes_path = data_dir / "clinical_outcomes.csv"
    features_path = data_dir / "features.csv"

    assert outcomes_path.is_file(), f"Missing clinical_outcomes.csv at {outcomes_path}"
    assert features_path.is_file(), f"Missing features.csv at {features_path}"

    df_outcomes = pd.read_csv(outcomes_path)
    df_features = pd.read_csv(features_path)

    log_pass(f"Loaded clinical outcomes for {len(df_outcomes)} patients")
    log_pass(f"Loaded {len(df_features)} extracted 10-minute clinical windows")

    # Verify 40 patient cohort records exist on disk
    dat_files = list(data_dir.glob("*.dat"))
    hea_files = list(data_dir.glob("*.hea"))
    log_pass(f"Verified {len(dat_files)} binary .dat records and {len(hea_files)} .hea headers")
    assert len(dat_files) >= 40, f"Expected at least 40 binary records, found {len(dat_files)}"

    # Verify severe acidemia cases
    ph_col = "pH" if "pH" in df_outcomes.columns else None
    assert ph_col is not None, "clinical_outcomes.csv must contain 'pH' column"
    acidemic_count = int((df_outcomes[ph_col] < 7.15).sum())
    log_pass(f"Verified {acidemic_count} severe acidemia records (pH < 7.15) in cohort")
    assert acidemic_count >= 9, f"Expected >= 9 acidemic patients, found {acidemic_count}"

    return df_outcomes, df_features


# --------------------------------------------------------------------------------------
# STAGE 2: Real Patient Signal Processing & Feature Extraction Pipeline Check
# --------------------------------------------------------------------------------------
def stage_2_signal_and_feature_pipeline(data_dir: Path, df_features: pd.DataFrame):
    log_header("STAGE 2: Real Patient Signal Processing & Feature Extraction")

    # Ingest and process a real patient binary record directly through the pipeline
    test_record_id = "1001"
    record_path = data_dir / f"{test_record_id}.dat"
    record = read_wfdb_record(record_path, data_dir=data_dir)

    log_pass(f"Loaded real patient record {record.rec_id} ({len(record.fhr)} samples at {record.fs} Hz, duration: {record.duration_sec:.1f}s)")
    assert record.fs == 4.0, f"Expected 4.0 Hz sampling rate, got {record.fs}"
    assert len(record.fhr) > 0, "FHR telemetry signal is empty"
    assert len(record.uc) > 0, "UC uterine contraction signal is empty"

    # Clean telemetry signal
    fhr_clean = clean_fhr_signal(record.fhr, fs=record.fs)
    loss_clean = calculate_signal_loss_pct(record.fhr)
    log_pass(f"Cleaned FHR signal: min={np.min(fhr_clean):.1f} bpm, max={np.max(fhr_clean):.1f} bpm, signal loss={loss_clean:.2f}%")
    assert np.all(fhr_clean >= 50.0) and np.all(fhr_clean <= 240.0), "Cleaned FHR signal exceeds physiological bounds"

    # Active demonstration of physical time-series perturbation from perturbations.py
    fhr_degraded = apply_perturbation_scenario(record.fhr, "scenario_2_moderate", fs=record.fs, seed=42)
    loss_degraded = calculate_signal_loss_pct(fhr_degraded)
    assert loss_degraded > loss_clean, "Perturbation scenario failed to inject telemetry dropout"
    log_pass(f"Signal perturbation verified on patient {test_record_id}: clean loss={loss_clean:.2f}% -> degraded loss={loss_degraded:.2f}%")

    # Extract 10-minute sliding window features from raw recording
    live_features_df = extract_features_from_signal(
        record.fhr, record.uc, fs=record.fs, window_len_sec=600, step_sec=300, patient_id=test_record_id
    )
    log_pass(f"Extracted {len(live_features_df)} 10-minute sliding windows from patient {test_record_id}")
    assert len(live_features_df) > 0, "No windows extracted from real recording"

    # Verify feature contract adherence
    for col in CONTRACT_FEATURE_COLS:
        assert col in live_features_df.columns, f"Live extracted features missing contract column: {col}"
        assert col in df_features.columns, f"Cohort features.csv missing contract column: {col}"
        nan_count = df_features[col].isna().sum()
        assert nan_count == 0, f"Feature column '{col}' contains {nan_count} NaNs in features.csv"

    log_pass("All 12 feature contract columns validated with zero missing values")


# --------------------------------------------------------------------------------------
# STAGE 3: Patient Stratification & Model Inference Validation
# --------------------------------------------------------------------------------------
def stage_3_stratification_and_inference(df_features: pd.DataFrame, models_dir: Path):
    log_header("STAGE 3: Patient Stratification (Zero Leakage) & Model Inference")

    # 1. Assert patient stratification zero-leakage invariant
    folds = list(get_patient_stratified_folds(df_features, n_splits=5, seed=42))
    assert len(folds) >= 2, "Expected at least 2 stratification folds"

    patient_ids = df_features["patient_id"].values
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])
        leakage = train_patients.intersection(val_patients)
        assert len(leakage) == 0, f"Critical Data Leakage in fold {fold_idx}: patients {leakage} appear in both train and validation!"

    log_pass("StratifiedGroupKFold verified: 0 patient overlap across all folds (Zero Data Leakage)")

    # 2. Self-healing model preflight: ensure genuine model artifact exists
    model_file = models_dir / "model_xgb.pkl"
    if not model_file.is_file():
        log_info("model_xgb.pkl not found on disk. Initiating automated training on cohort features...")
        bundle = train_model(df_features, n_splits=5, random_state=42)
        save_model_bundle(bundle, model_file)
        log_pass("Trained and serialized production model_xgb.pkl")

    # 3. Test genuine model inference
    preds_df = predict_risk(df_features, model_path=model_file)

    for col in ["prob_cat_i", "prob_cat_ii", "prob_cat_iii", "pred_category_raw", "risk_score"]:
        assert col in preds_df.columns, f"Missing prediction column: {col}"

    # Verify probability distribution invariants
    prob_sums = preds_df["prob_cat_i"] + preds_df["prob_cat_ii"] + preds_df["prob_cat_iii"]
    assert np.allclose(prob_sums, 1.0, atol=1e-3), "Prediction probabilities do not sum to 1.0"
    assert np.all((preds_df["risk_score"] >= 0.0) & (preds_df["risk_score"] <= 1.0)), "Risk scores outside [0, 1]"

    log_pass(f"Executed genuine model inference on {len(preds_df)} windows; risk scores and calibrated classes generated")


# --------------------------------------------------------------------------------------
# STAGE 4: Automated Clinical Benchmarking Runner
# --------------------------------------------------------------------------------------
def stage_4_clinical_benchmarking(data_dir: Path, models_dir: Path):
    log_header("STAGE 4: Automated Clinical Benchmarking Runner (FIGO 2015)")

    features_csv = data_dir / "features.csv"
    outcomes_csv = data_dir / "clinical_outcomes.csv"
    out_json = models_dir / "benchmark_results.json"
    out_csv = models_dir / "benchmark_summary.csv"
    model_file = models_dir / "model_xgb.pkl"

    results = run_clinical_benchmark(
        features_csv_path=features_csv,
        outcomes_csv_path=outcomes_csv,
        output_json_path=out_json,
        output_csv_path=out_csv,
        model_path=model_file if model_file.is_file() else None
    )

    acc = results["accuracy"]
    macro_f1 = results["macro_f1"]
    weighted_f1 = results["weighted_f1"]
    roc_auc = results["roc_auc"]
    cat3_rec = results["cat_iii_recall"]
    ph_715_rec = results["ph_under_715_recall"]
    ph_705_rec = results["ph_under_705_recall"]
    bdecf_rec = results["bdecf_ge_12_recall"]

    log_pass(f"Overall Accuracy: {acc*100:.2f}%")
    log_pass(f"Macro F1-Score: {macro_f1*100:.2f}% (Weighted F1: {weighted_f1*100:.2f}%)")
    roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    log_pass(f"Multi-Class ROC-AUC: {roc_auc_str}")
    log_pass(f"Category III (Pathological) Recall: {cat3_rec*100:.2f}%")
    log_pass(f"Severe Acidemia Recall (pH < 7.15): {ph_715_rec*100:.2f}%")
    log_pass(f"Severe Acidemia Recall (pH < 7.05): {ph_705_rec*100:.2f}%")
    log_pass(f"Severe Metabolic Acidosis Recall (BDecf >= 12.0): {bdecf_rec*100:.2f}%")

    # Metric threshold assertions
    assert acc >= 0.70, f"Accuracy below minimum acceptance threshold: {acc:.3f}"
    assert ph_715_rec >= 0.90, f"Severe acidemia recall (pH < 7.15) is {ph_715_rec*100:.1f}%, must be >= 90%"
    assert ph_705_rec >= 0.95, f"Critical acidosis recall (pH < 7.05) is {ph_705_rec*100:.1f}%, must be >= 95%"

    # Explicit disk persistence assertions
    assert out_json.is_file(), f"Benchmark results JSON not created on disk: {out_json}"
    assert out_json.stat().st_size > 100, f"Benchmark JSON file is empty or truncated: {out_json}"
    assert out_csv.is_file(), f"Benchmark summary CSV not created on disk: {out_csv}"
    assert out_csv.stat().st_size > 50, f"Benchmark CSV file is empty: {out_csv}"

    # Verify JSON content schema keys
    with open(out_json, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    for required_key in ["accuracy", "macro_f1", "roc_auc", "cat_iii_recall", "ph_under_715_recall"]:
        assert required_key in bench_data, f"Benchmark JSON missing required contract key: {required_key}"

    log_pass(f"Machine-readable deliverables verified on disk: {out_json.name} ({out_json.stat().st_size} bytes) & {out_csv.name} ({out_csv.stat().st_size} bytes)")


# --------------------------------------------------------------------------------------
# STAGE 5: Telemetry Degradation Stress-Testing Harness
# --------------------------------------------------------------------------------------
def stage_5_stress_testing_harness(df_features: pd.DataFrame, models_dir: Path):
    log_header("STAGE 5: Telemetry Degradation Stress-Testing Harness (3 Scenarios)")

    out_json = models_dir / "stress_test_results.json"
    model_file = models_dir / "model_xgb.pkl"

    results = run_stress_testing_harness(
        clean_records=df_features,
        output_json_path=out_json,
        model_path=model_file if model_file.is_file() else None
    )

    scenarios = results["scenarios"]
    assert len(scenarios) >= 3, f"Expected at least 3 perturbation scenarios, got {len(scenarios)}"

    for sc_name, sc_data in scenarios.items():
        drop_pct = sc_data["dropout_pct"]
        rec = sc_data["cat_iii_recall"]
        false_reassure = sc_data["false_reassurance_count"]
        status = sc_data["resilience_status"]
        log_pass(f"Scenario [{sc_name}] ({drop_pct:.0f}% dropout): Cat III Recall={rec*100:.1f}%, False Reassurances={false_reassure} ({status})")
        assert false_reassure == 0, f"False reassurance detected in {sc_name}: {false_reassure} cases"

    log_pass(f"Guardrail Enforcement Rate: {results['guardrail_enforcement_rate']*100:.1f}%")
    assert results["false_reassurance_count"] == 0, f"False reassurance count must be 0, got {results['false_reassurance_count']}"
    assert results["zero_false_reassurance_invariant_met"] is True, "Zero false reassurance invariant violated"

    # Explicit disk persistence assertions
    assert out_json.is_file(), f"Stress testing results JSON not created on disk: {out_json}"
    assert out_json.stat().st_size > 100, f"Stress test JSON is empty: {out_json}"

    with open(out_json, "r", encoding="utf-8") as f:
        stress_data = json.load(f)
    assert "scenarios" in stress_data, "Missing 'scenarios' in stress_test_results.json"
    assert len(stress_data["scenarios"]) >= 3, "Fewer than 3 scenarios in stress_test_results.json"
    assert stress_data["zero_false_reassurance_invariant_met"] is True, "Invariant not met in saved JSON"
    assert stress_data["false_reassurance_count"] == 0, "Non-zero false reassurance in saved JSON"

    log_pass(f"Machine-readable stress test results verified on disk: {out_json.name} ({out_json.stat().st_size} bytes)")


# --------------------------------------------------------------------------------------
# STAGE 6: Programmatic Clinical Safety Guardrail Assertions
# --------------------------------------------------------------------------------------
def stage_6_guardrail_edge_case_assertions():
    log_header("STAGE 6: Programmatic Clinical Safety Guardrail Assertions")

    # Edge Case 1: Sustained Severe Bradycardia (< 100 bpm)
    case_brady = {"baseline_bpm": 88.0, "stv_ms": 4.5, "signal_loss_pct": 2.0}
    res1 = evaluate_single_window_guardrail(case_brady, raw_category="Category I")
    assert res1["final_category"] == "Category III", f"Rule 1 failed: expected Category III, got {res1['final_category']}"
    log_pass("Assertion 1 Passed: Severe Bradycardia (88 bpm) forcefully overrides to Category III")

    # Edge Case 2: Prolonged Deceleration (depth >= 30 bpm, drop < 100 bpm)
    case_prolonged = {"baseline_bpm": 120.0, "decel_count": 1, "decel_max_depth_bpm": 35.0, "decel_total_area": 3000.0}
    res2 = evaluate_single_window_guardrail(case_prolonged, raw_category="Category I")
    assert res2["final_category"] == "Category III", f"Rule 2 failed: expected Category III, got {res2['final_category']}"
    log_pass("Assertion 2 Passed: Prolonged Deceleration (depth 35 bpm, area 3000) overrides to Category III")

    # Edge Case 3: Compound Late Decel + Depressed STV (lag >= 20s, STV < 3.0 ms)
    case_late = {"baseline_bpm": 135.0, "stv_ms": 2.2, "decel_count": 2, "decel_uc_lag_sec": 28.0}
    res3 = evaluate_single_window_guardrail(case_late, raw_category="Category I")
    assert res3["final_category"] == "Category III", f"Rule 3 failed: expected Category III, got {res3['final_category']}"
    log_pass("Assertion 3 Passed: Late Decel (lag 28s) + Low STV (2.2ms) overrides to Category III")

    # Edge Case 4: Absent Baseline Variability (STV < 2.5 ms, no accels)
    case_absent = {"baseline_bpm": 140.0, "stv_ms": 1.8, "ltv_bpm": 2.0, "accel_count": 0}
    res4 = evaluate_single_window_guardrail(case_absent, raw_category="Category I")
    assert res4["final_category"] == "Category III", f"Rule 4 failed: expected Category III, got {res4['final_category']}"
    log_pass("Assertion 4 Passed: Absent Variability (STV 1.8ms, LTV 2.0bpm) overrides to Category III")

    # Edge Case 5: Sinusoidal FHR Rhythm Pattern
    case_sinusoidal = {"baseline_bpm": 130.0, "stv_ms": 1.5, "ltv_bpm": 8.0, "accel_count": 0, "decel_count": 0}
    res5 = evaluate_single_window_guardrail(case_sinusoidal, raw_category="Category I")
    assert res5["final_category"] == "Category III", f"Rule 5 failed: expected Category III, got {res5['final_category']}"
    log_pass("Assertion 5 Passed: Sinusoidal Rhythm Pattern overrides to Category III")

    # Edge Case 6: Telemetry Signal Integrity Loss (> 40% signal loss)
    case_detached = {"baseline_bpm": 138.0, "signal_loss_pct": 52.0}
    res6 = evaluate_single_window_guardrail(case_detached, raw_category="Category I")
    assert res6["final_category"] != "Category I", f"Rule 6 failed: Category I allowed under 52% signal loss"
    log_pass("Assertion 6 Passed: High Signal Loss (52%) strictly suppresses reassuring Category I")

    # Edge Case 7: Intermediate Telemetry Dropout (> 25% signal loss)
    case_intermediate = {"baseline_bpm": 138.0, "signal_loss_pct": 28.0}
    res7 = evaluate_single_window_guardrail(case_intermediate, raw_category="Category I")
    assert res7["final_category"] != "Category I", f"Rule 6 failed: Category I allowed under 28% signal loss"
    assert res7["final_category"] == "Category II", f"Expected Category II under intermediate signal loss, got {res7['final_category']}"
    log_pass("Assertion 7 Passed: Intermediate Signal Loss (28%) strictly suppresses Category I and elevates to Category II")


def main():
    parser = argparse.ArgumentParser(description="NexoGuard AI Automated Non-Interactive Validation Suite")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to data directory")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR), help="Path to models artifact directory")
    args = parser.parse_args()

    start_time = time.time()
    print(f"\n{TerminalColors.BOLD}{TerminalColors.HEADER}{'='*80}{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.HEADER}    NEXOGUARD AI — AUTOMATED PREDICTIVE MODELING & VALIDATION SUITE            {TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.HEADER}{'='*80}{TerminalColors.ENDC}")
    print(f"  Execution Mode: Non-Interactive Headless CLI Runner")
    print(f"  Target Workspace: {PROJECT_ROOT}")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Models Directory: {args.models_dir}")

    try:
        data_path = Path(args.data_dir)
        models_path = Path(args.models_dir)

        # Execute 6 Stages in Sequential Order
        df_outcomes, df_features = stage_1_preflight_checks(data_path)
        stage_2_signal_and_feature_pipeline(data_path, df_features)
        stage_3_stratification_and_inference(df_features, models_path)
        stage_4_clinical_benchmarking(data_path, models_path)
        stage_5_stress_testing_harness(df_features, models_path)
        stage_6_guardrail_edge_case_assertions()

        elapsed = time.time() - start_time
        print(f"\n{TerminalColors.BOLD}{TerminalColors.OKGREEN}{'='*80}{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.OKGREEN}*** ALL 6 VALIDATION STAGES COMPLETED SUCCESSFULLY in {elapsed:.2f}s ***{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.OKGREEN}*** Zero False Reassurance Invariant Verified. Exiting with Status Code 0. ***{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.OKGREEN}{'='*80}{TerminalColors.ENDC}\n")
        sys.exit(0)

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{TerminalColors.BOLD}{TerminalColors.FAIL}{'='*80}{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.FAIL}[X] PIPELINE EXECUTION FAILED after {elapsed:.2f}s: {e}{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.FAIL}{'='*80}{TerminalColors.ENDC}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
