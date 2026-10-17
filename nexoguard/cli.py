"""
nexoguard.cli
Interactive and CLI patient-level validation runner for NexoGuard AI.
Supports testing PhysioNet CTU-UHB patient records (.dat/.hea), custom CSV files, or predefined FIGO test cases.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from nexoguard.signal.reader import read_wfdb_record, parse_header_file
from nexoguard.signal.preprocessor import clean_fhr_signal, calculate_signal_loss_pct
from nexoguard.features.extractor import extract_features_from_signal
from nexoguard.models.inference import predict_risk
from nexoguard.guardrails.clinical_rules import apply_safety_guardrails


def list_available_patients(data_dir: Optional[Path] = None):
    """Lists all available patient records in data/ with clinical ground truth."""
    target_data = data_dir or DATA_DIR
    outcomes_file = target_data / "clinical_outcomes.csv"
    if outcomes_file.is_file():
        df_out = pd.read_csv(outcomes_file)
        print("\n" + "=" * 80)
        print(f"AVAILABLE PATIENTS IN DATASET (Total: {len(df_out)})")
        print("=" * 80)
        print(f"{'Patient ID':<12} | {'pH':<6} | {'BDecf':<8} | {'Ground Truth Category':<22} | {'Apgar1':<8} | {'Delivery':<12}")
        print("-" * 80)
        for _, row in df_out.iterrows():
            pid = str(row.get("rec_id", row.get("patient_id", "Unknown")))
            ph = f"{row.get('pH', np.nan):.2f}" if pd.notna(row.get("pH")) else "N/A"
            bd = f"{row.get('BDecf', np.nan):.1f}" if pd.notna(row.get("BDecf")) else "N/A"
            try:
                ph_val = float(row.get("pH", 7.30))
                if ph_val < 7.15:
                    cat = "Pathological (Cat III)"
                elif ph_val <= 7.20:
                    cat = "Suspicious (Cat II)"
                else:
                    cat = "Normal (Cat I)"
            except Exception:
                cat = "Unknown"
            apg = str(row.get("Apgar1", "N/A"))
            deliv = str(row.get("delivery", row.get("delivery_type", "N/A")))
            print(f"{pid:<12} | {ph:<6} | {bd:<8} | {cat:<22} | {apg:<8} | {deliv:<12}")
    else:
        dat_files = sorted(target_data.glob("*.dat"))
        print("\n" + "=" * 80)
        print(f"AVAILABLE PATIENT DATA FILES (Total: {len(dat_files)})")
        print("=" * 80)
        for f in dat_files:
            print(f" - {f.stem}")


def analyze_patient(record_id_or_path: str, data_dir: Optional[Path] = None, models_dir: Optional[Path] = None):
    """Runs complete NexoGuard AI ingestion, extraction, ML inference, and guardrails for a patient."""
    target_data = data_dir or DATA_DIR
    target_models = models_dir or MODELS_DIR

    print("\n" + "=" * 80)
    print(f"  NEXOGUARD AI — CLINICAL PATIENT ANALYSIS")
    print(f"  Target: {record_id_or_path}")
    print("=" * 80)

    # 1. Ingest signal
    try:
        record = read_wfdb_record(record_id_or_path, data_dir=target_data)
    except Exception as e:
        print(f"[ERROR] Failed to load recording: {e}")
        return

    duration_min = record.duration_sec / 60.0
    loss_pct = calculate_signal_loss_pct(record.fhr)
    fhr_clean = clean_fhr_signal(record.fhr, fs=record.fs)

    print(f"\n[1] SIGNAL TELEMETRY ACQUISITION:")
    print(f"    - Patient ID:           {record.rec_id}")
    print(f"    - Sampling Frequency:   {record.fs} Hz (Dawes-Redman 4Hz CTG standard)")
    print(f"    - Total Duration:       {duration_min:.1f} minutes ({len(record.fhr)} samples)")
    print(f"    - FHR Dynamic Range:    {np.min(fhr_clean):.1f} - {np.max(fhr_clean):.1f} bpm")
    print(f"    - Overall Signal Loss:  {loss_pct:.2f}% (dropouts / sensor wander)")

    # Display clinical header metadata if available
    hea_path = target_data / f"{record.rec_id}.hea"
    if hea_path.is_file():
        _, _, meta = parse_header_file(hea_path)
        if meta:
            print(f"\n[2] CLINICAL GROUND TRUTH (from PhysioNet .hea):")
            for k in ["pH", "BDecf", "BE", "Apgar1", "Apgar5", "Gest. weeks", "Weight", "Delivery"]:
                if k in meta:
                    print(f"    - {k:<18}: {meta[k]}")

    # 2. Extract 10-Minute Sliding Windows
    print(f"\n[3] CLINICAL FEATURE EXTRACTION (10-minute sliding windows):")
    windows_df = extract_features_from_signal(
        record.fhr, record.uc, fs=record.fs, window_len_sec=600, step_sec=300, patient_id=record.rec_id
    )

    if windows_df.empty:
        print("    [WARNING] No valid windows extracted (recording may be too short or excessively corrupted).")
        return

    print(f"    - Extracted {len(windows_df)} sliding 10-minute analysis windows.")

    # 3. Model Inference (XGBoost)
    model_path = target_models / "model_xgb.pkl"
    if not model_path.is_file():
        print(f"    [ERROR] Trained model artifact not found at {model_path}.")
        return

    pred_df = predict_risk(windows_df, model_path=model_path)

    # 4. Deterministic Clinical Safety Guardrails
    guarded_df = apply_safety_guardrails(pred_df, windows_df)

    # 5. Display Window-by-Window Clinical Summary
    print(f"\n[4] WINDOW-BY-WINDOW CLINICAL TRIAGE TIMELINE:")
    print("-" * 105)
    print(f"{'Win':<4} | {'FHR Base':<9} | {'STV (ms)':<9} | {'LTV':<6} | {'Acc':<4} | {'Dec':<4} | {'Lag(s)':<7} | {'ML Cat':<10} | {'Final FIGO':<14} | {'Guardrail Overrides'}")
    print("-" * 105)

    category_counts = {"Category I": 0, "Category II": 0, "Category III": 0}

    for idx, row in guarded_df.iterrows():
        win = str(row["window_id"])
        base = f"{row['baseline_bpm']:.1f}"
        stv = f"{row['stv_ms']:.2f}"
        ltv = f"{row['ltv_bpm']:.1f}"
        acc = int(row["accel_count"])
        dec = int(row["decel_count"])
        lag = f"{row['decel_uc_lag_sec']:.0f}"
        ml_cat = str(row["pred_category_raw"])
        final_cat = str(row["final_category"])
        reasons = str(row.get("guardrail_reason", ""))
        if reasons == "Normal physiological trace" or not reasons:
            reasons_display = "--"
        else:
            reasons_display = reasons[:35] + ("..." if len(reasons) > 35 else "")

        category_counts[final_cat] = category_counts.get(final_cat, 0) + 1

        print(f"{win:<4} | {base:<9} | {stv:<9} | {ltv:<6} | {acc:<4} | {dec:<4} | {lag:<7} | {ml_cat:<10} | {final_cat:<14} | {reasons_display}")

    print("-" * 105)

    # 6. Overall Assessment
    max_risk = guarded_df["risk_score"].max()
    most_severe = "Category I"
    if category_counts["Category III"] > 0:
        most_severe = "Category III (Pathological)"
    elif category_counts["Category II"] > 0:
        most_severe = "Category II (Suspicious)"
    else:
        most_severe = "Category I (Normal)"

    print(f"\n[5] PATIENT SUMMARY & CLINICAL ACTION PLAN:")
    print(f"    - Most Severe FIGO Status: {most_severe}")
    print(f"    - Maximum Risk Score:     {max_risk:.1%}")
    print(f"    - Category Breakdown:      Cat I: {category_counts['Category I']}, Cat II: {category_counts['Category II']}, Cat III: {category_counts['Category III']}")

    if "Category III" in most_severe:
        print(f"    >>> CLINICAL ALERT: IMMEDIATE OBSTETRIC ACTION RECOMMENDED")
        print(f"        Persistent decelerations or severe loss of variability detected.")
        print(f"        Verify fetal scalp blood sampling, maternal oxygenation, or prepare for emergency operative delivery.")
    elif "Category II" in most_severe:
        print(f"    >>> CLINICAL ALERT: HEIGHTENED VIGILANCE REQUIRED (SUSPICIOUS)")
        print(f"        Conservative corrective measures: lateral maternal positioning, hydration, re-evaluate within 15-30 min.")
    else:
        print(f"    >>> CLINICAL STATUS: REASSURING (NORMAL)")
        print(f"        Normal baseline, reassuring STV/LTV variability, continue routine monitoring.")

    print("=" * 80 + "\n")


def run_predefined_case(case_num: int):
    """Executes a specific numbered predefined test case."""
    test_case_map = {
        1: ("1024", "TEST CASE 1: Textbook Healthy Reassuring FHR (pH 7.43, BD 0.9 mmol/L)"),
        2: ("1012", "TEST CASE 2: Normal Baseline with Physiological Reactivity (pH 7.36, BD 0.8 mmol/L)"),
        3: ("1006", "TEST CASE 3: Suspicious Indeterminate / Minimal Variability (pH 7.23, BD 1.3 mmol/L)"),
        4: ("1001", "TEST CASE 4: Pathological Recurrent Late Decelerations (pH 7.14, BD 8.1 mmol/L)"),
        5: ("1029", "TEST CASE 5: Critical Terminal Acidemia / Severe Distress (pH 6.97, BD 19.4 mmol/L)"),
    }
    if case_num in test_case_map:
        pid, title = test_case_map[case_num]
        print(f"\n{'*'*80}\n  LAUNCHING {title}\n{'*'*80}")
        analyze_patient(pid)
    elif case_num == 6:
        print(f"\n{'*'*80}\n  LAUNCHING TEST CASE 6: Telemetry Dropout Edge Case (Sensor Detachment Guardrail)\n{'*'*80}")
        from tests.test_predefined_cases import TestPredefinedClinicalCases
        t = TestPredefinedClinicalCases()
        t.setUpClass()
        t.test_case_6_sensor_detachment_guardrail_suppression()
    elif case_num == 7:
        print(f"\n{'*'*80}\n  LAUNCHING TEST CASE 7: Verification of Custom Upload Files across FIGO Categories\n{'*'*80}")
        from tests.test_predefined_cases import TestPredefinedClinicalCases
        t = TestPredefinedClinicalCases()
        t.setUpClass()
        t.test_case_7_custom_sample_upload_files()
    else:
        print(f"[ERROR] Invalid test case number: {case_num}. Choose between 1 and 7.")


def main():
    parser = argparse.ArgumentParser(description="NexoGuard AI — Clinical Patient Testing CLI")
    parser.add_argument("--test-case", type=int, choices=[1, 2, 3, 4, 5, 6, 7], help="Predefined test case number (1-7)")
    parser.add_argument("--patient", type=str, help="Patient ID to analyze (e.g. 1012, 1001)")
    parser.add_argument("--file", type=str, help="Path to custom .csv / .dat / .hea file (e.g. data/sample_uploads/sample_category_iii_pathological.csv)")
    parser.add_argument("--list", action="store_true", help="List all available patient records in the dataset")

    args = parser.parse_args()

    if args.test_case:
        run_predefined_case(args.test_case)
    elif args.list:
        list_available_patients()
    elif args.patient:
        analyze_patient(args.patient)
    elif args.file:
        analyze_patient(args.file)
    else:
        print("No patient or test case specified. Running default demonstration on TEST CASE 2 (Patient 1012):")
        run_predefined_case(2)


if __name__ == "__main__":
    main()
