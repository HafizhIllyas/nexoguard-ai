"""
nexoguard.guardrails.clinical_rules
Deterministic FIGO 2015 Clinical Safety Guardrails implementing the Zero False Reassurance invariant.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

RULE_SEVERE_BRADYCARDIA = "RULE_1_SEVERE_BRADYCARDIA"
RULE_PROLONGED_DECEL = "RULE_2_PROLONGED_DECEL"
RULE_LATE_DECEL_LOW_STV = "RULE_3_LATE_DECEL_LOW_STV"
RULE_ABSENT_VARIABILITY = "RULE_4_ABSENT_VARIABILITY"
RULE_SINUSOIDAL_PATTERN = "RULE_5_SINUSOIDAL_PATTERN"
RULE_SIGNAL_INTEGRITY = "RULE_6_SIGNAL_INTEGRITY"

ALL_RULES = [
    RULE_SEVERE_BRADYCARDIA,
    RULE_PROLONGED_DECEL,
    RULE_LATE_DECEL_LOW_STV,
    RULE_ABSENT_VARIABILITY,
    RULE_SINUSOIDAL_PATTERN,
    RULE_SIGNAL_INTEGRITY
]


def evaluate_single_window_guardrail(
    feature_dict: Dict[str, Any],
    raw_category: str = "Category I",
    raw_risk_score: float = 0.10
) -> Dict[str, Any]:
    """
    Evaluates a single window's feature dictionary against the 6 deterministic safety rules.
    Takes precedence over ML probabilistic inference to guarantee zero false reassurance.
    """
    # Normalize raw_category if numeric
    if str(raw_category) in ("0", "Category I"):
        norm_raw_category = "Category I"
    elif str(raw_category) in ("1", "Category II"):
        norm_raw_category = "Category II"
    elif str(raw_category) in ("2", "Category III"):
        norm_raw_category = "Category III"
    else:
        norm_raw_category = str(raw_category)

    baseline_bpm = float(feature_dict.get("baseline_bpm", 140.0))
    stv_ms = float(feature_dict.get("stv_ms", 5.0))
    ltv_bpm = float(feature_dict.get("ltv_bpm", 10.0))
    accel_count = int(feature_dict.get("accel_count", 0))
    decel_count = int(feature_dict.get("decel_count", 0))
    decel_depth = float(feature_dict.get("decel_max_depth_bpm", 0.0))
    decel_area = float(feature_dict.get("decel_total_area", 0.0))
    lag_sec = float(feature_dict.get("decel_uc_lag_sec", 0.0))
    signal_loss_pct = float(feature_dict.get("signal_loss_pct", 0.0))
    mhr_corr = float(feature_dict.get("mhr_fhr_cross_corr", 0.0))
    is_sinusoidal = bool(feature_dict.get("is_sinusoidal", False))
    window_id = str(feature_dict.get("window_id", "UNKNOWN"))

    triggered_rules: List[str] = []
    clinical_reasons: List[str] = []
    force_category_iii = False
    suppress_category_i = False
    alert_level = "REASSURING"
    final_risk = float(raw_risk_score)

    # Rule 1: Sustained Severe Bradycardia (< 100 bpm)
    if baseline_bpm < 100.0:
        triggered_rules.append(RULE_SEVERE_BRADYCARDIA)
        clinical_reasons.append("EMERGENCY: Sustained Severe Bradycardia (<100 bpm). High risk of acute fetal asphyxia.")
        force_category_iii = True
        final_risk = max(final_risk, 0.98)
        alert_level = "EMERGENCY"

    # Rule 2: Prolonged Deceleration (depth >= 30 bpm and area >= 3000 bpm·s exceeding 3-minute threshold)
    if decel_count >= 1 and decel_depth >= 30.0 and decel_area >= 3000.0:
        triggered_rules.append(RULE_PROLONGED_DECEL)
        clinical_reasons.append("EMERGENCY: Prolonged Deceleration exceeding 3-minute threshold. Prepare for expedited operative delivery.")
        force_category_iii = True
        final_risk = max(final_risk, 0.99)
        alert_level = "EMERGENCY"

    # Rule 3: Late Deceleration + Depressed Acidemic STV (lag >= 20s, STV < 3.0 ms, decel present)
    if lag_sec >= 20.0 and stv_ms < 3.0 and decel_count >= 1:
        triggered_rules.append(RULE_LATE_DECEL_LOW_STV)
        clinical_reasons.append("PATHOLOGICAL: Late Decelerations (Lag >= 20s) with Depressed STV (<3.0ms). Severe metabolic acidemia alert.")
        force_category_iii = True
        final_risk = max(final_risk, 0.96)
        if alert_level != "EMERGENCY":
            alert_level = "PATHOLOGICAL"

    # Rule 4: Sustained Absent Baseline Variability (STV < 2.0 ms AND LTV < 4.0 bpm without accels)
    if (stv_ms < 2.0 and ltv_bpm < 4.0) and accel_count == 0:
        triggered_rules.append(RULE_ABSENT_VARIABILITY)
        clinical_reasons.append("CRITICAL: Absent baseline variability (STV < 2.0ms and LTV < 4.0 bpm). Autonomic tone suppressed.")
        force_category_iii = True
        final_risk = max(final_risk, 0.92)
        if alert_level not in ("EMERGENCY", "PATHOLOGICAL"):
            alert_level = "CRITICAL"

    # Rule 5: Sinusoidal Rhythm Pattern (5-15 bpm amplitude, STV < 2.0 ms, no accels/decels)
    if is_sinusoidal or (5.0 <= ltv_bpm <= 15.0 and stv_ms < 2.0 and accel_count == 0 and decel_count == 0):
        triggered_rules.append(RULE_SINUSOIDAL_PATTERN)
        clinical_reasons.append("EMERGENCY: Sinusoidal FHR pattern detected. Severe fetal anemia or terminal asphyxia suspected.")
        force_category_iii = True
        final_risk = max(final_risk, 0.99)
        alert_level = "EMERGENCY"

    # Rule 6: Telemetry Integrity Loss (> 25% loss or MHR cross-correlation > 0.80)
    if signal_loss_pct > 25.0 or mhr_corr > 0.80:
        triggered_rules.append(RULE_SIGNAL_INTEGRITY)
        clinical_reasons.append(
            f"DATA INTEGRITY ALERT: Telemetry signal loss ({signal_loss_pct:.1f}%) > 25% or MHR cross-correlation > 0.80. Reassuring status suppressed."
        )
        suppress_category_i = True
        final_risk = max(final_risk, 0.65)
        if alert_level == "REASSURING":
            alert_level = "DATA_INTEGRITY_ALERT"

    # Resolve Final Category
    if force_category_iii:
        final_category = "Category III"
    elif suppress_category_i:
        final_category = "Category II" if norm_raw_category == "Category I" else norm_raw_category
    else:
        final_category = norm_raw_category

    # Invariant Verification: Zero False Reassurance clamp
    zero_false_reassurance_verified = True
    if (force_category_iii or suppress_category_i) and final_category == "Category I":
        zero_false_reassurance_verified = False
        final_category = "Category III"  # Absolute fail-safe clamp

    return {
        "window_id": window_id,
        "raw_category": norm_raw_category,
        "final_category": final_category,
        "raw_risk_score": float(raw_risk_score),
        "final_risk_score": float(final_risk),
        "guardrail_triggered": len(triggered_rules) > 0,
        "triggered_rules": triggered_rules,
        "clinical_reasons": clinical_reasons,
        "alert_level": alert_level,
        "zero_false_reassurance_verified": zero_false_reassurance_verified
    }


def apply_safety_guardrails(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Public Shared Contract (PROJECT.md:98-109):
    apply_safety_guardrails(predictions_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame

    Outputs:
      pd.DataFrame with columns:
      ['final_category', 'guardrail_triggered', 'guardrail_reason', 'clinical_alert_level', 'risk_score']

    Contract Invariants:
      - If severe bradycardia (<100 bpm), prolonged decel (>3 min), or late decel with STV <3.0 ms
        is present, final_category MUST be Category III.
      - If signal loss > 40%, final_category CANNOT be Category I.
    """
    merged = features_df.copy()
    for col in predictions_df.columns:
        if col not in merged.columns:
            merged[col] = predictions_df[col].values

    results = []
    for _, row in merged.iterrows():
        feat = row.to_dict()
        raw_cat = str(feat.get("pred_category_raw", feat.get("pred_category", "Category I")))
        raw_risk = float(feat.get("risk_score", feat.get("prob_cat_iii", 0.10)))
        res = evaluate_single_window_guardrail(feat, raw_category=raw_cat, raw_risk_score=raw_risk)
        results.append(res)

    res_df = pd.DataFrame(results)
    output_df = predictions_df.copy()
    output_df["final_category"] = res_df["final_category"].values
    output_df["guardrail_triggered"] = res_df["guardrail_triggered"].values
    output_df["guardrail_reason"] = res_df["clinical_reasons"].apply(
        lambda l: "; ".join(l) if l else "Normal physiological trace"
    )
    output_df["clinical_alert_level"] = res_df["alert_level"].values
    output_df["risk_score"] = res_df["final_risk_score"].values
    output_df["triggered_rules"] = res_df["triggered_rules"].values

    return output_df
