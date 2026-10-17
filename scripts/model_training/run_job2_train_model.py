import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import shap

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "baseline_bpm", "stv_ms", "ltv_bpm", "accel_count", "decel_count",
    "decel_max_depth_bpm", "decel_total_area", "contraction_count",
    "decel_uc_lag_sec", "mhr_fhr_cross_corr", "signal_loss_pct"
]

def train_and_evaluate():
    csv_path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run Job 1 first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print("Class counts:\n", df["label"].value_counts())

    X = df[FEATURE_COLS]
    y = df["label"].to_numpy()
    groups = df["patient_id"].to_numpy() if "patient_id" in df.columns else np.arange(len(df))

    # Asymmetric cost weights (Penalize missing Category III heavily for fetal safety)
    sample_weights = np.array([1.0 if label == 0 else (2.0 if label == 1 else 10.0) for label in y])

    # 5-Fold Stratified Group K-Fold to prevent patient data leakage
    sgkf = StratifiedGroupKFold(n_splits=min(5, len(np.unique(groups))))
    
    oof_preds = np.zeros(len(df))
    oof_probs = np.zeros((len(df), 3))

    best_model = None

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        w_train = sample_weights[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        clf = XGBClassifier(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
            random_state=42 + fold,
            eval_metric="mlogloss"
        )
        clf.fit(X_train, y_train, sample_weight=w_train)

        probs = clf.predict_proba(X_val)
        oof_probs[val_idx] = probs
        oof_preds[val_idx] = np.argmax(probs, axis=1)
        best_model = clf

    print("\n================ OUT-OF-FOLD EVALUATION REPORT ================")
    report_dict = classification_report(y, oof_preds, target_names=["Normal (Cat I)", "Suspicious (Cat II)", "Pathological (Cat III)"], output_dict=True)
    print(classification_report(y, oof_preds, target_names=["Normal (Cat I)", "Suspicious (Cat II)", "Pathological (Cat III)"]))
    
    cm = confusion_matrix(y, oof_preds)
    print("Confusion Matrix:\n", cm)

    cat3_recall = report_dict["Pathological (Cat III)"]["recall"]
    print(f"\n[METRIC] Pathological / Fetal Acidemia Recall: {cat3_recall*100:.1f}%")
    print(f"[METRIC] NexoPrima baseline paper had 85% decel recall. Our Hybrid AI reaches {cat3_recall*100:.1f}%!")

    # Train final production model on full dataset
    print("\nTraining final production model on full cohort...")
    final_clf = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.06,
        objective="multi:softprob",
        num_class=3,
        random_state=42
    )
    final_clf.fit(X, y, sample_weight=sample_weights)

    # Fit TreeExplainer
    print("Fitting TreeSHAP explainer for real-time doctor interpretability...")
    explainer = shap.TreeExplainer(final_clf)

    # Save model package
    model_pkg = {
        "model": final_clf,
        "explainer": explainer,
        "feature_cols": FEATURE_COLS,
        "metrics": {
            "cat3_recall": cat3_recall,
            "overall_accuracy": report_dict["accuracy"],
            "confusion_matrix": cm.tolist()
        }
    }
    model_path = os.path.join(MODEL_DIR, "model_xgb.pkl")
    joblib.dump(model_pkg, model_path)
    print(f"\n[JOB 2 COMPLETE] Saved production model and SHAP explainer to: {model_path}")

    # Save summary report for Person 4's pitch deck
    with open(os.path.join(MODEL_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

if __name__ == "__main__":
    train_and_evaluate()
