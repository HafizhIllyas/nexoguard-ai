# ROLE GUIDE — PERSON 2: ML & EXPLAINABILITY (XAI) LEAD

**Title**: Lead Machine Learning Engineer & Clinical AI Modeler  
**Core Mission**: Train the ensemble risk classifier (`XGBoost` / `RandomForest`), enforce asymmetric cost penalties to eliminate missed fetal decelerations, implement FIGO deterministic safety guardrails, and extract SHAP clinical explainability factors.  
**Tech Stack**: Python (`scikit-learn`, `xgboost`, `shap`, `joblib`)

---

## 1. Quick Checklist & Deliverables

- [ ] **Hour 0–2**: Create a synthetic dataset generator so you can build and test your model training pipeline *before* Person 1 finishes data cleaning.
- [ ] **Hour 2–6**: Build the Stratified Cross-Validation harness (grouped by patient to prevent data leakage).
- [ ] **Hour 6–10**: Train `XGBClassifier` and tune asymmetric class weights (`scale_pos_weight` / `class_weight`) to force **Deceleration/Hypoxia Recall > 95%** (fixing NexoPrima's 85% miss rate).
- [ ] **Hour 10–14**: Write the **Deterministic FIGO Safety Guardrail** wrapper (hard-coded safety overrides for critical physiological markers).
- [ ] **Hour 14–18**: Integrate `shap.TreeExplainer` to produce instant doctor attribution cards (e.g. `"+38% Risk: Late Deceleration Lag"`).
- [ ] **Hour 18+**: Export `model_xgb.pkl` and deliver the clean `predict_window(features_dict)` function to Person 3.

---

## 2. Complete Starter Code: `model_pipeline.py`

Save this file in your project folder as `model_pipeline.py`:

```python
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from xgboost import XGBClassifier
import shap

# Feature columns according to shared contract (excluding window_id and label)
FEATURE_COLS = [
    "baseline_bpm", "stv_ms", "ltv_bpm", "accel_count", "decel_count",
    "decel_max_depth_bpm", "decel_total_area", "contraction_count",
    "decel_uc_lag_sec", "mhr_fhr_cross_corr", "signal_loss_pct"
]

def generate_mock_training_data(n_samples=500):
    """
    Generates realistic synthetic data matching Person 1's features.csv schema
    so you can build and test the entire training pipeline immediately.
    """
    np.random.seed(42)
    data = []
    for i in range(n_samples):
        # 70% Normal, 20% Suspicious, 10% Pathological
        cls = np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10])
        
        if cls == 0: # Normal
            baseline = np.random.normal(138, 8)
            stv = np.random.uniform(3.8, 7.5)
            ltv = np.random.uniform(10, 22)
            accel = np.random.poisson(3)
            decel = 0
            decel_depth = 0.0
            decel_area = 0.0
            contractions = np.random.poisson(3)
            lag = 0.0
        elif cls == 1: # Suspicious
            baseline = np.random.choice([np.random.uniform(105, 110), np.random.uniform(160, 175)])
            stv = np.random.uniform(3.0, 3.8)
            ltv = np.random.uniform(4, 9)
            accel = np.random.poisson(1)
            decel = np.random.choice([0, 1])
            decel_depth = np.random.uniform(15, 25) if decel > 0 else 0.0
            decel_area = decel_depth * 15
            contractions = np.random.poisson(4)
            lag = np.random.uniform(-5, 15) # early/benign decel
        else: # Pathological
            baseline = np.random.choice([np.random.uniform(90, 105), np.random.uniform(120, 150)])
            stv = np.random.uniform(1.5, 2.9) # critical acidemia marker
            ltv = np.random.uniform(2, 6)
            accel = 0
            decel = np.random.randint(1, 4)
            decel_depth = np.random.uniform(25, 45)
            decel_area = decel_depth * 40
            contractions = np.random.poisson(5)
            lag = np.random.uniform(22, 45) # LATE DECELERATION!
            
        corr = np.random.uniform(0.05, 0.35)
        loss = np.random.uniform(0.5, 4.0)
        
        data.append({
            "window_id": f"PAT_{i:03d}",
            "baseline_bpm": round(baseline, 2),
            "stv_ms": round(stv, 2),
            "ltv_bpm": round(ltv, 2),
            "accel_count": accel,
            "decel_count": decel,
            "decel_max_depth_bpm": round(decel_depth, 2),
            "decel_total_area": round(decel_area, 2),
            "contraction_count": contractions,
            "decel_uc_lag_sec": round(lag, 2),
            "mhr_fhr_cross_corr": round(corr, 2),
            "signal_loss_pct": round(loss, 2),
            "label": cls
        })
    return pd.DataFrame(data)

class NexoGuardModel:
    def __init__(self):
        # Multi-class XGBoost with max_depth=3 to prevent overfitting on tabular data
        self.clf = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            objective="multi:softprob",
            num_class=3,
            random_state=42
        )
        self.explainer = None

    def train(self, df):
        X = df[FEATURE_COLS]
        y = df["label"]
        
        # Heavy class weight penalty for missing Category II / III
        sample_weights = y.map({0: 1.0, 1: 3.0, 2: 8.0}).to_numpy()
        
        print("Training XGBoost with Asymmetric Cost Weights...")
        self.clf.fit(X, y, sample_weight=sample_weights)
        
        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.clf)
        print("Model trained and TreeExplainer fitted successfully!")

    def predict_window(self, feature_dict):
        """
        The production inference API for Person 3's UI.
        Incorporates FIGO Safety Guardrail overrides + TreeSHAP explainability.
        """
        # 1. FIGO DETERMINISTIC SAFETY GUARDRAIL (Fail-Safe Override)
        # Even if the ML model is uncertain, critical physiological red flags trigger emergency alerts!
        is_late_decel = feature_dict.get("decel_uc_lag_sec", 0) > 20.0 and feature_dict.get("decel_count", 0) >= 1
        is_acidemia_stv = feature_dict.get("stv_ms", 5.0) < 3.0
        is_severe_brady = feature_dict.get("baseline_bpm", 140) < 100.0

        if (is_late_decel and is_acidemia_stv) or is_severe_brady:
            return {
                "category": "Category III",
                "risk_score": 0.95,
                "status": "Pathological (Safety Override)",
                "color": "rose",
                "override_triggered": True,
                "top_factors": [
                    "⚠️ Late Deceleration Lag > 20s (Cord/Placental Hypoxia)",
                    "⚠️ Short-Term Variability < 3.0ms (Metabolic Acidemia)",
                    "⚡ Forced Category III Safety Guardrail Active"
                ]
            }

        # 2. MACHINE LEARNING PREDICTION
        X_sample = pd.DataFrame([feature_dict])[FEATURE_COLS]
        probs = self.clf.predict_proba(X_sample)[0]
        pred_class = int(np.argmax(probs))
        
        categories = ["Category I", "Category II", "Category III"]
        statuses = ["Normal (Reassuring)", "Suspicious (Indeterminate)", "Pathological (Urgent)"]
        colors = ["emerald", "amber", "rose"]
        
        # 3. SHAP EXPLAINABILITY ATTRIBUTION
        shap_values = self.explainer.shap_values(X_sample)
        # For multi-class, shap_values has shape (classes, samples, features) or list
        class_shap = shap_values[pred_class][0] if isinstance(shap_values, list) else shap_values[0]
        
        # Find top 3 features pushing toward this classification
        top_indices = np.argsort(np.abs(class_shap))[-3:][::-1]
        top_factors = []
        for idx in top_indices:
            feat_name = FEATURE_COLS[idx]
            val = feature_dict[feat_name]
            impact = class_shap[idx]
            sign = "+" if impact > 0 else "-"
            top_factors.append(f"{feat_name} = {val} ({sign}{abs(impact):.2f} impact)")
            
        return {
            "category": categories[pred_class],
            "risk_score": round(float(probs[2]), 3), # Probability of Category III
            "status": statuses[pred_class],
            "color": colors[pred_class],
            "override_triggered": False,
            "top_factors": top_factors,
            "probabilities": {
                "Category_I": round(float(probs[0]), 3),
                "Category_II": round(float(probs[1]), 3),
                "Category_III": round(float(probs[2]), 3)
            }
        }

    def save(self, filepath="model_xgb.pkl"):
        joblib.dump({"model": self.clf, "explainer": self.explainer}, filepath)
        print(f"Model saved to {filepath}")

if __name__ == "__main__":
    df_train = generate_mock_training_data(600)
    model = NexoGuardModel()
    model.train(df_train)
    model.save("model_xgb.pkl")
    
    # Test high-risk sample
    sample_risk = {
        "baseline_bpm": 122.0, "stv_ms": 2.4, "ltv_bpm": 4.5, "accel_count": 0,
        "decel_count": 2, "decel_max_depth_bpm": 32.0, "decel_total_area": 450.0,
        "contraction_count": 4, "decel_uc_lag_sec": 34.0, "mhr_fhr_cross_corr": 0.12,
        "signal_loss_pct": 1.5
    }
    result = model.predict_window(sample_risk)
    print("\nInference Output Test:\n", result)
```

---

## 3. What to Hand Off & Coordinate

1. **For Person 3 (App Lead)**: 
   * Deliver `model_xgb.pkl` and `predict_window()`.
   * Person 3 simply loads the model and calls `model.predict_window(window_features)` whenever the doctor scrubs the timeline!
2. **For Person 4 (Pitch Lead)**:
   * Provide the validation metrics: **Deceleration Recall (aim for >95%)**, **Accuracy**, and sample SHAP graphs to place into the pitch deck.

---

## 4. Stuck? How to Ask AI

Copy the content of `guides/PROJECT_MASTER_CONTEXT_FOR_AI.md` into your AI prompt, and add:
```
I am Person 2 (ML Lead). 
I am currently working on [e.g. tuning class weights in XGBoost / calculating SHAP values].
Here is my current code: [paste code]
The issue is: [paste error or question]
Please optimize this to maximize sensitivity on Category III while maintaining strict compatibility with FEATURE_COLS.
```
