"""
nexoguard.api.service
Core service logic for NexoGuard API layer.
Decouples clinical data loading, signal preprocessing, ML inference, and FIGO guardrails.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from nexoguard.api.errors import PatientNotFoundError, InsufficientDataError, ValidationError
from nexoguard.api.schemas import (
    PatientSummary,
    PatientDetailResponse,
    StreamResponse,
    PredictResponse,
    FileUploadResponse
)
from nexoguard.signal.reader import read_wfdb_record
from nexoguard.signal.preprocessor import clean_fhr_signal, calculate_signal_loss_pct
from nexoguard.features.extractor import extract_features_from_signal, CONTRACT_FEATURE_COLS
from nexoguard.guardrails.clinical_rules import apply_safety_guardrails
from nexoguard.models.inference import predict_risk

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class NexoGuardService:
    def __init__(self, data_dir: Optional[Path] = None, models_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.models_dir = models_dir or MODELS_DIR
        self._patients_cache: Optional[pd.DataFrame] = None
        self._streams_cache: Optional[Dict[str, Any]] = None

    def get_patients_df(self) -> pd.DataFrame:
        if self._patients_cache is None:
            outcomes_file = self.data_dir / "clinical_outcomes.csv"
            if not outcomes_file.exists():
                raise FileNotFoundError(f"Missing clinical outcomes dataset: {outcomes_file}")
            df = pd.read_csv(outcomes_file)
            
            if "rec_id" in df.columns and "patient_id" not in df.columns:
                df["patient_id"] = df["rec_id"].astype(str)
            else:
                df["patient_id"] = df["patient_id"].astype(str)

            if "ground_truth_category" not in df.columns:
                categories = []
                for _, r in df.iterrows():
                    ph = float(r["pH"])
                    bdecf = float(r["BDecf"]) if pd.notna(r.get("BDecf")) else 0.0
                    if ph < 7.15 or bdecf >= 12.0:
                        categories.append("Category III")
                    elif ph < 7.20 or bdecf >= 8.0:
                        categories.append("Category II")
                    else:
                        categories.append("Category I")
                df["ground_truth_category"] = categories

            self._patients_cache = df
        return self._patients_cache

    def get_streams_dict(self) -> Dict[str, Any]:
        if self._streams_cache is None:
            streams_file = self.data_dir / "real_patient_streams.json"
            if streams_file.exists():
                with open(streams_file, "r", encoding="utf-8") as f:
                    self._streams_cache = json.load(f)
            else:
                self._streams_cache = {}
        return self._streams_cache

    def list_patients(self) -> List[PatientSummary]:
        df = self.get_patients_df()
        patients = []
        for _, row in df.iterrows():
            gt_cat = str(row.get("ground_truth_category", "Category I"))
            if gt_cat == "0":
                gt_cat = "Category I"
            elif gt_cat == "1":
                gt_cat = "Category II"
            elif gt_cat == "2":
                gt_cat = "Category III"

            patients.append(
                PatientSummary(
                    patient_id=str(row["patient_id"]),
                    pH=float(row["pH"]),
                    BDecf=float(row["BDecf"]) if pd.notna(row.get("BDecf")) else None,
                    ground_truth_category=gt_cat,
                    Apgar1=float(row["Apgar1"]) if pd.notna(row.get("Apgar1")) else None,
                    delivery_type=str(row.get("delivery_type", "normal")) if pd.notna(row.get("delivery_type")) else None,
                )
            )
        return patients

    def get_patient(self, patient_id: str) -> PatientDetailResponse:
        df = self.get_patients_df()
        pid_str = str(patient_id)
        match = df[df["patient_id"] == pid_str]
        if match.empty:
            raise PatientNotFoundError(patient_id=pid_str)

        row = match.iloc[0]
        gt_cat = str(row.get("ground_truth_category", "Category I"))
        if gt_cat == "0":
            gt_cat = "Category I"
        elif gt_cat == "1":
            gt_cat = "Category II"
        elif gt_cat == "2":
            gt_cat = "Category III"

        # Check stream duration
        total_samples = 0
        duration = 0.0
        has_rec = False

        streams = self.get_streams_dict()
        if pid_str in streams:
            fhr_data = streams[pid_str].get("fhr_points") or streams[pid_str].get("fhr") or []
            total_samples = len(fhr_data)
            duration = total_samples / 4.0
            has_rec = total_samples > 0
        elif (self.data_dir / f"{pid_str}.dat").exists():
            has_rec = True
            hea_file = self.data_dir / f"{pid_str}.hea"
            if hea_file.exists():
                with open(hea_file, "r") as hf:
                    first_line = hf.readline().split()
                    if len(first_line) >= 4:
                        total_samples = int(first_line[3])
                        duration = total_samples / 4.0

        return PatientDetailResponse(
            patient_id=pid_str,
            pH=float(row["pH"]),
            BDecf=float(row["BDecf"]) if pd.notna(row.get("BDecf")) else None,
            ground_truth_category=gt_cat,
            Apgar1=float(row["Apgar1"]) if pd.notna(row.get("Apgar1")) else None,
            delivery_type=str(row.get("delivery_type", "normal")) if pd.notna(row.get("delivery_type")) else None,
            has_recording=has_rec,
            total_samples=total_samples,
            duration_seconds=duration,
        )

    def get_patient_stream(self, patient_id: str) -> StreamResponse:
        pid_str = str(patient_id)
        streams = self.get_streams_dict()

        fhr: List[float] = []
        uc: List[float] = []
        sampling_rate = 4.0

        if pid_str in streams:
            p_dict = streams[pid_str]
            fhr = p_dict.get("fhr_points") or p_dict.get("fhr") or []
            uc = p_dict.get("uc_points") or p_dict.get("uc") or []
            sampling_rate = float(p_dict.get("fs", 4.0))
        elif (self.data_dir / f"{pid_str}.dat").exists():
            rec = read_wfdb_record(str(self.data_dir / pid_str))
            fhr = rec.fhr.tolist()
            uc = rec.uc.tolist()
            sampling_rate = float(rec.fs)
        else:
            raise PatientNotFoundError(patient_id=pid_str)

        duration = len(fhr) / sampling_rate if sampling_rate > 0 else 0.0
        return StreamResponse(
            patient_id=pid_str,
            sampling_rate_hz=sampling_rate,
            fhr=fhr,
            uc=uc,
            duration_seconds=duration,
        )

    def predict_signal(
        self,
        fhr_raw: List[float],
        uc_raw: Optional[List[float]] = None,
        sampling_rate_hz: float = 4.0
    ) -> PredictResponse:
        duration_sec = len(fhr_raw) / sampling_rate_hz if sampling_rate_hz > 0 else 0.0
        if duration_sec < 60.0:
            raise InsufficientDataError(duration_seconds=duration_sec, min_required=60.0)

        fhr_arr = np.array(fhr_raw, dtype=float)
        if uc_raw is not None and len(uc_raw) == len(fhr_raw):
            uc_arr = np.array(uc_raw, dtype=float)
        else:
            uc_arr = np.zeros_like(fhr_arr)

        # 1. Prepare raw signal for clinical feature window extraction
        min_samples = int(sampling_rate_hz * 600)  # 600s = 10 min window

        if len(fhr_arr) < min_samples:
            repeats = int(np.ceil(min_samples / len(fhr_arr)))
            fhr_eval = np.tile(fhr_arr, repeats)[:min_samples]
            uc_eval = np.tile(uc_arr, repeats)[:min_samples]
        else:
            fhr_eval = fhr_arr
            uc_eval = uc_arr

        # 2. Extract clinical features across window(s)
        df_feat = extract_features_from_signal(
            fhr_raw=fhr_eval,
            uc_raw=uc_eval,
            fs=sampling_rate_hz,
            window_len_sec=600,
            step_sec=300,
            patient_id="API_Patient"
        )

        # 3. Model inference
        model_file = self.models_dir / "model_xgb.pkl"
        df_pred = predict_risk(df_feat, model_path=model_file)

        # 4. Deterministic FIGO Clinical Safety Guardrails
        df_final = apply_safety_guardrails(df_pred, df_feat)

        # Triage to most severe window across sequence (Zero False Reassurance)
        cat_severity = {"Category I": 1, "Category II": 2, "Category III": 3}
        df_final["severity"] = df_final["final_category"].map(cat_severity)
        idx_max = df_final["severity"].idxmax()
        row_final = df_final.loc[idx_max]
        row_pred = df_pred.loc[idx_max]
        row_feat = df_feat.loc[idx_max]

        final_category = str(row_final["final_category"])
        final_risk = float(row_final["risk_score"]) * 100.0
        override = bool(row_final["guardrail_triggered"])
        raw_prob_cat3 = float(row_pred["prob_cat_iii"])
        reason_str = str(row_final.get("guardrail_reason", ""))
        reasons = reason_str.split("; ") if reason_str and reason_str != "Normal physiological trace" else []

        # Recommendation mapping
        if final_category == "Category III":
            rec = "Immediate Obstetric Red Alert. Urgent bedside evaluation by obstetric specialist required. Prepare for emergency intrauterine resuscitation or operative delivery."
        elif final_category == "Category II":
            rec = "Suspicious fetal pattern. Initiate maternal repositioning (left lateral), verify maternal hydration and oxygenation, and evaluate for potential reversible causes."
        else:
            rec = "Reassuring fetal status. Standard continuous/intermittent intrapartum surveillance per FIGO guidelines."

        feat_dict = {
            k: round(float(v), 2) if isinstance(v, (float, np.floating)) else v
            for k, v in row_feat.to_dict().items()
        }

        return PredictResponse(
            category=final_category,
            risk_score=round(final_risk, 1),
            acidemia_probability=round(raw_prob_cat3, 4),
            guardrail_override=override,
            guardrail_reasons=reasons,
            features=feat_dict,
            clinical_recommendation=rec
        )

    def process_csv_upload(self, content_str: str, filename: str = "upload.csv") -> FileUploadResponse:
        import io
        try:
            df = pd.read_csv(io.StringIO(content_str))
        except Exception as e:
            raise ValidationError(f"Invalid CSV file format: {e}", code="INVALID_CSV")

        # Find FHR and UC columns flexibly
        cols_lower = {c.lower().strip(): c for c in df.columns}
        fhr_col = None
        for candidate in ["fhr", "fetal_heart_rate", "hr", "channel_1", "ch1"]:
            if candidate in cols_lower:
                fhr_col = cols_lower[candidate]
                break
        if not fhr_col and len(df.columns) >= 1:
            fhr_col = df.columns[0]

        uc_col = None
        for candidate in ["uc", "toco", "uterine_contraction", "contractions", "channel_2", "ch2"]:
            if candidate in cols_lower:
                uc_col = cols_lower[candidate]
                break
        if not uc_col and len(df.columns) >= 2:
            uc_col = df.columns[1]

        fhr_raw = df[fhr_col].astype(float).tolist()
        uc_raw = df[uc_col].astype(float).tolist() if uc_col else [0.0] * len(fhr_raw)

        pred_res = self.predict_signal(fhr_raw=fhr_raw, uc_raw=uc_raw, sampling_rate_hz=4.0)
        total_samples = len(fhr_raw)
        duration_sec = total_samples / 4.0
        windows_count = max(1, int(duration_sec // 300))

        return FileUploadResponse(
            filename=filename,
            total_samples=total_samples,
            duration_seconds=duration_sec,
            windows_analyzed=windows_count,
            category=pred_res.category,
            risk_score=pred_res.risk_score,
            guardrail_override=pred_res.guardrail_override,
            guardrail_reasons=pred_res.guardrail_reasons,
            clinical_recommendation=pred_res.clinical_recommendation
        )
