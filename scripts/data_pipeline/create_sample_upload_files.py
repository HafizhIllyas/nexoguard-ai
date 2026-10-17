"""
create_sample_upload_files.py
Generates 4 clinical sample CSV files directly from validated PhysioNet patient telemetry streams.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STREAMS_PATH = PROJECT_ROOT / "data" / "real_patient_streams.json"
UPLOAD_DIR = PROJECT_ROOT / "data" / "sample_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

with open(STREAMS_PATH, "r") as f:
    streams = json.load(f)

# 1. Category I Reassuring (from Patient 1011 verified reassuring trace, pH 7.37, BD 1.7)
from nexoguard.signal.reader import read_wfdb_record
rec_1011 = read_wfdb_record("1011", data_dir=PROJECT_ROOT / "data")
fhr_cat1 = rec_1011.fhr[7200:9600]
uc_cat1 = rec_1011.uc[7200:9600]
n_pts1 = len(fhr_cat1)
time_arr1 = np.round(np.arange(n_pts1) / 4.0, 2)
df_cat1 = pd.DataFrame({
    "time_sec": time_arr1,
    "fhr": np.round(fhr_cat1, 1),
    "uc": np.round(uc_cat1, 1)
})
df_cat1.to_csv(UPLOAD_DIR / "sample_category_i_reassuring.csv", index=False)
print("Saved: sample_category_i_reassuring.csv (from Patient 1011, pH 7.37)")

# 2. Category II Suspicious (from Patient 1006)
p_1006 = streams["1006"]
n_pts = min(len(p_1006["fhr_points"]), len(p_1006["uc_points"]))
time_arr = np.round(np.arange(n_pts) / 4.0, 2)
df_cat2 = pd.DataFrame({
    "time_sec": time_arr,
    "fhr": p_1006["fhr_points"][:n_pts],
    "uc": p_1006["uc_points"][:n_pts]
})
df_cat2.to_csv(UPLOAD_DIR / "sample_category_ii_suspicious.csv", index=False)
print("Saved: sample_category_ii_suspicious.csv (from Patient 1006, pH 7.23)")

# 3. Category III Pathological (from Patient 1029)
p_1029 = streams["1029"]
n_pts = min(len(p_1029["fhr_points"]), len(p_1029["uc_points"]))
time_arr = np.round(np.arange(n_pts) / 4.0, 2)
df_cat3 = pd.DataFrame({
    "time_sec": time_arr,
    "fhr": p_1029["fhr_points"][:n_pts],
    "uc": p_1029["uc_points"][:n_pts]
})
df_cat3.to_csv(UPLOAD_DIR / "sample_category_iii_pathological.csv", index=False)
print("Saved: sample_category_iii_pathological.csv (from Patient 1029, pH 6.97)")

# 4. Telemetry Dropout Edge Case (>35% loss)
fhr_degraded = np.array(fhr_cat1[:n_pts1]).copy()
dropout_indices = np.random.RandomState(42).choice(n_pts1, size=int(0.38 * n_pts1), replace=False)
fhr_degraded[dropout_indices] = 0.0

df_edge = pd.DataFrame({
    "time_sec": time_arr1,
    "fhr": np.round(fhr_degraded, 1),
    "uc": np.round(uc_cat1[:n_pts1], 1)
})
df_edge.to_csv(UPLOAD_DIR / "sample_edge_case_sensor_dropout.csv", index=False)
print("Saved: sample_edge_case_sensor_dropout.csv (38% telemetry dropouts)")
