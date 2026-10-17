import os
import urllib.request
import re
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/"

def get_record_list():
    records_url = BASE_URL + "RECORDS"
    print(f"Fetching record list from {records_url}...")
    req = urllib.request.Request(records_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode('utf-8')
    records = [line.strip() for line in text.strip().splitlines() if line.strip()]
    print(f"Found {len(records)} clinical CTG records in CTU-UHB database.")
    return records

def download_file(filename):
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    url = BASE_URL + filename
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp, open(local_path, "wb") as out:
            out.write(resp.read())
        return local_path
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None

def parse_record_hea(rec_id):
    """
    Parses header file for sampling rate, duration, and clinical outcome markers (pH, BE, Apgar).
    """
    hea_file = os.path.join(DATA_DIR, f"{rec_id}.hea")
    if not os.path.exists(hea_file):
        return None
    
    metadata = {"rec_id": rec_id, "pH": np.nan, "BDecf": np.nan, "Apgar1": np.nan, "Apgar5": np.nan, "delivery": "normal"}
    with open(hea_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                # e.g. #pH 7.25 or #pH = 7.25
                m_ph = re.search(r"#\s*pH\s*[:=]?\s*([0-9.]+)", line, re.IGNORECASE)
                if m_ph:
                    metadata["pH"] = float(m_ph.group(1))
                m_be = re.search(r"#\s*BDecf\s*[:=]?\s*([0-9.-]+)", line, re.IGNORECASE)
                if m_be:
                    metadata["BDecf"] = float(m_be.group(1))
                m_ap1 = re.search(r"#\s*Apgar1\s*[:=]?\s*([0-9.]+)", line, re.IGNORECASE)
                if m_ap1:
                    metadata["Apgar1"] = float(m_ap1.group(1))
                m_ap5 = re.search(r"#\s*Apgar5\s*[:=]?\s*([0-9.]+)", line, re.IGNORECASE)
                if m_ap5:
                    metadata["Apgar5"] = float(m_ap5.group(1))
    return metadata

def parse_record_dat(rec_id):
    """
    Parses the 16-bit binary signal file for CTU-UHB:
    2 channels: FHR and UC at 4 Hz. Format 16: 16-bit signed integer, 2 channels interleaved.
    """
    dat_file = os.path.join(DATA_DIR, f"{rec_id}.dat")
    if not os.path.exists(dat_file):
        return None
    # 16-bit signed integer little-endian
    data = np.fromfile(dat_file, dtype=np.int16)
    if len(data) % 2 != 0:
        data = data[:len(data) - (len(data) % 2)]
    data = data.reshape(-1, 2)
    # Channel 0 is FHR (bpm, values roughly 50-200, 0 is missing)
    # Channel 1 is UC (mmHg, values 0-100)
    df = pd.DataFrame({
        "time_sec": np.arange(len(data)) / 4.0,
        "fhr_bpm": data[:, 0].astype(float),
        "uc_mmhg": data[:, 1].astype(float)
    })
    # Replace 0 bpm dropouts with NaN
    df.loc[df["fhr_bpm"] <= 0, "fhr_bpm"] = np.nan
    return df

if __name__ == "__main__":
    records = get_record_list()
    # Download a balanced initial batch of 35 clinical records
    print("Downloading initial cohort of clinical CTG records (hea and dat)...")
    parsed_metadata = []
    
    for rec in records[:40]:
        hea = download_file(f"{rec}.hea")
        dat = download_file(f"{rec}.dat")
        if hea and dat:
            meta = parse_record_hea(rec)
            if meta:
                parsed_metadata.append(meta)
                print(f"Loaded Record {rec}: pH = {meta.get('pH')}, BDecf = {meta.get('BDecf')}")
                
    meta_df = pd.DataFrame(parsed_metadata)
    meta_df.to_csv(os.path.join(DATA_DIR, "clinical_outcomes.csv"), index=False)
    print(f"\nSaved metadata for {len(meta_df)} patients to data/clinical_outcomes.csv!")
    print(meta_df.describe())
