"""
nexoguard.signal.reader
Portable WFDB Format 16 binary reader and header parser for PhysioNet CTU-UHB CTG records.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass(frozen=True)
class CTGRecord:
    rec_id: str
    fhr: np.ndarray          # 1D float array, raw FHR in bpm
    uc: np.ndarray           # 1D float array, raw UC in relative units
    fs: float = 4.0          # Sampling rate in Hz
    duration_sec: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def parse_header_file(hea_path: Union[str, Path]) -> Tuple[float, int, Dict[str, Any]]:
    """
    Parses a PhysioNet WFDB .hea plaintext header file.
    Extracts sampling frequency, total samples, and clinical comment metadata.
    """
    hea_path = Path(hea_path)
    if not hea_path.is_file():
        return 4.0, 0, {}

    fs = 4.0
    n_samples = 0
    meta: Dict[str, Any] = {}

    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if lines:
        header_parts = lines[0].strip().split()
        if len(header_parts) >= 4:
            try:
                fs = float(header_parts[2])
                n_samples = int(header_parts[3])
            except ValueError:
                pass

    for line in lines[1:]:
        line_str = line.strip()
        if line_str.startswith("#"):
            comment = line_str.lstrip("#").strip()
            # Match metadata keys: e.g. "pH 7.14", "BDecf 8.14", "Apgar1 6"
            match = re.match(r"^([\w\.\s\(\)-]+?)\s+([-+]?\d*\.?\d+)$", comment)
            if match:
                key = match.group(1).strip()
                val = float(match.group(2))
                meta[key] = val
            else:
                parts = comment.split(None, 1)
                if len(parts) == 2:
                    meta[parts[0].strip()] = parts[1].strip()

    return fs, n_samples, meta


def read_wfdb_record(
    record_id_or_path: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None,
    fhr_gain: float = 100.0,
    uc_gain: float = 100.0
) -> CTGRecord:
    """
    Loads 16-bit interleaved binary CTU-UHB signals (4 Hz FHR & UC).
    Supports relative record IDs ("1001"), filenames ("1001.dat"), or explicit Paths.
    """
    path_obj = Path(record_id_or_path)

    if path_obj.suffix.lower() == ".csv" or (data_dir and (Path(data_dir) / path_obj.name).with_suffix(".csv").is_file()):
        csv_file = path_obj if path_obj.is_file() else (Path(data_dir) / path_obj.name).with_suffix(".csv")
        df_csv = pd.read_csv(csv_file)
        cols_lower = {str(c).lower(): c for c in df_csv.columns}
        fhr_col = cols_lower.get("fhr", cols_lower.get("fhr_bpm", cols_lower.get("fetal_heart_rate", list(df_csv.columns)[0])))
        uc_col = cols_lower.get("uc", cols_lower.get("toco", cols_lower.get("uterine_contractions", list(df_csv.columns)[1] if len(df_csv.columns) > 1 else None)))
        fhr_arr = df_csv[fhr_col].values.astype(float)
        uc_arr = df_csv[uc_col].values.astype(float) if uc_col else np.zeros(len(fhr_arr)) + 15.0
        return CTGRecord(
            rec_id=path_obj.stem,
            fhr=fhr_arr,
            uc=uc_arr,
            fs=4.0,
            duration_sec=len(fhr_arr) / 4.0,
            meta={"source": "csv_upload", "filename": path_obj.name}
        )

    if path_obj.suffix in [".dat", ".hea"]:
        base_path = path_obj.with_suffix("")
    elif data_dir is not None:
        base_path = Path(data_dir) / path_obj.name
    else:
        # Check relative candidate in cwd/data, then relative to package root
        candidate_cwd = Path("data") / path_obj.name
        if candidate_cwd.with_suffix(".dat").is_file():
            base_path = candidate_cwd
        elif (DEFAULT_DATA_DIR / path_obj.name).with_suffix(".dat").is_file():
            base_path = DEFAULT_DATA_DIR / path_obj.name
        else:
            base_path = Path(record_id_or_path)

    dat_path = base_path.with_suffix(".dat")
    hea_path = base_path.with_suffix(".hea")
    rec_id = base_path.stem

    if not dat_path.is_file():
        raise FileNotFoundError(f"WFDB binary signal file not found: {dat_path}")

    fs, n_samples_hdr, meta = parse_header_file(hea_path)

    # Read binary 16-bit signed integers (Format 16)
    raw_bytes = np.fromfile(str(dat_path), dtype=np.int16)
    if len(raw_bytes) % 2 != 0:
        raw_bytes = raw_bytes[:-1]

    if len(raw_bytes) == 0:
        raise ValueError(f"WFDB binary signal file is empty: {dat_path}")

    signals = raw_bytes.reshape(-1, 2)
    fhr_raw = signals[:, 0].astype(float) / fhr_gain
    uc_raw = signals[:, 1].astype(float) / uc_gain
    duration_sec = len(fhr_raw) / fs

    return CTGRecord(
        rec_id=rec_id,
        fhr=fhr_raw,
        uc=uc_raw,
        fs=fs,
        duration_sec=duration_sec,
        meta=meta
    )
