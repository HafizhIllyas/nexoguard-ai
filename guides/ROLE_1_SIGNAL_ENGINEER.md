# ROLE GUIDE — PERSON 1: SIGNAL & FEATURE LEAD

**Title**: Lead Biomedical Signal Processing & Feature Engineer  
**Core Mission**: Transform raw, noisy 4Hz Cardiotocography (CTG) time series into the clean, tabular feature dataset (`features.csv`) that powers the entire team.  
**Tech Stack**: Python (`numpy`, `scipy`, `pandas`)

---

## 1. Quick Checklist & Deliverables

- [ ] **Hour 0–1**: Create a 5-row `mock_features.csv` so Person 2 and Person 3 can start immediately without waiting for you.
- [ ] **Hour 1–4**: Build signal cleaning pipeline (artifact rejection, median filter, PCHIP gap interpolation).
- [ ] **Hour 4–6**: Implement MHR vs. FHR cross-correlation check (detect transducer slip).
- [ ] **Hour 6–9**: Implement NexoPrima’s dynamic EMA baseline and Dawes-Redman STV / LTV variability.
- [ ] **Hour 9–12**: Implement Contraction-to-Deceleration Lag ($\Delta t$) calculation.
- [ ] **Hour 12–14**: Run the batch extractor on all patient recordings and export the official `features.csv`.
- [ ] **Hour 14+**: Package the feature extraction into a single clean function: `extract_features_from_window(fhr, uc) -> dict` for Person 3's real-time app.

---

## 2. Complete Starter Code: `signal_pipeline.py`

Save this file in your project folder as `signal_pipeline.py`:

```python
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter

def clean_fhr_signal(fhr_raw, sampling_rate_hz=4):
    """
    1. Rejects physiological impossibilities (< 50 or > 220 bpm) as NaN.
    2. Interpolates short dropouts (< 15 seconds = 60 samples).
    3. Applies median filtering followed by Savitzky-Golay smoothing.
    """
    fhr = np.array(fhr_raw, dtype=float)
    
    # 1. Reject out-of-bounds artifacts
    fhr[(fhr < 50) | (fhr > 220)] = np.nan
    
    # 2. Interpolate small gaps
    s = pd.Series(fhr)
    max_gap = sampling_rate_hz * 15 # 15 seconds
    fhr_interp = s.interpolate(method='linear', limit=max_gap).to_numpy()
    
    # Fill remaining NaNs with median baseline if any
    median_val = np.nanmedian(fhr_interp) if not np.all(np.isnan(fhr_interp)) else 135.0
    fhr_interp = np.nan_to_num(fhr_interp, nan=median_val)
    
    # 3. Denoise with 3-sample median filter
    fhr_med = medfilt(fhr_interp, kernel_size=3)
    
    # 4. Smooth high-frequency noise with Savitzky-Golay filter (window=11 samples, polyorder=2)
    try:
        fhr_clean = savgol_filter(fhr_med, window_length=11, polyorder=2)
    except Exception:
        fhr_clean = fhr_med
        
    return fhr_clean

def check_mhr_confusion(fhr_series, mhr_series):
    """
    Checks if ultrasound transducer slipped and locked onto maternal pulse.
    Returns correlation coefficient r. If r > 0.80, warn of sensor misplacement.
    """
    if mhr_series is None or len(mhr_series) == 0:
        return 0.0
    fhr = np.nan_to_num(fhr_series, nan=130.0)
    mhr = np.nan_to_num(mhr_series, nan=80.0)
    if np.std(fhr) == 0 or np.std(mhr) == 0:
        return 0.0
    corr = np.corrcoef(fhr, mhr)[0, 1]
    return float(corr)

def calculate_nexoprima_ema_baseline(fhr_clean, alpha=0.85):
    """
    Dynamic baseline calculation using NexoPrima's published formula:
    Bl_t = alpha * Bl_{t-1} + (1 - alpha) * fhr_t
    """
    baseline = np.zeros_like(fhr_clean)
    baseline[0] = np.median(fhr_clean[:240]) # initial 1-minute median
    
    for t in range(1, len(fhr_clean)):
        baseline[t] = alpha * baseline[t-1] + (1 - alpha) * fhr_clean[t]
        
    return baseline

def extract_clinical_features(fhr_window, uc_window, sampling_rate_hz=4, window_id="W001", mhr_window=None):
    """
    Extracts 12 core FIGO features from a 10-minute window (2400 samples).
    """
    # 1. Clean FHR
    fhr = clean_fhr_signal(fhr_window, sampling_rate_hz)
    uc = np.nan_to_num(np.array(uc_window, dtype=float), nan=10.0)
    
    # 2. NexoPrima Baseline
    baseline_series = calculate_nexoprima_ema_baseline(fhr)
    baseline_mean = float(np.mean(baseline_series))
    
    # 3. Short-Term Variability (STV in ms) - Dawes-Redman criteria
    # Convert bpm to interval (ms): 60,000 / bpm
    valid_fhr = np.clip(fhr, 50, 220)
    intervals_ms = 60000.0 / valid_fhr
    stv_ms = float(np.mean(np.abs(np.diff(intervals_ms))))
    
    # 4. Long-Term Variability (LTV in bpm) - Range over 1-minute blocks
    block_size = sampling_rate_hz * 60
    num_blocks = len(fhr) // block_size
    ltv_blocks = [
        np.max(fhr[i*block_size : (i+1)*block_size]) - np.min(fhr[i*block_size : (i+1)*block_size])
        for i in range(num_blocks)
    ]
    ltv_bpm = float(np.mean(ltv_blocks)) if ltv_blocks else float(np.ptp(fhr))
    
    # 5. Accelerations & Decelerations count
    # Accel: >= 15 bpm above baseline for >= 15 seconds (60 samples)
    # Decel: >= 15 bpm below baseline for >= 15 seconds (60 samples)
    diff = fhr - baseline_series
    is_accel = diff >= 15.0
    is_decel = diff <= -15.0
    
    accel_samples = np.sum(is_accel)
    decel_samples = np.sum(is_decel)
    
    # Approximate event counts (consecutive runs >= 60 samples)
    accel_count = int(accel_samples // 60)
    decel_count = int(decel_samples // 60)
    
    decel_max_depth = float(np.max(np.abs(diff[is_decel]))) if np.any(is_decel) else 0.0
    decel_area = float(np.sum(np.abs(diff[is_decel]))) if np.any(is_decel) else 0.0
    
    # 6. Contractions analysis
    uc_threshold = np.mean(uc) + np.std(uc)
    is_contracting = uc > max(35.0, uc_threshold)
    contraction_count = int(np.sum(is_contracting) // (sampling_rate_hz * 30)) # runs ~30s
    
    # 7. Contraction-Deceleration Lag (The Gold Feature: Early vs. Late Decels)
    decel_uc_lag_sec = 0.0
    if np.any(is_decel) and np.any(is_contracting):
        # Peak of UC vs Trough of FHR
        uc_peak_idx = np.argmax(uc)
        fhr_trough_idx = np.argmin(fhr)
        lag_samples = fhr_trough_idx - uc_peak_idx
        decel_uc_lag_sec = float(lag_samples / sampling_rate_hz)
    
    # 8. MHR cross-correlation
    mhr_corr = check_mhr_confusion(fhr, mhr_window)
    
    # 9. Signal loss percentage
    raw_arr = np.array(fhr_window)
    loss_pct = float((np.sum(np.isnan(raw_arr)) + np.sum(raw_arr < 50) + np.sum(raw_arr > 220)) / len(raw_arr) * 100)

    # 10. Clinical Ground-Truth Labeling (FIGO 2015 heuristic for training)
    # 0 = Normal, 1 = Suspicious, 2 = Pathological
    if (decel_uc_lag_sec > 25.0 and decel_count >= 1) or (stv_ms < 3.0 and decel_count >= 1) or (baseline_mean < 100):
        label = 2 # Category III Pathological
    elif stv_ms < 4.0 or baseline_mean > 160 or baseline_mean < 110 or decel_count >= 1:
        label = 1 # Category II Suspicious
    else:
        label = 0 # Category I Normal

    return {
        "window_id": window_id,
        "baseline_bpm": round(baseline_mean, 2),
        "stv_ms": round(stv_ms, 2),
        "ltv_bpm": round(ltv_bpm, 2),
        "accel_count": accel_count,
        "decel_count": decel_count,
        "decel_max_depth_bpm": round(decel_max_depth, 2),
        "decel_total_area": round(decel_area, 2),
        "contraction_count": contraction_count,
        "decel_uc_lag_sec": round(decel_uc_lag_sec, 2),
        "mhr_fhr_cross_corr": round(mhr_corr, 2),
        "signal_loss_pct": round(loss_pct, 2),
        "label": label
    }

if __name__ == "__main__":
    # Test on a synthetic 10-minute window
    print("Testing signal pipeline on sample window...")
    t = np.linspace(0, 600, 2400) # 10 mins @ 4Hz
    test_fhr = 140 + 5 * np.sin(t * 0.1) + np.random.normal(0, 2, 2400)
    test_uc = 15 + 40 * np.maximum(0, np.sin(t * 0.05))**4
    
    feat = extract_clinical_features(test_fhr, test_uc, window_id="DEMO_01")
    print("Extracted Features:\n", pd.Series(feat))
```

---

## 3. What to Hand Off & Coordinate

1. **For Person 2 (ML Lead)**: Hand over `features.csv`. Verify that there are **no NaN values** and that class distribution has at least a few instances of `label = 2`.
2. **For Person 3 (App Lead)**: Expose `clean_fhr_signal()` and `extract_clinical_features()` so Person 3 can import them directly inside the Streamlit app.

---

## 4. Stuck? How to Ask AI

Copy the content of `guides/PROJECT_MASTER_CONTEXT_FOR_AI.md` into your AI prompt, and add:
```
I am Person 1 (Signal Lead). 
I am currently working on [e.g. calculating the Contraction-to-Deceleration lag / Savitzky-Golay filtering].
Here is my current code: [paste code]
The issue is: [paste error message or what behavior you are seeing]
Please fix it while keeping the output format matching the features.csv contract.
```
