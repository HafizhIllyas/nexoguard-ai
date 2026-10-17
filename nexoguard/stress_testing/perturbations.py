"""
nexoguard.stress_testing.perturbations
Signal perturbation functions modeling real-world obstetric telemetry impairments.
"""

import numpy as np


def inject_gilbert_elliott_dropout(
    fhr_signal: np.ndarray,
    target_dropout_pct: float = 10.0,
    mean_burst_sec: float = 5.0,
    fs: float = 4.0,
    random_seed: int = 42
) -> np.ndarray:
    """
    Applies 2-state Markov burst dropouts (Gilbert-Elliott model).
    Simulates acoustic decoupling between ultrasound transducer and maternal abdomen.
    """
    rng = np.random.default_rng(random_seed)
    n = len(fhr_signal)
    signal = np.array(fhr_signal, dtype=float).copy()

    p_rec = 1.0 / max(mean_burst_sec * fs, 1.0)
    target_fraction = target_dropout_pct / 100.0
    p_drop = (target_fraction * p_rec) / max(1.0 - target_fraction, 1e-4)

    state = 0  # 0: Good (Acoustic Lock), 1: Bad (Dropout)
    for t in range(n):
        if state == 0:
            if rng.random() < p_drop:
                state = 1
        else:
            if rng.random() < p_rec:
                state = 0

        if state == 1:
            signal[t] = 0.0  # Dropout sample

    return signal


def inject_telemetry_noise(
    fhr_signal: np.ndarray,
    sigma_bpm: float = 2.0,
    random_seed: int = 42
) -> np.ndarray:
    """
    Adds zero-mean Gaussian electronic and transducer jitter.
    Preserves existing dropouts (values <= 0 or NaN).
    """
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(0.0, sigma_bpm, size=len(fhr_signal))
    degraded = np.array(fhr_signal, dtype=float) + noise
    degraded[(fhr_signal <= 0.0) | np.isnan(fhr_signal)] = 0.0
    return degraded


def inject_baseline_wander(
    fhr_signal: np.ndarray,
    amplitude_bpm: float = 5.0,
    freq_hz: float = 0.15,
    fs: float = 4.0
) -> np.ndarray:
    """
    Adds low-frequency sinusoidal drift modeling maternal respiration and body movement.
    """
    t = np.arange(len(fhr_signal)) / fs
    wander = amplitude_bpm * np.sin(2.0 * np.pi * freq_hz * t)
    degraded = np.array(fhr_signal, dtype=float) + wander
    degraded[(fhr_signal <= 0.0) | np.isnan(fhr_signal)] = 0.0
    return degraded


def inject_spike_impulses(
    fhr_signal: np.ndarray,
    spike_prob: float = 0.02,
    max_amp_bpm: float = 35.0,
    random_seed: int = 42
) -> np.ndarray:
    """
    Adds random high-frequency transient spike impulses simulating transducer bumping.
    """
    rng = np.random.default_rng(random_seed)
    n = len(fhr_signal)
    degraded = np.array(fhr_signal, dtype=float).copy()
    mask = (rng.random(n) < spike_prob) & (fhr_signal > 0.0) & (~np.isnan(fhr_signal))
    spikes = rng.uniform(-max_amp_bpm, max_amp_bpm, size=n)
    degraded[mask] += spikes[mask]
    return degraded


def apply_perturbation_scenario(
    fhr_signal: np.ndarray,
    scenario_name: str,
    fs: float = 4.0,
    seed: int = 42
) -> np.ndarray:
    """
    Executes one of the 3 calibrated telemetry perturbation scenarios:
    - scenario_1_mild: 10% dropout + 2 bpm Gaussian jitter
    - scenario_2_moderate: 25% burst dropout + 4 bpm jitter + baseline wander + 2% spikes
    - scenario_3_severe: 40% detachment + 6 bpm jitter + 5% spikes
    """
    if scenario_name == "scenario_1_mild":
        sig = inject_gilbert_elliott_dropout(
            fhr_signal, target_dropout_pct=10.0, mean_burst_sec=3.0, fs=fs, random_seed=seed
        )
        sig = inject_telemetry_noise(sig, sigma_bpm=2.0, random_seed=seed)
        return sig

    elif scenario_name == "scenario_2_moderate":
        sig = inject_gilbert_elliott_dropout(
            fhr_signal, target_dropout_pct=25.0, mean_burst_sec=12.0, fs=fs, random_seed=seed
        )
        sig = inject_telemetry_noise(sig, sigma_bpm=4.0, random_seed=seed)
        sig = inject_baseline_wander(sig, amplitude_bpm=5.0, freq_hz=0.15, fs=fs)
        sig = inject_spike_impulses(sig, spike_prob=0.02, max_amp_bpm=25.0, random_seed=seed)
        return sig

    elif scenario_name == "scenario_3_severe":
        sig = inject_gilbert_elliott_dropout(
            fhr_signal, target_dropout_pct=40.0, mean_burst_sec=35.0, fs=fs, random_seed=seed
        )
        sig = inject_telemetry_noise(sig, sigma_bpm=6.0, random_seed=seed)
        sig = inject_spike_impulses(sig, spike_prob=0.05, max_amp_bpm=35.0, random_seed=seed)
        return sig

    else:
        raise ValueError(f"Unknown perturbation scenario: '{scenario_name}'. Expected scenario_1_mild, scenario_2_moderate, or scenario_3_severe.")
