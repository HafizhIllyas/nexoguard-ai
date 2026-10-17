"""
NexoGuard AI
Clinical Decision Support and Intrapartum CTG Predictive Modeling Suite.
Complies with FIGO 2015 Guidelines and Dawes-Redman criteria.
"""

__version__ = "1.0.0"

from nexoguard.signal.reader import read_wfdb_record, CTGRecord
from nexoguard.signal.preprocessor import clean_fhr_signal, calculate_signal_loss_pct
from nexoguard.signal.windowing import slice_signal_windows, SignalWindow
from nexoguard.features.extractor import extract_features_from_signal, CONTRACT_FEATURE_COLS
from nexoguard.models.inference import predict_risk
from nexoguard.guardrails.clinical_rules import apply_safety_guardrails, evaluate_single_window_guardrail
from nexoguard.benchmark.runner import run_clinical_benchmark
from nexoguard.stress_testing.harness import run_stress_testing_harness

__all__ = [
    "__version__",
    "read_wfdb_record",
    "CTGRecord",
    "clean_fhr_signal",
    "calculate_signal_loss_pct",
    "slice_signal_windows",
    "SignalWindow",
    "extract_features_from_signal",
    "CONTRACT_FEATURE_COLS",
    "predict_risk",
    "apply_safety_guardrails",
    "evaluate_single_window_guardrail",
    "run_clinical_benchmark",
    "run_stress_testing_harness",
]
