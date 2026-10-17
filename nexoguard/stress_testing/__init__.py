"""
nexoguard.stress_testing
Degradation harness and telemetry perturbations simulating real-world sensor impairments.
"""

from nexoguard.stress_testing.perturbations import (
    inject_gilbert_elliott_dropout,
    inject_telemetry_noise,
    inject_baseline_wander,
    inject_spike_impulses,
    apply_perturbation_scenario
)
from nexoguard.stress_testing.harness import run_stress_testing_harness, SCENARIOS

__all__ = [
    "inject_gilbert_elliott_dropout",
    "inject_telemetry_noise",
    "inject_baseline_wander",
    "inject_spike_impulses",
    "apply_perturbation_scenario",
    "run_stress_testing_harness",
    "SCENARIOS",
]
