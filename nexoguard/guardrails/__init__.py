"""
nexoguard.guardrails
Deterministic FIGO 2015 safety guardrails and clinical fail-safe envelope.
"""

from nexoguard.guardrails.clinical_rules import (
    evaluate_single_window_guardrail,
    apply_safety_guardrails,
    ALL_RULES,
    RULE_SEVERE_BRADYCARDIA,
    RULE_PROLONGED_DECEL,
    RULE_LATE_DECEL_LOW_STV,
    RULE_ABSENT_VARIABILITY,
    RULE_SINUSOIDAL_PATTERN,
    RULE_SIGNAL_INTEGRITY
)

__all__ = [
    "evaluate_single_window_guardrail",
    "apply_safety_guardrails",
    "ALL_RULES",
    "RULE_SEVERE_BRADYCARDIA",
    "RULE_PROLONGED_DECEL",
    "RULE_LATE_DECEL_LOW_STV",
    "RULE_ABSENT_VARIABILITY",
    "RULE_SINUSOIDAL_PATTERN",
    "RULE_SIGNAL_INTEGRITY",
]
