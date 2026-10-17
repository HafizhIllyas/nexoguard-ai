"""
nexoguard.benchmark
Automated clinical benchmarking runner against FIGO 2015 criteria and cord pH gold standards.
"""

from nexoguard.benchmark.runner import run_clinical_benchmark, CATEGORY_TO_INT

__all__ = [
    "run_clinical_benchmark",
    "CATEGORY_TO_INT",
]
