"""
test_patient.py
CLI patient-level validation runner for NexoGuard AI.
Maintained for backward compatibility; delegates directly to nexoguard.cli.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexoguard.cli import main

if __name__ == "__main__":
    main()
