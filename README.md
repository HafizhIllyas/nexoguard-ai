# NexoGuard AI: Intrapartum Cardiotocography (CTG) Clinical Decision Support System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Clinical Standard](https://img.shields.io/badge/Clinical_Standard-FIGO_2015-emerald.svg)](https://www.figo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NexoGuard AI** is a clinical-grade decision support platform and predictive modeling engine designed for continuous intrapartum fetal monitoring. Combining real-time digital signal processing (DSP) of dual-channel Cardiotocography (Fetal Heart Rate + Uterine Contractions) with calibrated machine learning and deterministic clinical safety guardrails, NexoGuard AI helps obstetrical teams detect fetal acidemia, hypoxia, and distress early while eliminating false reassurance.

---

## Clinical Foundation: FIGO 2015 3-Tier Classification

The system strictly operationalizes the **FIGO 2015 Intrapartum Fetal Monitoring Guidelines**:

| Category | Clinical Interpretation | Physiological Criteria | System Action |
|:---|:---|:---|:---|
| **Category I (Normal)** | Reassuring, normal acid-base status (Risk of acidemia < 1%) | Baseline 110–160 bpm, variability 5–25 bpm, absence of repetitive late/variable decelerations. | Standard intrapartum monitoring. Continuous telemetry. |
| **Category II (Suspicious)** | Equivocal; not predictive of abnormal acid-base status, but warrants surveillance | Mild bradycardia (100–110 bpm) with normal variability, tachycardia (>160 bpm), minimal variability (<5 bpm) without decelerations, or marked variability (>25 bpm). | Clinical alert, maternal repositioning, oxygenation/hydration assessment, continuous close surveillance. |
| **Category III (Pathological)** | High risk of fetal acidemia (cord arterial pH < 7.15) & metabolic acidosis | Sustained bradycardia (<100 bpm), absent/minimal variability with recurrent late or prolonged decelerations, or sinusoidal pattern. | **Immediate Obstetric Red Alert**. Mandates urgent clinical evaluation, intrauterine resuscitation, or operative delivery. |

### Deterministic Clinical Safety Guardrails
Machine learning predictions alone are never solely trusted with fetal life. NexoGuard AI enforces **hardcoded, deterministic safety guardrails**:
- If severe physiological red flags (e.g. baseline < 100 bpm, prolonged deceleration > 3 minutes, sinusoidal pattern, or profound variability collapse) are detected by DSP, the guardrail **overrides** ML predictions to force Category III / High Risk.
- **Zero False Reassurance Invariant**: A Category III pathological pattern can never be downgraded to Category I (Normal) regardless of model probabilities.

---

## Architectural Overview

```
                                    ┌────────────────────────────────────────────────────────┐
                                    │       Dual-Channel CTG Telemetry (FHR + UC at 4 Hz)     │
                                    └──────────────────────────┬─────────────────────────────┘
                                                               │
                                                               ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │          Digital Signal Processing (DSP) Engine        │
                                    │  • Artifact rejection & interpolation                  │
                                    │  • Baseline estimation (10-min moving window + SG)     │
                                    │  • Micro-variability (Short-term STV & Long-term LTV)  │
                                    │  • Deceleration detection (Early, Late, Variable)      │
                                    │  • Uterine Contraction frequency, duration & tonus     │
                                    └──────────────┬──────────────────────────┬──────────────┘
                                                   │                          │
                         ML Feature Vector (35 dim)│                          │ Physiological Red Flags
                                                   ▼                          ▼
                   ┌────────────────────────────────────────┐   ┌─────────────────────────────┐
                   │    Calibrated Machine Learning Models  │   │ Deterministic FIGO 2015     │
                   │ • Gradient Boosting & XGBoost Classifiers│   │ Clinical Safety Guardrails   │
                   │ • Probabilistic Risk Scoring (0-100%)  │   │ (Zero False Reassurance)    │
                   └───────────────────┬────────────────────┘   └─────────────┬───────────────┘
                                       │                                      │
                                       └───────────────────┬──────────────────┘
                                                           ▼
                                           ┌───────────────────────────────┐
                                           │   Unified Risk Assessment     │
                                           │ • FIGO Category (I / II / III)│
                                           │ • Acidemia Risk Score (%)     │
                                           │ • Actionable Recommendations  │
                                           └───────────────┬───────────────┘
                                                           │
                                           ┌───────────────┴───────────────┐
                                           ▼                               ▼
                           ┌───────────────────────────────┐   ┌───────────────────────────────┐
                           │ 60 FPS Live Web Telemetry UI  │   │  CLI / Automated Batch Runner │
                           │ (Dual-channel Canvas Monitor) │   │  (`nexoguard` / `pytest`)     │
                           └───────────────────────────────┘   └───────────────────────────────┘
```

---

## Directory Structure

```
nexoguard-team/
├── nexoguard/                  # Core installable Python package
│   ├── __init__.py             # Package namespace & version
│   ├── __main__.py             # CLI entry point (`python -m nexoguard`)
│   ├── cli.py                  # Command-line interface & interactive patient inspector
│   ├── signal_processing.py    # Dual-channel CTG DSP algorithms
│   ├── guardrails.py           # Deterministic FIGO 2015 safety invariants
│   ├── inference.py            # Calibrated model inference & risk scoring
│   ├── models/                 # Pre-trained production model weights & artifacts
│   └── utils/                  # Helper functions and dataset loaders
├── data/                       # Real PhysioNet CTG recordings and reference data
├── scripts/                    # Organized workflow utilities
│   ├── data_pipeline/          # Dataset downloading, preprocessing & sample generation
│   ├── model_training/         # Signal processing jobs & model training pipelines
│   └── web_assets/             # Web telemetry upgrade & documentation embedding
├── tests/                      # Comprehensive test suite
│   ├── test_predefined_cases.py# Unit tests across clinical test scenarios 1-7
│   └── test_safety_guardrails.py# Adversarial stress tests for safety guardrails
├── app.py                      # Interactive Streamlit dashboard with live 60 FPS monitor
├── test_pipeline.py            # Automated 6-stage end-to-end DSLC validation runner
├── test_patient.py             # Backward-compatible patient CLI shim
├── pyproject.toml              # Modern PEP 517/518 build specification
└── requirements.txt            # Python dependencies
```

---

## Quickstart & Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/nexoguard/nexoguard-ai.git
cd nexoguard-team

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies and editable package
pip install -e .
```

### 2. Launch the Web Telemetry Dashboard

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view:
- **Live 60 FPS Dual-Channel Telemetry Monitor** (FHR on channel 1, UC on channel 2).
- **Predefined Clinical Test Cases** (Scenarios 1 through 7 covering Normal, Variable, Late, and Terminal Decelerations).
- **Direct CSV File Upload** with automatic FIGO classification and risk grading.

### 3. Launch the Decoupled FastAPI / ASGI Service

```bash
# Launch API server using console entry point
nexoguard-api

# Or directly with uvicorn
uvicorn nexoguard.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at `http://localhost:8000/docs`:
- `GET /api/health`: Operational readiness check
- `GET /api/patients`: Full patient cohort metadata & ground truth
- `GET /api/patients/{id}/stream`: Real-time 4 Hz telemetry time-series
- `POST /api/predict`: Evaluate raw FHR/UC signal with FIGO guardrails
- `POST /api/predict/upload`: Multipart CSV upload & triage
- `WebSocket /api/ws/stream/{id}`: Sub-10ms latency live CTG telemetry stream

### 4. Command-Line Interface (CLI)

```bash
# List all available clinical test scenarios
nexoguard --list

# Evaluate a specific clinical test case (e.g., Scenario 7: Prolonged Deceleration / Category III)
nexoguard --test-case 7

# Run live evaluation on a specific patient recording
nexoguard --patient 1001

# Inspect an arbitrary patient CSV file
nexoguard --file path/to/patient_ctg.csv
```

### 5. Running the Verification Suite

Run all automated unit tests and the 6-stage DSLC benchmark runner:

```bash
# Run clinical scenario unit tests (100% pass required)
python -m unittest tests/test_predefined_cases.py

# Run full DSLC 6-stage pipeline validation and stress-testing harness
python test_pipeline.py
```

---

## Clinical Safety & Disclaimer

> [!IMPORTANT]
> **Investigational & Decision Support Use Only**: NexoGuard AI is designed as a secondary clinical decision support aid to augment obstetrical vigilance. It does not replace clinical judgment, physical examination, fetal scalp blood sampling, or direct maternal-fetal medical supervision.
