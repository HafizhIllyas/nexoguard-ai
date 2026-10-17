# Original User Request

## 2026-10-13T16:50:35Z

Build an automated predictive modeling and validation suite for the NexoGuard AI fetal monitoring system following the Data Science Life Cycle (DSLC). The system trains predictive models on the clinical intrapartum CTG dataset to prediagnose fetal acidemia and distress, benchmarks performance against clinical FIGO 2015 criteria, and stress-tests model resilience under real-world signal degradation.

Working directory: C:\Users\hafiz\.gemini\antigravity\scratch\nexoguard-team
Integrity mode: development

## Requirements

### R1. Predictive Modeling & DSLC Pipeline
Train and validate predictive models on the available hospital CTG dataset (`data/` directory). The pipeline must handle data preparation, feature extraction, patient-stratified train/test splitting (to avoid data leakage), model training, and probabilistic risk scoring for fetal distress (Category I, II, and III).

### R2. Automated Clinical Benchmarking Suite
Provide an automated benchmarking runner that evaluates predictive performance across all clinical cases. The suite must report accuracy, multi-class F1-scores, ROC-AUC, and specifically measure sensitivity/recall on severe pathological cases (umbilical cord pH < 7.15).

### R3. Signal Degradation & Stress-Testing Harness
Implement an automated stress-testing module that tests the predictive model's behavior under real-world telemetry impairments:
- Sensor dropouts and missing data intervals (simulating transducer detachment)
- Transducer motion artifacts and additive signal noise
- Verification that deterministic clinical safety guardrails trigger appropriately even when signal quality degrades

## Acceptance Criteria

### Execution & Integration
- [ ] An automated test runner (e.g., `test_pipeline.py` or equivalent command) executes from the command line without interactive input and exits with status 0 upon successful completion.
- [ ] The pipeline loads real patient recordings from `data/` and processes them through feature extraction and model inference.

### Metric Reporting
- [ ] The benchmark runner outputs an objective, machine-readable results summary (JSON or CSV) containing classification metrics across all classes.
- [ ] Model recall on severe pathological cases (Category III / pH < 7.15) is explicitly quantified.

### Stress-Testing Verification
- [ ] The stress-testing harness tests at least 3 distinct perturbation scenarios (e.g., varying dropout rates and noise levels) and reports model resilience under degraded signal quality.
- [ ] The safety guardrail mechanism is programmatically tested against simulated edge cases to verify that critical physiological warning signs cannot be misclassified as reassuring.
