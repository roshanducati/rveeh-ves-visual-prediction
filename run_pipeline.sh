#!/usr/bin/env bash
# Reproduce the published results end to end (strict outcome threshold).
# Requires the registry export at data/raw/ (see DATA_SCHEMA.md).
set -euo pipefail

echo "[1/9] Preprocess (strict)"
python 01_preprocess.py --strict

echo "[2/9] Train full model (with microbiology)"
python 02_train.py --strict

echo "[3/9] Train pre-culture model (without microbiology)"
python 02_train.py --strict --nomicro

echo "[4/9] Evaluate full and pre-culture models (XGBoost lead)"
python 03_evaluate.py --strict --lead_model XGB
python 03_evaluate.py --strict --nomicro --lead_model XGB

echo "[5/9] Participant flow"
python participant_flow.py

echo "[6/9] Timing analysis"
python 04_timing_analysis.py

echo "[7/9] Missing-data sensitivity analysis"
python 05_sensitivity_analysis.py

echo "[8/9] Class-imbalance sensitivity analysis"
python 06_imbalance_sensitivity.py

echo "[9/9] Follow-up sensitivity analysis"
python 07_followup_sensitivity.py

echo "Done. See results_strict*/ for generated outputs."
