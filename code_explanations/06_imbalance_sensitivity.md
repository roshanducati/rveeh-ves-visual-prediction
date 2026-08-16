# `06_imbalance_sensitivity.py`

This script reproduces Supplement S3. It loads the tuned Random Forest and
XGBoost hyperparameters, refits each model with and without class-imbalance
correction, and holds the patient partition, preprocessing, hyperparameters, and
estimation value fixed.

The only toggles are:

- XGBoost: development-set `scale_pos_weight` versus `1.0`;
- Random Forest: `class_weight="balanced"` versus `None`.

Held-out AUROC, AUPRC, Brier score, ICI, ECE, calibration intercept/slope, and
mean predicted risk are reported. Paired 2,000-replicate bootstrap intervals are
reported as uncorrected minus corrected.
