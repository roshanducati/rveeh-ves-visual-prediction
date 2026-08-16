# `05_sensitivity_analysis.py`

This script tests whether performance depends on the two sparsely documented
predictors. It evaluates nested feature sets:

1. all 16 predictors;
2. immune suppression removed (15 predictors); and
3. immune suppression and RAPD removed (14 predictors).

Random Forest and XGBoost are re-tuned from scratch at every tier using the same
preprocessing, fixed patient partition (61), estimation value (42), 5-fold
grouped cross-validation, class weighting, and 100-configuration search as the
main analysis.

Every reduced tier is evaluated on the same held-out episodes as its full-model
reference. `auroc_diff_vs_full` is reduced minus full, matching manuscript Table
5. DeLong p-values and paired bootstrap confidence intervals account for the
paired predictions.
