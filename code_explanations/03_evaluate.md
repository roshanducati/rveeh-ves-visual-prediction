# `03_evaluate.py`

This script consumes the fitted pipelines and held-out predictions from
`02_train.py` and produces the submitted evaluation analyses.

It generates:

- ROC and precision-recall curves;
- reliability diagrams, Loess-based integrated calibration index, and Brier
  score;
- confusion matrices at the frozen development-set thresholds;
- Random Forest impurity importance and XGBoost gain importance;
- full and concise SHAP summaries plus example waterfalls;
- manuscript Table 1 with observed denominators and the stated univariable
  comparisons;
- the incremental microbiology comparison in manuscript Table 3, using
  DeLong's test and a paired bootstrap interval;
- the descriptive held-out threshold grid in manuscript Table 4;
- decision-curve net benefit; and
- descriptive held-out AUROC by sex and age group.

AUROC and AUPRC confidence intervals use 1,000 non-parametric bootstrap
replicates. ICI is the point estimate from the original held-out sample; its
interval is bootstrapped. The fixed evaluation random value is 42.
The ROC and precision-recall figure labels read the same point estimates and
confidence intervals as the corresponding performance table, preventing
reporting drift between outputs.

For Table 1, continuous variables use two-sided Mann-Whitney U tests and
categorical variables use chi-squared tests with Yates' continuity correction.
Each test uses the observations available for that variable.

`--lead_model XGB` selects the lead model reported in the manuscript and is also
the default. Full and pre-culture predictions are checked for identical episode
identity before a paired comparison is performed.
