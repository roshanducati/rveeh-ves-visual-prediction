# `07_followup_sensitivity.py`

This script supports the manuscript's variable-follow-up limitation analysis.
The outcome-known cohort is restricted before modelling to minimum follow-up
durations of 0, 30, 60, 90, 120, and 180 days. At each level, the lead XGBoost
pipeline is re-tuned and evaluated using the reported fixed random values and the
same modelling design.

Each follow-up restriction is repartitioned after eligibility is applied. This
estimates performance within that restricted cohort, but the test rows are not
paired across thresholds. Performance differences therefore reflect both the
eligibility restriction and the resulting sample composition, and are treated
as robustness checks rather than paired effects of follow-up duration.

`followup_sensitivity.csv` records cohort size, patient count, poor-outcome rate,
cross-validated AUROC, and held-out performance for each restriction.

`followup_descriptive_checks.csv` separately reproduces:

- the Mann-Whitney comparison of follow-up duration by outcome;
- the association between presenting logMAR and outcome among episodes followed
  for at least 90 days; and
- the frequency of eye loss among poor outcomes recorded within 14 days.
