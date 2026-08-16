# `02_train.py`

This script develops the Random Forest and XGBoost models reported in the
manuscript.

The full model uses 15 numeric/binary predictors plus one categorical organism
predictor. `--nomicro` removes culture status and organism category, leaving the
14-predictor pre-culture model. Initial procedure and follow-up duration are not
predictors.

The analysis uses a fixed patient-level 80/20 `GroupShuffleSplit` (`SPLIT_SEED =
61`). The development set is tuned with 5-fold `StratifiedGroupKFold` and 100
randomised configurations per model. Estimation, search, imputation, and
bootstrap operations use `ESTIMATOR_SEED = 42`.

All imputation, scaling, and one-hot encoding occur inside the fitted pipeline:

- numeric/binary values: iterative imputation by chained equations using
  `IterativeImputer(BayesianRidge, max_iter=10, sample_posterior=False)`, then
  `StandardScaler`. The imputer is fitted once per fold and produces one
  completed dataset; no multiple datasets are pooled;
- organism category: most-frequent imputation, then one-hot encoding with
  unknown categories ignored.

The operating threshold is selected by the Youden index from development-set
predictions, frozen, and applied to the held-out test set. The script also writes
the development/test comparison, including early-vitrectomy balance; procedure
is descriptive only.

Outputs include fitted pipelines, cross-validation summaries, held-out metrics,
and held-out predictions. These contain identifiers or derived patient data and
are git-ignored.
