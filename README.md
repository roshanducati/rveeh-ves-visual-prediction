# Predicting poor visual outcomes in endophthalmitis

Reproducibility code for *Development and internal validation of machine
learning models to predict poor visual outcomes in endophthalmitis*.

This repository contains the analytical code used for the submitted manuscript:
REDCap preprocessing, patient-level partitioning, model development, held-out
evaluation, explainability, and the reported sensitivity analyses. It contains
no patient data, derived patient-level tables, fitted models, or generated
figures.

The models are research outputs and have not undergone external validation.


## Analysis map

| Stage | Script | Manuscript role |
|---|---|---|
| 1 | `01_preprocess.py` | Consolidates registry records into episodes; decodes visual acuity; derives the strict outcome, 16 predictors, initial-procedure descriptors, follow-up duration, and data-quality audits. |
| 2 | `02_train.py` | Creates the fixed 80/20 patient-level partition, checks development/test comparability (including early vitrectomy), performs 5-fold grouped tuning, fits Random Forest and XGBoost, fixes the development-set Youden thresholds, and evaluates the held-out set. |
| 3 | `03_evaluate.py` | Produces discrimination, calibration, operating-point, decision-curve, fairness, feature-importance, SHAP, and full-versus-pre-culture analyses. |
| 4 | `participant_flow.py` | Reconstructs the participant flow and partition counts used in manuscript Figure 1. |
| 5 | `04_timing_analysis.py` | Summarises presentation-to-review, intervention, treatment, and discharge intervals. |
| 6 | `05_sensitivity_analysis.py` | Re-tunes both models after sequential removal of immune suppression and RAPD, the two highest-missingness predictors. |
| 7 | `06_imbalance_sensitivity.py` | Compares the reported class-weighted models with otherwise identical unweighted fits. |
| 8 | `07_followup_sensitivity.py` | Repeats XGBoost development at increasing minimum follow-up durations and reproduces the descriptive follow-up checks cited in the limitations. |

Plain-language script notes are provided in [`code_explanations/`](code_explanations).

The two reported feature sets are:

- Full model: 16 predictors, including documented culture positivity and
  organism category.
- Pre-culture model: 14 predictors, with both microbiology dimensions removed
  and the model fitted separately.

Initial procedure is retained only for the partition-balance analysis. It is
never a model predictor.

The culture indicator is coded as documented positive versus not documented
positive. The accompanying organism category distinguishes confirmed no growth
from unavailable culture status, so these states remain separate in the full
model. This coding is described in detail in [`DATA_SCHEMA.md`](DATA_SCHEMA.md).

## Environment

The analysis used Python 3.11. Package versions are pinned in
[`requirements.txt`](requirements.txt), including scikit-learn 1.6.1 and
XGBoost 3.2.0.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the data-free regression tests with:

```bash
python -m unittest discover -s tests -v
```

## Data access and placement

No data are distributed with this repository. The underlying registry contains
potentially identifiable clinical information and cannot be shared publicly.
De-identified data may be available on reasonable request, subject to Human
Research Ethics Committee approval and an appropriate data-sharing agreement.

Approved users should create `data/raw/` locally and place the REDCap workbook
at:

```text
data/raw/VictorianEndophthalm_DATA_2026-05-17_2131.xlsx
```

The required structure is described in [`DATA_SCHEMA.md`](DATA_SCHEMA.md). The
entire `data/` tree, all derived datasets, fitted models, result directories,
and common clinical-data formats are excluded by [`.gitignore`](.gitignore).

## Reproduce the submitted analysis

Run from the repository root:

```bash
# Preprocessing with the submitted outcome definition:
# poor outcome = final logMAR > 1.00 or loss of the eye.
python 01_preprocess.py --strict

# Full and pre-culture model development.
python 02_train.py --strict
python 02_train.py --strict --nomicro

# Held-out evaluation and manuscript analyses (XGBoost lead model).
python 03_evaluate.py --strict --lead_model XGB
python 03_evaluate.py --strict --nomicro --lead_model XGB

# Participant flow and timing summaries.
python participant_flow.py
python 04_timing_analysis.py

# Sensitivity analyses.
python 05_sensitivity_analysis.py
python 06_imbalance_sensitivity.py
python 07_followup_sensitivity.py
```

The same sequence is available through `run_pipeline.sh` and
`run_pipeline.ps1`.

The numbered evaluation files map to the submitted figures as follows:

| Submitted figure | Generated file stem |
|---|---|
| Figure 1, participant flow | `fig1_participant_flow` |
| Figure 2, ROC curves | `results_strict/figures/fig1_roc_curves` |
| Figure 3, XGBoost gain importance | `results_strict/figures/fig6_feature_importance_xgb` |
| Figure 4, concise SHAP summary | `results_strict/figures/fig7_shap_summary_concise` |
| Figure 5, decision curve | `results_strict/figures/fig9_decision_curve` |
| Figure SF1, precision-recall curves | `results_strict/figures/fig2_pr_curves` |
| Figure SF2, confusion matrices | `results_strict/figures/fig4_confusion_matrices` |
| Figure SF3, calibration | `results_strict/figures/fig3_calibration` |
| Figure SF4, Random Forest importance | `results_strict/figures/fig5_feature_importance_rf` |

Generated directories contain episode identifiers and are deliberately ignored.

## Fixed random values

The submitted analysis used:

| Value | Constant | Use |
|---:|---|---|
| `61` | `SPLIT_SEED` | Patient-level `GroupShuffleSplit` development/test partition. |
| `42` | `ESTIMATOR_SEED` | Estimator fitting, randomised hyperparameter search, bootstrap resampling, imputation, and SHAP background sampling. |

Imputation, scaling, and one-hot encoding are contained inside scikit-learn
pipelines fitted within each development fold. Patient grouping is used for
both the held-out partition and cross-validation.

Numeric and binary missing values are handled by iterative imputation by
chained equations, implemented with `IterativeImputer` and a `BayesianRidge`
estimator over 10 iterations. The imputer is fitted once within each fold with
posterior sampling disabled and produces one completed dataset; multiple
completed datasets are not generated or pooled.

## Citation and licence

Citation information: TBC

The code is released under the [`MIT License`](LICENSE); the licence confers no
rights to the underlying registry data.
