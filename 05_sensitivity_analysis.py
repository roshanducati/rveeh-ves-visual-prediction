"""
05_sensitivity_analysis.py
==========================
Missing-data sensitivity analysis for the Victorian Endophthalmitis ML study.

Motivation
----------
Several predictors are recorded for only a minority of episodes in the full
registry export:
    immune_suppressed   93.0% missing  (119/1700 observed)
    rapd_present        60.9% missing  (665/1700 observed)

The submitted feature set excludes prior_surgery. This REDCap field is shown
only for surgical aetiologies and is therefore collinear with the aetiology
flags, so it is outside this ablation.

The headline model handles these values with IterativeImputer fitted once
within each training fold. It produces one completed dataset and relies on a
missing-at-random (MAR) assumption. MAR cannot be tested from the observed data
alone, and for a registry field that is blank in 93% of episodes the
missingness is plausibly informative. The sensitivity analysis therefore tests
whether the conclusions depend on these sparsely recorded variables.

This script provides that sensitivity analysis. Rather than defend MAR, it asks
the more decision-relevant question: **does discrimination depend on these
sparsely-recorded variables at all?** If AUROC is unchanged when they are
dropped, the conclusions are robust regardless of the missingness mechanism.

Design
------
Tiered (nested) ablation, each compared against the full-feature model:
    full                    all numeric features (headline feature set)
    drop_immune             - immune_suppressed                       (93%)
    drop_immune_rapd        - immune_suppressed, rapd_present         (93%, 61%)

Microbiology features (culture_positive, organism_cat) are RETAINED in every
tier; excluding them is a separate analysis (02_train.py --nomicro).

For each tier, both reported models (Random Forest and the lead XGBoost model)
are re-tuned from scratch (RandomizedSearchCV, 100 iter,
StratifiedGroupKFold) on the SAME patient-level train/test split (seed 61) used
for the headline model. Re-tuning per tier (rather than reusing the full-feature
hyperparameters) gives each reduced feature set a fair chance and avoids biasing
the comparison towards the full model.

Because every tier shares an identical held-out test set, the full vs reduced
comparison is paired: AUROC differences are tested with DeLong's method and a
paired bootstrap 95% CI (same resampled indices applied to both models).

Input
-----
  data_strict/processed_episodes.csv: produced by 01_preprocess.py --strict

Outputs (results_strict_sensitivity/)
-------------------------------------
  sensitivity_missingness.csv: per-tier performance + comparison vs full
  test_predictions.csv: per-episode predicted probabilities, all tiers

Usage
-----
  python 05_sensitivity_analysis.py                 # RF + XGB, 100 iterations
  python 05_sensitivity_analysis.py --n_iter 30     # quicker pass
  python 05_sensitivity_analysis.py --models XGB    # single model
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import randint, loguniform, uniform

from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedGroupKFold, GroupShuffleSplit, RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_FILE = Path("data_strict/processed_episodes.csv")
OUT_DIR   = Path("results_strict_sensitivity")

# Mirror 02_train.py exactly so this analysis is comparable to the headline run.
SPLIT_SEED     = 61    # fixed patient-level train/test split seed
ESTIMATOR_SEED = 42    # fixed RNG for estimators / bootstrap / CV
TEST_SIZE      = 0.20
CV_FOLDS       = 5
N_BOOTSTRAP    = 1000

NUMERIC_FEATURES = [
    "age_years", "pres_logmar", "gender_female", "fundus_visible",
    "rapd_present", "diabetes", "immune_suppressed",
    "etiol_cataract_sx", "etiol_glaucoma_sx", "etiol_trauma",
    "etiol_corneal_ulcer", "etiol_endogenous", "etiol_ivi", "etiol_other",
    "culture_positive",
]
CATEGORICAL_FEATURES = ["organism_cat"]
TARGET = "poor_outcome"

# Tiered (nested) ablation. Each tier drops the listed numeric variables from
# NUMERIC_FEATURES; categorical (microbiology) features are unchanged.
# The submitted feature set excludes prior_surgery. The high-missingness
# ablation therefore covers immune_suppressed (93%) and rapd_present (61%).
ABLATION_TIERS = [
    ("full",             []),
    ("drop_immune",      ["immune_suppressed"]),
    ("drop_immune_rapd", ["immune_suppressed", "rapd_present"]),
]

# XGBoost hyperparameter search space, identical to 02_train.py.
XGB_PARAM_SPACE = {
    "clf__n_estimators":     randint(100, 800),
    "clf__max_depth":        randint(2, 8),
    "clf__learning_rate":    loguniform(0.005, 0.3),
    "clf__subsample":        uniform(0.5, 0.5),
    "clf__colsample_bytree": uniform(0.4, 0.6),
    "clf__min_child_weight": randint(1, 20),
    "clf__gamma":            uniform(0, 1),
    "clf__reg_alpha":        loguniform(1e-4, 10),
    "clf__reg_lambda":       loguniform(1e-4, 10),
}

RF_PARAM_SPACE = {
    "clf__n_estimators":      randint(200, 1000),
    "clf__max_depth":         [None, 5, 10, 15, 20],
    "clf__min_samples_leaf":  randint(1, 30),
    "clf__max_features":      ["sqrt", "log2", 0.5, 0.7],
    "clf__min_samples_split": randint(2, 20),
}


# -----------------------------------------------------------------------------
# Preprocessing & model builders (mirror 02_train.py)
# -----------------------------------------------------------------------------

def make_preprocessor(num_feats, cat_feats):
    numeric_pipeline = Pipeline([
        ("iterative_imputer",
         IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                          sample_posterior=False, add_indicator=False,
                          random_state=ESTIMATOR_SEED)),
        ("scale", StandardScaler()),
    ])
    transformers = [("num", numeric_pipeline, num_feats)]
    if cat_feats:
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, cat_feats))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_estimator(name, spw):
    if name == "XGB":
        return xgb.XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                                 random_state=ESTIMATOR_SEED, n_jobs=-1), XGB_PARAM_SPACE
    if name == "RF":
        return RandomForestClassifier(n_jobs=-1, class_weight="balanced",
                                      random_state=ESTIMATOR_SEED), RF_PARAM_SPACE
    raise ValueError(f"Unsupported model: {name}")


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def _bootstrap_ci(y_true, y_score, metric_fn, seed=ESTIMATOR_SEED):
    """Bootstrap 95% CI for a scalar metric. Returns (lower, upper)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def _delong_components(y_true, y_score):
    """Kernel (structural) components used by DeLong's test."""
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    V10 = np.array([
        np.mean(y_score[pos[i]] > y_score[neg]) +
        0.5 * np.mean(y_score[pos[i]] == y_score[neg])
        for i in range(len(pos))
    ])
    V01 = np.array([
        np.mean(y_score[pos] > y_score[neg[j]]) +
        0.5 * np.mean(y_score[pos] == y_score[neg[j]])
        for j in range(len(neg))
    ])
    return V10.mean(), V10, V01


def delong_compare(y_true, prob_a, prob_b):
    """DeLong's paired test for two AUROCs on the same test set.

    Returns (auc_a, auc_b, diff, z, p_two_sided).
    Reference: DeLong et al. (1988) Biometrics 44(3):837-845.
    """
    y_true = np.asarray(y_true, dtype=int)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    auc_a, V10_a, V01_a = _delong_components(y_true, prob_a)
    auc_b, V10_b, V01_b = _delong_components(y_true, prob_b)
    S10 = np.cov(np.vstack([V10_a, V10_b])) / n1
    S01 = np.cov(np.vstack([V01_a, V01_b])) / n0
    S   = S10 + S01
    L        = np.array([1.0, -1.0])
    var_diff = float(L @ S @ L)
    if var_diff <= 0:
        return auc_a, auc_b, auc_a - auc_b, np.nan, np.nan
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * stats.norm.sf(abs(z)))
    return auc_a, auc_b, auc_a - auc_b, z, p


def paired_bootstrap_diff_ci(y_true, prob_a, prob_b, seed=ESTIMATOR_SEED):
    """Paired bootstrap 95% CI for AUROC(a) - AUROC(b) on the same test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], prob_a[idx]) -
                     roc_auc_score(y_true[idx], prob_b[idx]))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# -----------------------------------------------------------------------------
# Core: split, tune one tier, evaluate
# -----------------------------------------------------------------------------

def load_and_split():
    df = pd.read_csv(DATA_FILE)
    df_known = df.dropna(subset=[TARGET]).copy()
    log.info("Loaded %d episodes; %d with known outcome.", len(df), len(df_known))

    y      = df_known[TARGET].astype(int)
    groups = df_known["rveeh_ur"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=SPLIT_SEED)
    train_idx, test_idx = next(gss.split(df_known, y, groups=groups))

    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        log.warning("Patient leakage: %d patients in both sets!", len(overlap))
    else:
        log.info("Patient-level split verified; no train/test overlap.")
    log.info("Train: %d episodes   Test: %d episodes",
             len(train_idx), len(test_idx))
    return df_known, train_idx, test_idx


def tune_and_evaluate(model_name, num_feats, cat_feats,
                      df_known, train_idx, test_idx, n_iter):
    """Re-tune `model_name` on `num_feats + cat_feats`, evaluate on the held-out
    test set. Returns (metrics_dict, test_proba)."""
    y      = df_known[TARGET].astype(int)
    X      = df_known[num_feats + cat_feats].copy()
    groups = df_known["rveeh_ur"].values

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train    = groups[train_idx]

    n_neg = int((y_train == 0).sum()); n_pos = int((y_train == 1).sum())
    spw   = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    estimator, param_space = make_estimator(model_name, spw)
    pipe = Pipeline([("pre", make_preprocessor(num_feats, cat_feats)),
                     ("clf", estimator)])
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS)
    search = RandomizedSearchCV(
        pipe, param_distributions=param_space, n_iter=n_iter,
        scoring={"auroc": "roc_auc", "auprc": "average_precision"},
        refit="auroc", cv=cv, n_jobs=-1, random_state=ESTIMATOR_SEED,
        return_train_score=False, verbose=0,
    )
    search.fit(X_train, y_train, groups=groups_train)
    best = search.best_estimator_
    cv_auc = float(search.cv_results_["mean_test_auroc"][search.best_index_])

    # Lock Youden threshold on train ROC, apply to test (matches 02_train.py).
    proba_train = best.predict_proba(X_train)[:, 1]
    fpr_tr, tpr_tr, thr_tr = roc_curve(y_train, proba_train)
    opt_thr = float(thr_tr[np.argmax(tpr_tr - fpr_tr)])

    proba = best.predict_proba(X_test)[:, 1]
    yt    = y_test.values
    auroc = roc_auc_score(yt, proba)
    auprc = average_precision_score(yt, proba)
    brier = brier_score_loss(yt, proba)
    auroc_lo, auroc_hi = _bootstrap_ci(yt, proba, roc_auc_score)

    y_pred = (proba >= opt_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    metrics = {
        "model":          model_name,
        "n_numeric":      len(num_feats),
        "n_predictors":   len(num_feats) + len(cat_feats),
        "cv_auroc":       round(cv_auc, 4),
        "test_auroc":     round(auroc, 4),
        "auroc_lower_95": round(auroc_lo, 4),
        "auroc_upper_95": round(auroc_hi, 4),
        "test_auprc":     round(auprc, 4),
        "brier_score":    round(brier, 4),
        "opt_threshold":  round(opt_thr, 3),
        "sensitivity":    round(float(sens), 3),
        "specificity":    round(float(spec), 3),
        "ppv":            round(float(ppv), 3),
        "npv":            round(float(npv), 3),
    }
    return metrics, proba, yt


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Missing-data sensitivity analysis (high-missingness "
                    "feature ablation).")
    parser.add_argument("--n_iter", type=int, default=100,
        help="RandomizedSearchCV iterations per tier (default: 100, matching "
             "02_train.py).")
    parser.add_argument("--models", nargs="+", default=["RF", "XGB"],
        choices=["XGB", "RF"],
        help="Models to evaluate (default: RF XGB; XGB is the lead model).")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    log.info("Output directory: %s", OUT_DIR)
    log.info("Fixed seeds: patient split=%d; estimators/search/bootstrap=%d",
             SPLIT_SEED, ESTIMATOR_SEED)
    log.info("Search iterations/tier: %d   Models: %s", args.n_iter, args.models)

    df_known, train_idx, test_idx = load_and_split()

    all_rows  = []
    pred_cols = {}
    y_true_ref = None

    for model_name in args.models:
        log.info("\n" + "=" * 70)
        log.info("MODEL: %s", model_name)
        log.info("=" * 70)

        tier_results = {}   # tier_name -> (metrics, proba)
        for tier_name, drop_vars in ABLATION_TIERS:
            num_feats = [f for f in NUMERIC_FEATURES if f not in drop_vars]
            log.info("\nTier '%s' (%s): %d numeric features",
                     tier_name,
                     "none dropped" if not drop_vars else "drop " + ", ".join(drop_vars),
                     len(num_feats))
            metrics, proba, yt = tune_and_evaluate(
                model_name, num_feats, CATEGORICAL_FEATURES,
                df_known, train_idx, test_idx, args.n_iter)

            if y_true_ref is None:
                y_true_ref = yt
            tier_results[tier_name] = (metrics, proba)
            pred_cols[f"prob_{model_name}_{tier_name}"] = proba
            log.info("%s/%s  test AUROC=%.4f (95%%CI %.4f-%.4f)  AUPRC=%.4f  "
                     "Sen=%.3f Spe=%.3f",
                     model_name, tier_name, metrics["test_auroc"],
                     metrics["auroc_lower_95"], metrics["auroc_upper_95"],
                     metrics["test_auprc"], metrics["sensitivity"],
                     metrics["specificity"])

        # Compare every reduced tier against the full-feature model (paired).
        full_metrics, full_proba = tier_results["full"]
        for tier_name, drop_vars in ABLATION_TIERS:
            metrics, proba = tier_results[tier_name]
            row = dict(metrics)
            row["tier"]         = tier_name
            row["dropped_vars"] = ";".join(drop_vars) if drop_vars else "(none)"
            if tier_name == "full":
                row.update({"auroc_diff_vs_full": 0.0,
                            "diff_lower_95": 0.0, "diff_upper_95": 0.0,
                            "delong_z": np.nan, "delong_p": np.nan})
            else:
                # Report reduced minus full, matching the manuscript's
                # "Change vs full" column (positive means the reduced tier had
                # the higher AUROC).
                auc_red, auc_full, diff, z, p = delong_compare(
                    y_true_ref, proba, full_proba)
                lo, hi = paired_bootstrap_diff_ci(y_true_ref, proba, full_proba)
                row.update({
                    "auroc_diff_vs_full": round(diff, 4),
                    "diff_lower_95": round(lo, 4),
                    "diff_upper_95": round(hi, 4),
                    "delong_z": round(z, 3) if not np.isnan(z) else np.nan,
                    "delong_p": round(p, 4) if not np.isnan(p) else np.nan,
                })
                log.info("%s/%s vs full: dAUROC=%.4f (95%%CI %.4f-%.4f)  "
                         "DeLong z=%.3f p=%.4f",
                         model_name, tier_name, diff, lo, hi,
                         row["delong_z"], row["delong_p"])
            all_rows.append(row)

    # Save summary table
    col_order = [
        "model", "tier", "dropped_vars", "n_predictors", "n_numeric",
        "cv_auroc", "test_auroc", "auroc_lower_95", "auroc_upper_95",
        "auroc_diff_vs_full", "diff_lower_95", "diff_upper_95",
        "delong_z", "delong_p",
        "test_auprc", "brier_score", "opt_threshold",
        "sensitivity", "specificity", "ppv", "npv",
    ]
    summary = pd.DataFrame(all_rows)[col_order]
    summary_path = OUT_DIR / "sensitivity_missingness.csv"
    summary.to_csv(summary_path, index=False)
    log.info("\nSaved: %s", summary_path)

    # Save per-episode predictions across all tiers
    pred_df = pd.DataFrame({"y_true": y_true_ref})
    for k, v in pred_cols.items():
        pred_df[k] = v
    pred_path = OUT_DIR / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info("Saved: %s", pred_path)

    # Console summary
    log.info("\n" + "=" * 70)
    log.info("SENSITIVITY ANALYSIS SUMMARY")
    log.info("=" * 70)
    log.info("\n%s", summary[
        ["model", "tier", "n_predictors", "test_auroc",
         "auroc_diff_vs_full", "delong_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
