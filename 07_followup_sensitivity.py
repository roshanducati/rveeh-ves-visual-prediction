"""
07_followup_sensitivity.py
==========================
Minimum-follow-up sensitivity analysis for the Victorian Endophthalmitis ML study.

Rationale
---------
The outcome (poor visual outcome) is ascertained at the last documented visit,
and episodes with a good outcome were followed for longer than those with a poor
outcome (median 63 vs 38 days; univariable Mann-Whitney p < 0.001). Could the
poor-outcome label therefore be an artefact of measuring acuity too early on the
recovery curve in shorter-follow-up patients?

Design
------
Mirror the headline pipeline exactly (XGBoost lead model, split seed 61,
patient-level GroupShuffleSplit, single-dataset IterativeImputer and
StandardScaler fitted in-fold, RandomizedSearchCV with 100 iterations and
StratifiedGroupKFold, and the Youden threshold locked on the training ROC). The
only change is that the outcome-known cohort is
progressively restricted to a minimum follow-up duration BEFORE the split. Each
subset is re-tuned and re-evaluated from scratch on its own held-out test set.
This deliberately estimates performance within each eligible follow-up cohort,
but it also changes the composition of both the development and test sets. The
test rows are therefore not paired across minimum-follow-up thresholds. Changes
in performance reflect the eligibility restriction and the resulting sample
composition, and should be interpreted as a robustness check rather than as a
paired estimate of the effect of follow-up duration.

min_fu = 0 reproduces the headline cohort (all 1300 outcome-known episodes) and
recovers the published XGBoost AUROC (~0.863), confirming the mirror is faithful.

The `poor_rate` column is the quantity cited in the Discussion: the proportion of
poor outcomes remains in the 39-44% band as the minimum follow-up requirement is
raised to 180 days, rather than falling as it would if poor outcomes were largely
an artefact of premature assessment.

Input
-----
  data_strict/processed_episodes.csv   - produced by 01_preprocess.py --strict

Output (results_strict_followup/)
---------------------------------
  followup_sensitivity.csv   - per-subset cohort size, poor rate, and performance
  followup_descriptive_checks.csv - statistics cited in the limitations

Usage
-----
  python 07_followup_sensitivity.py
"""
import warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import randint, loguniform, mannwhitneyu, uniform
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve, confusion_matrix)
from sklearn.model_selection import (StratifiedGroupKFold, GroupShuffleSplit,
                                     RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_FILE = Path("data_strict/processed_episodes.csv")
OUT_DIR   = Path("results_strict_followup")
SPLIT_SEED, ESTIMATOR_SEED = 61, 42          # mirror 02_train.py
TEST_SIZE, CV_FOLDS, N_BOOTSTRAP = 0.20, 5, 1000
N_ITER = 100
FU_COL, TARGET = "days_to_final_visit", "poor_outcome"
# Thresholds span the range quoted in the Discussion ("39% to 44% at minimum
# follow-up thresholds of up to 180 days"), so every figure in that sentence is
# reproducible from this script. min_fu = 0 keeps all outcome-known episodes,
# including those with no recorded follow-up duration; every other level drops
# them, since the restriction cannot be evaluated without a duration.
MIN_FU_LEVELS = [0, 30, 60, 90, 120, 180]    # days

NUMERIC_FEATURES = [
    "age_years", "pres_logmar", "gender_female", "fundus_visible",
    "rapd_present", "diabetes", "immune_suppressed",
    "etiol_cataract_sx", "etiol_glaucoma_sx", "etiol_trauma",
    "etiol_corneal_ulcer", "etiol_endogenous", "etiol_ivi", "etiol_other",
    "culture_positive",
]
CATEGORICAL_FEATURES = ["organism_cat"]

XGB_PARAM_SPACE = {                          # identical to 02_train.py
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
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_feats))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _bootstrap_ci(y_true, y_score, metric_fn, seed=ESTIMATOR_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true); scores = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def descriptive_followup_checks(df_known):
    """Reproduce the follow-up statistics cited in the manuscript limitations."""
    observed = df_known.dropna(subset=[FU_COL]).copy()
    good = observed.loc[observed[TARGET] == 0, FU_COL]
    poor = observed.loc[observed[TARGET] == 1, FU_COL]
    followup_test = mannwhitneyu(good, poor, alternative="two-sided")

    at_least_90 = observed.loc[observed[FU_COL] >= 90].dropna(
        subset=["pres_logmar"]
    )
    va_good = at_least_90.loc[at_least_90[TARGET] == 0, "pres_logmar"]
    va_poor = at_least_90.loc[at_least_90[TARGET] == 1, "pres_logmar"]
    va_test = mannwhitneyu(va_good, va_poor, alternative="two-sided")

    short_poor = observed.loc[
        (observed[TARGET] == 1) & (observed[FU_COL] <= 14)
    ]
    n_eye_loss = int((short_poor["final_logmar"] >= 4.0).sum())

    return pd.DataFrame([
        {
            "analysis": "followup_duration_by_outcome",
            "n": len(observed),
            "group_0": "good_outcome",
            "n_0": len(good),
            "estimate_0": float(good.median()),
            "group_1": "poor_outcome",
            "n_1": len(poor),
            "estimate_1": float(poor.median()),
            "p_value": float(followup_test.pvalue),
        },
        {
            "analysis": "presenting_logmar_by_outcome_followup_ge_90d",
            "n": len(at_least_90),
            "group_0": "good_outcome",
            "n_0": len(va_good),
            "estimate_0": float(va_good.median()),
            "group_1": "poor_outcome",
            "n_1": len(va_poor),
            "estimate_1": float(va_poor.median()),
            "p_value": float(va_test.pvalue),
        },
        {
            "analysis": "eye_loss_among_poor_outcomes_followup_le_14d",
            "n": len(short_poor),
            "group_0": "eye_loss",
            "n_0": n_eye_loss,
            "estimate_0": (
                float(n_eye_loss / len(short_poor)) if len(short_poor) else np.nan
            ),
            "group_1": "all_short_followup_poor_outcomes",
            "n_1": len(short_poor),
            "estimate_1": 1.0,
            "p_value": np.nan,
        },
    ])


def evaluate_subset(df_sub, min_fu):
    y = df_sub[TARGET].astype(int)
    X = df_sub[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    groups = df_sub["rveeh_ur"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    assert not (set(groups[train_idx]) & set(groups[test_idx])), "patient leakage!"

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    g_tr = groups[train_idx]

    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    pipe = Pipeline([("pre", make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
                     ("clf", xgb.XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                                               random_state=ESTIMATOR_SEED, n_jobs=-1))])
    search = RandomizedSearchCV(
        pipe, XGB_PARAM_SPACE, n_iter=N_ITER,
        scoring={"auroc": "roc_auc", "auprc": "average_precision"},
        refit="auroc", cv=StratifiedGroupKFold(n_splits=CV_FOLDS),
        n_jobs=-1, random_state=ESTIMATOR_SEED, verbose=0)
    search.fit(X_tr, y_tr, groups=g_tr)
    best = search.best_estimator_
    cv_auc = float(search.cv_results_["mean_test_auroc"][search.best_index_])

    p_tr = best.predict_proba(X_tr)[:, 1]
    fpr, tpr, thr = roc_curve(y_tr, p_tr)
    opt_thr = float(thr[np.argmax(tpr - fpr)])

    proba = best.predict_proba(X_te)[:, 1]
    yt = y_te.values
    auroc = roc_auc_score(yt, proba)
    lo, hi = _bootstrap_ci(yt, proba, roc_auc_score)
    y_pred = (proba >= opt_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "min_fu_days": min_fu, "n_total": len(df_sub),
        "n_patients": int(pd.Series(groups).nunique()),
        "poor_rate": round(float(y.mean()), 3),
        "n_train": len(train_idx), "n_test": len(test_idx),
        "n_test_poor": int(yt.sum()), "cv_auroc": round(cv_auc, 4),
        "test_auroc": round(auroc, 4), "auroc_lo95": round(lo, 4),
        "auroc_hi95": round(hi, 4),
        "test_auprc": round(average_precision_score(yt, proba), 4),
        "brier": round(brier_score_loss(yt, proba), 4), "threshold": round(opt_thr, 3),
        "sens": round(float(sens), 3), "spec": round(float(spec), 3),
        "ppv": round(float(ppv), 3), "npv": round(float(npv), 3),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    df_known = df.dropna(subset=[TARGET]).copy()
    log.info("Loaded %d episodes; %d outcome-known.", len(df), len(df_known))
    log.info("Fixed random seeds: patient split=%d; estimators/search/bootstrap=%d",
             SPLIT_SEED, ESTIMATOR_SEED)

    checks = descriptive_followup_checks(df_known)
    checks_path = OUT_DIR / "followup_descriptive_checks.csv"
    checks.to_csv(checks_path, index=False)
    log.info("Saved: %s\n%s", checks_path.resolve(), checks.to_string(index=False))

    rows = []
    for min_fu in MIN_FU_LEVELS:
        sub = df_known.copy() if min_fu == 0 else df_known[df_known[FU_COL] >= min_fu].copy()
        log.info("\n=== min follow-up >= %d d : n=%d (poor=%.1f%%) ===",
                 min_fu, len(sub), 100 * sub[TARGET].mean())
        res = evaluate_subset(sub, min_fu)
        log.info("XGB  test AUROC=%.4f (%.4f-%.4f)  AUPRC=%.4f  Sen=%.3f Spe=%.3f  "
                 "n_test=%d (poor=%d)", res["test_auroc"], res["auroc_lo95"],
                 res["auroc_hi95"], res["test_auprc"], res["sens"], res["spec"],
                 res["n_test"], res["n_test_poor"])
        rows.append(res)

    out = pd.DataFrame(rows)
    outpath = OUT_DIR / "followup_sensitivity.csv"
    out.to_csv(outpath, index=False)
    log.info("\nSaved: %s", outpath.resolve())
    log.info("\n%s", out[["min_fu_days", "n_total", "poor_rate", "n_test",
                          "test_auroc", "auroc_lo95", "auroc_hi95",
                          "sens", "spec"]].to_string(index=False))


if __name__ == "__main__":
    main()
