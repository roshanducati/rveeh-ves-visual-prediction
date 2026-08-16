"""
02_train.py
===========
Model development, training, and hyperparameter tuning for the Victorian
Endophthalmitis outcome-prediction study.

Input
-----
  data/processed_episodes.csv: produced by 01_preprocess.py

Outputs
-------
  results/models/: serialised model objects (.joblib)
  results/cv_results.csv: per-fold and aggregate CV metrics
  results/test_predictions.csv: hold-out test-set predictions
  results/feature_columns.txt: ordered feature names used at training time

Study design
------------
The dataset is split 80/20 into a development set and a held-out test set
using GroupShuffleSplit (grouped by patient UR number) so that no patient's
data appears in both sets.  Cross-validation likewise uses StratifiedGroupKFold
to enforce patient-level integrity within folds.

Imputation strategy
-------------------
Missing values are handled within a scikit-learn Pipeline so that leakage
from the test fold is impossible:

  Numeric features: iterative imputation by chained equations using
                    IterativeImputer with a BayesianRidge estimator over 10
                    iterations (sample_posterior=False, add_indicator=False),
                    followed by StandardScaler. It is fitted once per fold and
                    produces one completed dataset.
  Categorical: SimpleImputer (most-frequent), then one-hot encoding.

Command-line flags
------------------
  --strict        Read from data_strict/ and write to results_strict/.
  --nomicro       Exclude culture_positive and organism_cat features.
                  Writes to results[_strict]_nomicro/.
  --models M ...  Space-separated list of models to train (default: RF XGB).

Models available
----------------
  RF: Random Forest Classifier
  XGB: XGBoost Classifier
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform, uniform

from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)
from sklearn.model_selection import (
    StratifiedGroupKFold, GroupShuffleSplit, RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Overridden by CLI flags in main()
IN_FILE   = Path("data/processed_episodes.csv")
OUT_DIR   = Path("results")
MDL_DIR   = OUT_DIR / "models"

# Fixed values used for the reported analysis. SPLIT_SEED controls the single
# patient-level development/test partition; ESTIMATOR_SEED controls model
# fitting, hyperparameter search and bootstrap resampling.
SPLIT_SEED     = 61
ESTIMATOR_SEED = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5
N_ITER_SEARCH = 100

# -----------------------------------------------------------------------------
# 1. FEATURE DEFINITIONS
# -----------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age_years",
    # Presenting VA enters the model ONLY as this continuous logMAR value.
    # The binned `pres_poor_va` (in processed_episodes.csv) is descriptive-only
    # (Table 1) and deliberately NOT listed here.
    "pres_logmar",
    "gender_female",
    "fundus_visible",
    "rapd_present",
    "diabetes",
    "immune_suppressed",
    # prior_surgery removed 2026-06-05: it is REDCap surgery_yn, displayed only
    # for surgical aetiologies (codes 1/2/3/98), so it is largely collinear with
    # the aetiology flags and its "missingness" encodes non-surgical aetiology.
    # Retained in processed_episodes.csv as a descriptive
    # (Table 1) variable only.
    "etiol_cataract_sx",
    "etiol_glaucoma_sx",
    "etiol_trauma",
    "etiol_corneal_ulcer",
    "etiol_endogenous",
    "etiol_ivi",
    "etiol_other",
    "culture_positive",
]
CATEGORICAL_FEATURES = ["organism_cat"]

# Microbiology-excluded feature set
NUMERIC_FEATURES_NOMICRO = [f for f in NUMERIC_FEATURES if f != "culture_positive"]
CATEGORICAL_FEATURES_NOMICRO = []

TARGET = "poor_outcome"


# -----------------------------------------------------------------------------
# 2. DATA LOADING & PATIENT-LEVEL SPLIT
# -----------------------------------------------------------------------------

def load_and_split(num_feats, cat_feats):
    df = pd.read_csv(IN_FILE)
    log.info("Loaded %d episodes from %s", len(df), IN_FILE)

    df_known = df.dropna(subset=[TARGET]).copy()
    log.info("Episodes with known outcome: %d", len(df_known))

    y      = df_known[TARGET].astype(int)
    X      = df_known[num_feats + cat_feats].copy()
    groups = df_known["rveeh_ur"].values

    pos = y.sum(); neg = len(y) - pos
    log.info("Class balance: poor: %d (%.1f%%)  good: %d (%.1f%%)",
             pos, 100*pos/len(y), neg, 100*neg/len(y))
    log.info("Unique patients: %d  (multi-episode patients: %d)",
             len(np.unique(groups)),
             sum(v > 1 for v in pd.Series(groups).value_counts()))

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=SPLIT_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train    = groups[train_idx]

    # Verify no patient leakage
    pts_train = set(groups[train_idx])
    pts_test  = set(groups[test_idx])
    overlap   = pts_train & pts_test
    if overlap:
        log.warning("Patient leakage detected: %d patients in both sets!", len(overlap))
    else:
        log.info("Patient-level split verified; no overlap between train and test.")

    log.info("Train: %d episodes (%d patients)   Test: %d episodes (%d patients)",
             len(X_train), len(pts_train), len(X_test), len(pts_test))

    return X_train, X_test, y_train, y_test, df_known, groups_train, train_idx, test_idx


# -----------------------------------------------------------------------------
# 2b. TRAIN VS TEST CHARACTERISTICS TABLE
# -----------------------------------------------------------------------------

def write_train_test_comparison(processed_path, data_out_dir, strict):
    """
    Emit a Table-1-style comparison of demographic and clinical features
    between the development (train) and held-out (test) sets, with a p-value
    per row (MWU for continuous, chi-squared / Fisher's exact for categorical).
    Output: `data_strict/train_test_comparison.csv`, consumed by
    generate_manuscript.py.

    This is the data-level characterisation of the patient split (not model
    performance) and is therefore computed once, regardless of which model
    feature set is being trained.
    """
    from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

    df = pd.read_csv(processed_path)
    df_known = df.dropna(subset=[TARGET]).copy()
    # Match the manuscript calculation: every analysed episode is classified as
    # early vitrectomy or not, with unknown/non-vitrectomy procedures in the
    # latter group. This variable is descriptive only and never enters a model.
    df_known["early_vitrectomy"] = (
        df_known["procedure_group"] == "vitrectomy"
    ).astype(float)
    groups   = df_known["rveeh_ur"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=SPLIT_SEED)
    train_idx, test_idx = next(gss.split(df_known, df_known[TARGET],
                                          groups=groups))
    train_df = df_known.iloc[train_idx]
    test_df  = df_known.iloc[test_idx]

    rows = []

    def section(title):
        rows.append({"Variable": f"--- {title} ---",
                     "Train": "", "Test": "", "p": "", "Stat": ""})

    def cont_row(label, col):
        tr = train_df[col].dropna()
        te = test_df[col].dropna()
        if len(tr) == 0 or len(te) == 0:
            return
        p = mannwhitneyu(tr, te, alternative="two-sided").pvalue
        rows.append({
            "Variable": label,
            "Train":    f"{tr.median():.1f} ({tr.quantile(0.25):.1f}-"
                         f"{tr.quantile(0.75):.1f})  [n={len(tr)}]",
            "Test":     f"{te.median():.1f} ({te.quantile(0.25):.1f}-"
                         f"{te.quantile(0.75):.1f})  [n={len(te)}]",
            "p":        f"{p:.3f}",
            "Stat":     "MWU",
        })

    def bin_row(label, col):
        tr = train_df[col].dropna()
        te = test_df[col].dropna()
        if len(tr) == 0 or len(te) == 0:
            return
        n_tr_pos = int((tr == 1).sum()); n_tr = len(tr)
        n_te_pos = int((te == 1).sum()); n_te = len(te)
        ct = np.array([[n_tr_pos, n_tr - n_tr_pos],
                       [n_te_pos, n_te - n_te_pos]])
        if (ct < 5).any():
            p = fisher_exact(ct)[1]; stat = "Fisher"
        else:
            p = chi2_contingency(ct)[1]; stat = "Chi2"
        rows.append({
            "Variable": label,
            "Train":    f"{n_tr_pos}/{n_tr} ({100*n_tr_pos/n_tr:.1f}%)",
            "Test":     f"{n_te_pos}/{n_te} ({100*n_te_pos/n_te:.1f}%)",
            "p":        f"{p:.3f}",
            "Stat":     stat,
        })

    def cat_row(label, col):
        tr_counts = train_df[col].value_counts()
        te_counts = test_df[col].value_counts()
        cats = sorted(set(tr_counts.index) | set(te_counts.index))
        ct = np.array([[tr_counts.get(c, 0) for c in cats],
                       [te_counts.get(c, 0) for c in cats]])
        try:
            p = chi2_contingency(ct)[1]; stat = "Chi2"
        except ValueError:
            p = np.nan; stat = "Chi2"
        n_tr = len(train_df); n_te = len(test_df)
        rows.append({
            "Variable": label, "Train": "", "Test": "",
            "p":        f"{p:.3f}" if not np.isnan(p) else "",
            "Stat":     stat,
        })
        for c in cats:
            tr_n = int(tr_counts.get(c, 0))
            te_n = int(te_counts.get(c, 0))
            rows.append({
                "Variable": f"  {c}",
                "Train":    f"{tr_n}/{n_tr} ({100*tr_n/n_tr:.1f}%)",
                "Test":     f"{te_n}/{n_te} ({100*te_n/n_te:.1f}%)",
                "p": "", "Stat": "",
            })

    section("Demographics")
    cont_row("Age, years  median (IQR)", "age_years")
    bin_row("Female sex", "gender_female")

    section("Presenting VA")
    cont_row("Presenting VA, logMAR  median (IQR)", "pres_logmar")
    pres_label = ("Presenting VA > 6/60 (logMAR > 1.0)" if strict
                  else "Presenting VA >= 6/60 (logMAR >= 1.0)")
    bin_row(pres_label, "pres_poor_va")

    section("Clinical signs")
    bin_row("Fundus visible",  "fundus_visible")
    bin_row("RAPD present",    "rapd_present")

    section("Comorbidities")
    bin_row("Diabetes mellitus",        "diabetes")
    bin_row("Immune suppression",       "immune_suppressed")
    bin_row("Prior ophthalmic surgery", "prior_surgery")

    section("Aetiology")
    bin_row("Post-cataract/IOL surgery",      "etiol_cataract_sx")
    bin_row("Post-glaucoma/drainage surgery", "etiol_glaucoma_sx")
    bin_row("Penetrating eye injury",         "etiol_trauma")
    bin_row("Corneal ulcer",                  "etiol_corneal_ulcer")
    bin_row("Metastatic/endogenous",          "etiol_endogenous")
    bin_row("Post-intravitreal injection",    "etiol_ivi")
    bin_row("Other ocular procedure",         "etiol_other")

    section("Microbiology")
    bin_row("Culture positive", "culture_positive")
    cat_row("Organism category", "organism_cat")

    section("Initial procedure (descriptive only; excluded from predictors)")
    bin_row("Early pars plana vitrectomy", "early_vitrectomy")

    section("Outcome")
    outcome_label = ("Poor outcome (final VA > 6/60)" if strict
                     else "Poor outcome (final VA >= 6/60)")
    bin_row(outcome_label, TARGET)
    cont_row("Final VA, logMAR  median (IQR)", "final_logmar")

    df_out  = pd.DataFrame(rows, columns=["Variable", "Train", "Test", "p", "Stat"])
    out_path = data_out_dir / "train_test_comparison.csv"
    df_out.to_csv(out_path, index=False)
    log.info("Saved: %s", out_path)


# -----------------------------------------------------------------------------
# 3. PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------

def make_preprocessor(num_feats, cat_feats):
    numeric_pipeline = Pipeline([
        ("iterative_imputer",
         IterativeImputer(estimator=BayesianRidge(),
                          max_iter=10,
                          sample_posterior=False,
                          add_indicator=False,
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


# -----------------------------------------------------------------------------
# 4. MODEL DEFINITIONS & HYPERPARAMETER SPACES
# -----------------------------------------------------------------------------

def imbalance_ratio(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return float(n_neg) / float(n_pos)


def get_models_and_spaces(spw, model_subset):
    all_models = [
        (
            "RF",
            RandomForestClassifier(n_jobs=-1, class_weight="balanced",
                                   random_state=ESTIMATOR_SEED),
            {
                "clf__n_estimators":      randint(200, 1000),
                "clf__max_depth":         [None, 5, 10, 15, 20],
                "clf__min_samples_leaf":  randint(1, 30),
                "clf__max_features":      ["sqrt", "log2", 0.5, 0.7],
                "clf__min_samples_split": randint(2, 20),
            }
        ),
        (
            "XGB",
            xgb.XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=spw,
                random_state=ESTIMATOR_SEED,
                n_jobs=-1,
            ),
            {
                "clf__n_estimators":      randint(100, 800),
                "clf__max_depth":         randint(2, 8),
                "clf__learning_rate":     loguniform(0.005, 0.3),
                "clf__subsample":         uniform(0.5, 0.5),
                "clf__colsample_bytree":  uniform(0.4, 0.6),
                "clf__min_child_weight":  randint(1, 20),
                "clf__gamma":             uniform(0, 1),
                "clf__reg_alpha":         loguniform(1e-4, 10),
                "clf__reg_lambda":        loguniform(1e-4, 10),
            }
        ),
    ]
    return [(n, e, p) for n, e, p in all_models if n in model_subset]


# -----------------------------------------------------------------------------
# 5. TRAINING LOOP
# -----------------------------------------------------------------------------

def tune_and_train(name, estimator, param_dist, preprocessor,
                   X_train, y_train, groups_train):
    log.info("\nTraining %s", name)

    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", estimator),
    ])

    cv = StratifiedGroupKFold(n_splits=CV_FOLDS)

    scoring = {"auroc": "roc_auc", "auprc": "average_precision"}
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring=scoring,
        refit="auroc",
        cv=cv,
        n_jobs=-1,
        random_state=ESTIMATOR_SEED,
        return_train_score=True,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups_train)

    best_idx = search.best_index_
    cv_auc    = search.cv_results_["mean_test_auroc"][best_idx]
    cv_auc_sd = search.cv_results_["std_test_auroc"][best_idx]
    cv_ap     = search.cv_results_["mean_test_auprc"][best_idx]
    cv_ap_sd  = search.cv_results_["std_test_auprc"][best_idx]

    log.info("%s  best CV AUROC = %.4f (+/- %.4f)  CV AUPRC = %.4f (+/- %.4f)",
             name, cv_auc, cv_auc_sd, cv_ap, cv_ap_sd)
    log.info("Best params: %s", json.dumps(
        {k: (float(v) if isinstance(v, (np.floating, float)) else v)
         for k, v in search.best_params_.items()}, indent=2))

    return search


def collect_cv_results(searches):
    rows = []
    for name, search in searches.items():
        idx = search.best_index_
        rows.append({
            "model":                 name,
            "best_cv_auroc_mean":    round(search.cv_results_["mean_test_auroc"][idx], 4),
            "best_cv_auroc_std":     round(search.cv_results_["std_test_auroc"][idx], 4),
            "best_cv_auprc_mean":    round(search.cv_results_["mean_test_auprc"][idx], 4),
            "best_cv_auprc_std":     round(search.cv_results_["std_test_auprc"][idx], 4),
            "best_train_auroc_mean": round(search.cv_results_["mean_train_auroc"][idx], 4),
            "n_iter_searched":       N_ITER_SEARCH,
            "cv_folds":              CV_FOLDS,
            "best_params":           str(search.best_params_),
        })
    return pd.DataFrame(rows).sort_values("best_cv_auroc_mean", ascending=False)


# -----------------------------------------------------------------------------
# 6. HOLD-OUT EVALUATION
# -----------------------------------------------------------------------------

N_BOOTSTRAP = 1000


def _bootstrap_ci(y_true, y_score, metric_fn, n_boot=N_BOOTSTRAP, seed=ESTIMATOR_SEED):
    """Bootstrap 95% CI for a scalar metric. Returns (lower, upper)."""
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate_on_test(searches, X_train, y_train, X_test, y_test):
    """Evaluate each tuned model on the held-out test set.

    Operating-point convention
    --------------------------
    The Youden-optimal probability threshold is derived from the **training
    set's in-sample predicted probabilities** (per model) and then frozen and
    applied to the held-out test set. Sensitivity / specificity / PPV / NPV /
    confusion-matrix counts therefore correspond to a threshold that was *not*
    chosen on test data. This removes the operating-point optimism that arises
    when Youden is recomputed on the same test set the metrics are reported on.
    The chosen threshold's source is recorded in `threshold_source`.
    """
    from sklearn.metrics import roc_curve, confusion_matrix

    rows    = []
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values

    for name, search in searches.items():
        pipe  = search.best_estimator_

        # Lock the operating point from the *training* ROC, then apply to test.
        proba_train = pipe.predict_proba(X_train)[:, 1]
        fpr_tr, tpr_tr, thr_tr = roc_curve(y_train, proba_train)
        j_tr_idx = np.argmax(tpr_tr - fpr_tr)
        opt_thr  = float(thr_tr[j_tr_idx])

        proba = pipe.predict_proba(X_test)[:, 1]
        pred_df[f"prob_{name}"] = proba

        auroc = roc_auc_score(y_test, proba)
        auprc = average_precision_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)

        yt = y_test.values
        auroc_lo, auroc_hi = _bootstrap_ci(yt, proba, roc_auc_score)
        auprc_lo, auprc_hi = _bootstrap_ci(yt, proba, average_precision_score)

        y_pred = (proba >= opt_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        rows.append({
            "model":            name,
            "test_auroc":       round(auroc, 4),
            "auroc_lower_95":   round(auroc_lo, 4),
            "auroc_upper_95":   round(auroc_hi, 4),
            "test_auprc":       round(auprc, 4),
            "auprc_lower_95":   round(auprc_lo, 4),
            "auprc_upper_95":   round(auprc_hi, 4),
            "brier_score":      round(brier, 4),
            # Keep enough precision that reapplying the saved threshold to
            # test_predictions.csv reproduces the stored confusion matrix.
            "opt_threshold":    round(opt_thr, 6),
            "threshold_source": "train_youden",
            "sensitivity":      round(float(sens), 3),
            "specificity":      round(float(spec), 3),
            "ppv":              round(float(ppv), 3),
            "npv":              round(float(npv), 3),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })
        log.info("%s  TEST  AUROC=%.4f (95%%CI %.4f-%.4f)  AUPRC=%.4f  "
                 "Brier=%.4f  thr=%.3f (train-Youden)  "
                 "Sen=%.3f  Spe=%.3f  PPV=%.3f  NPV=%.3f",
                 name, auroc, auroc_lo, auroc_hi, auprc, brier,
                 opt_thr, sens, spec, ppv, npv)

    return pd.DataFrame(rows).sort_values("test_auroc", ascending=False), pred_df


# -----------------------------------------------------------------------------
# 7. MAIN
# -----------------------------------------------------------------------------

def main():
    global IN_FILE, OUT_DIR, MDL_DIR

    parser = argparse.ArgumentParser(description="Train endophthalmitis outcome models.")
    parser.add_argument("--strict", action="store_true",
        help="Use strict-threshold data (data_strict/) and write to results_strict/.")
    parser.add_argument("--nomicro", action="store_true",
        help="Exclude microbiology features (culture_positive, organism_cat).")
    parser.add_argument("--models", nargs="+",
        default=["RF", "XGB"],
        choices=["RF", "XGB"],
        help="Models to train (default: RF XGB).")
    args = parser.parse_args()

    log.info("Fixed random seeds: patient split=%d; estimators/search/bootstrap=%d",
             SPLIT_SEED, ESTIMATOR_SEED)

    # Feature set
    if args.nomicro:
        num_feats = NUMERIC_FEATURES_NOMICRO
        cat_feats = CATEGORICAL_FEATURES_NOMICRO
        log.info("Feature set: NO-MICROBIOLOGY (culture_positive and organism_cat excluded)")
    else:
        num_feats = NUMERIC_FEATURES
        cat_feats = CATEGORICAL_FEATURES
        log.info("Feature set: FULL (including microbiology)")

    # Input/output directories
    if args.strict:
        IN_FILE = Path("data_strict/processed_episodes.csv")
        base    = "results_strict"
    else:
        IN_FILE = Path("data/processed_episodes.csv")
        base    = "results"

    if args.nomicro:
        base += "_nomicro"

    OUT_DIR = Path(base)
    MDL_DIR = OUT_DIR / "models"

    OUT_DIR.mkdir(exist_ok=True)
    MDL_DIR.mkdir(exist_ok=True)

    log.info("Output directory: %s", OUT_DIR)

    X_train, X_test, y_train, y_test, df_known, groups_train, _, test_idx = (
        load_and_split(num_feats, cat_feats))

    # Emit the train/test demographic comparison once (full-feature run only;
    # the patient split itself does not depend on the feature subset).
    if not args.nomicro:
        write_train_test_comparison(
            processed_path=IN_FILE,
            data_out_dir=IN_FILE.parent,
            strict=args.strict,
        )

    spw = imbalance_ratio(y_train)
    log.info("Scale-pos-weight (neg/pos ratio): %.2f", spw)

    preprocessor = make_preprocessor(num_feats, cat_feats)

    models_and_spaces = get_models_and_spaces(spw, args.models)
    log.info("Training models: %s", [m[0] for m in models_and_spaces])

    searches = {}
    for name, estimator, param_dist in models_and_spaces:
        searches[name] = tune_and_train(
            name, estimator, param_dist, preprocessor,
            X_train, y_train, groups_train)

    # CV summary
    cv_df = collect_cv_results(searches)
    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)
    log.info("\nCV results:\n%s",
             cv_df[["model", "best_cv_auroc_mean", "best_cv_auroc_std"]].to_string(index=False))

    # Test-set evaluation (threshold locked from training-set Youden)
    test_metrics, pred_df = evaluate_on_test(
        searches, X_train, y_train, X_test, y_test)
    test_metrics.to_csv(OUT_DIR / "test_metrics.csv", index=False)
    # Attach the true episode identifiers (rveeh_ur + admission_date) so that
    # downstream scripts can join predictions on episode identity rather than a
    # collision-prone composite of feature values.
    # df_known.iloc[test_idx] is row-aligned with X_test (hence pred_df).
    test_ids = df_known.iloc[test_idx][["rveeh_ur", "admission_date"]].reset_index(drop=True)
    pred_df  = pred_df.reset_index(drop=True)
    pred_df.insert(0, "rveeh_ur",       test_ids["rveeh_ur"].values)
    pred_df.insert(1, "admission_date", test_ids["admission_date"].values)
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # Save models
    for name, search in searches.items():
        joblib.dump(search.best_estimator_, MDL_DIR / f"{name}_pipeline.joblib")
        log.info("Saved model: %s", MDL_DIR / f"{name}_pipeline.joblib")

    # Save feature column order and test labels
    feat_cols = num_feats + cat_feats
    (OUT_DIR / "feature_columns.txt").write_text("\n".join(feat_cols))
    np.save(OUT_DIR / "y_test.npy", y_test.values)

    log.info("\nTraining complete.")


if __name__ == "__main__":
    main()
