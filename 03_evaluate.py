"""
03_evaluate.py
==============
Evaluation and visualisation for the Victorian Endophthalmitis ML study.

Inputs
------
  data/processed_episodes.csv: preprocessed data (from 01_preprocess.py)
  results/test_predictions.csv: hold-out probabilities (from 02_train.py)
  results/test_metrics.csv: per-model test metrics (from 02_train.py)
  results/cv_results.csv: cross-validation summary (from 02_train.py)
  results/models/*.joblib: fitted model pipelines (from 02_train.py)
  results/feature_columns.txt: ordered feature list (from 02_train.py)

Outputs (results/figures/)
--------------------------
  fig1_roc_curves.pdf/.png: multi-model ROC overlay
  fig2_pr_curves.pdf/.png: multi-model precision-recall overlay
  fig3_calibration.pdf/.png: calibration / reliability diagrams
  fig4_confusion_matrices.pdf/.png: confusion matrices at Youden threshold
  fig5_feature_importance_rf.pdf/.png: Random Forest impurity importance
  fig6_feature_importance_xgb.pdf/.png: XGBoost gain-based importance
  fig7_shap_summary.pdf/.png: SHAP beeswarm (lead model)
  fig7_shap_summary_concise.pdf/.png: manuscript SHAP beeswarm (top 10)
  fig8_shap_waterfall_examples.pdf/.png: SHAP waterfall for selected cases
  table1_patient_characteristics.csv: baseline cohort table (Table 1)
  table2_model_performance.csv: test-set performance table (Table 2)
  table3_micro_nomicro_comparison.csv: paired microbiology comparison
  table4_clinical_thresholds.csv: clinical operating points table
  table_decision_curve.csv: decision-curve net benefit
  table_fairness_subgroups.csv: held-out sex and age subgroup AUROCs

Reporting conventions
---------------------
All figures follow a style suitable for ophthalmology / clinical ML journals
(Ophthalmology, JAMA Ophthalmol, Br J Ophthalmol).  The colour palette is
colour-blind-friendly (Paul Tol palette).

AUROC and AUPRC 95% CIs use non-parametric bootstrap resampling. DeLong's
method is used for paired comparisons between full and pre-culture AUROCs.
"""

import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy import stats
from scipy.interpolate import interp1d
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, brier_score_loss,
)
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Paths (may be overridden in main() via --strict)
DATA_DIR = Path("data")
RES_DIR  = Path("results")
FIG_DIR  = RES_DIR / "figures"
MDL_DIR  = RES_DIR / "models"

# Style
# Paul Tol colour-blind-safe palette
TOL_COLOURS = {
    "RF":  "#EE6677",   # rose
    "XGB": "#228833",   # green
}
MODEL_NAMES = {
    "RF":  "Random Forest",
    "XGB": "XGBoost",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi":     150,
})

# Fixed RNG for bootstrap CIs and SHAP sampling. SEPARATE from the data-split
# seed (61, set in 02_train.py); held constant so evaluation is reproducible.
ESTIMATOR_SEED = 42
N_BOOTSTRAP  = 1000   # for CI estimation


# -----------------------------------------------------------------------------
# 1. UTILITIES
# -----------------------------------------------------------------------------

def save_fig(name: str, fig):
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
    log.info("Saved figure: %s", FIG_DIR / f"{name}.png")
    plt.close(fig)


def bootstrap_auc(y_true, y_score, n_boot=N_BOOTSTRAP, rng=None):
    """Return (mean, lower_95, upper_95) via bootstrap."""
    if rng is None:
        rng = np.random.default_rng(ESTIMATOR_SEED)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_score[idx]))
    scores = np.array(scores)
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def bootstrap_auprc(y_true, y_score, n_boot=N_BOOTSTRAP, rng=None):
    if rng is None:
        rng = np.random.default_rng(ESTIMATOR_SEED)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(average_precision_score(y_true[idx], y_score[idx]))
    scores = np.array(scores)
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def compute_ici(y_true, y_score, frac=0.75):
    """Integrated Calibration Index (Austin & Steyerberg, Stat Med 2019).

    Fits a loess curve regressing observed outcomes on predicted probabilities,
    then returns the mean absolute difference between the smoothed curve and
    the 45-degree calibration line: ICI = mean(|loess(p_i) - p_i|).
    frac=0.75 matches the R default used in the original paper.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    smoothed = sm_lowess(y_true, y_score, frac=frac, it=0, return_sorted=True)
    fn = interp1d(smoothed[:, 0], smoothed[:, 1],
                  bounds_error=False,
                  fill_value=(smoothed[0, 1], smoothed[-1, 1]))
    return float(np.mean(np.abs(fn(y_score) - y_score)))


def bootstrap_ici(y_true, y_score, n_boot=N_BOOTSTRAP, rng=None, frac=0.75):
    """Bootstrap 95% CI for ICI."""
    if rng is None:
        rng = np.random.default_rng(ESTIMATOR_SEED)
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(compute_ici(y_true[idx], y_score[idx], frac=frac))
        except Exception:
            continue
    scores = np.array(scores)
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def load_resources():
    pred_df  = pd.read_csv(RES_DIR / "test_predictions.csv")
    metrics  = pd.read_csv(RES_DIR / "test_metrics.csv")
    cv_res   = pd.read_csv(RES_DIR / "cv_results.csv")
    feat_cols = (RES_DIR / "feature_columns.txt").read_text().split("\n")
    proc_df  = pd.read_csv(DATA_DIR / "processed_episodes.csv")

    models = {}
    for p in MDL_DIR.glob("*.joblib"):
        name = p.stem.split("_")[0]
        models[name] = joblib.load(p)

    y_true = pred_df["y_true"].values
    prob_cols = {n: f"prob_{n}" for n in models}
    probs = {n: pred_df[col].values for n, col in prob_cols.items() if col in pred_df}

    log.info("Loaded %d models: %s", len(models), list(models.keys()))
    log.info("Test set: %d cases  (poor outcome: %d, good outcome: %d)",
             len(y_true), y_true.sum(), (y_true==0).sum())

    return pred_df, metrics, cv_res, feat_cols, proc_df, models, y_true, probs


# -----------------------------------------------------------------------------
# 2. ROC CURVES
# -----------------------------------------------------------------------------

def plot_roc(y_true, probs, metrics):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    rng = np.random.default_rng(ESTIMATOR_SEED)

    for name in ["RF", "XGB"]:
        if name not in probs:
            continue
        y_score = probs[name]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # Label with the SAME point estimate and CI that Table 2 reports, so the
        # figure and the table cannot drift apart. Fall back to a bootstrap only
        # when test_metrics.csv predates the CI columns.
        row = metrics[metrics["model"] == name]
        if len(row) and {"auroc_lower_95", "auroc_upper_95"} <= set(metrics.columns):
            row = row.iloc[0]
            auc_pt = float(row["test_auroc"])
            auc_lo = float(row["auroc_lower_95"])
            auc_hi = float(row["auroc_upper_95"])
        else:
            auc_pt, auc_lo, auc_hi = bootstrap_auc(y_true, y_score, rng=rng)
        label = (f"{MODEL_NAMES[name]}  "
                 f"AUC={auc_pt:.3f} (95% CI {auc_lo:.3f}-{auc_hi:.3f})")
        ax.plot(fpr, tpr, color=TOL_COLOURS[name], lw=1.8, label=label)
        ax.fill_between(fpr, tpr, alpha=0.05, color=TOL_COLOURS[name])

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="No skill (AUC=0.500)")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("Receiver Operating Characteristic: Test Set")
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    sns.despine(ax=ax)
    save_fig("fig1_roc_curves", fig)


# -----------------------------------------------------------------------------
# 3. PRECISION-RECALL CURVES
# -----------------------------------------------------------------------------

def reported_auprc(y_true, y_score, metrics, name, rng=None):
    """Return the Table 2 AUPRC point estimate and its bootstrap interval."""
    row = metrics.loc[metrics["model"] == name]
    required = {"test_auprc", "auprc_lower_95", "auprc_upper_95"}
    if not row.empty and required <= set(metrics.columns):
        row = row.iloc[0]
        return (
            float(row["test_auprc"]),
            float(row["auprc_lower_95"]),
            float(row["auprc_upper_95"]),
        )

    ap_point = float(average_precision_score(y_true, y_score))
    _, ap_lo, ap_hi = bootstrap_auprc(y_true, y_score, rng=rng)
    return ap_point, ap_lo, ap_hi


def plot_pr(y_true, probs, metrics):
    prevalence = y_true.mean()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    rng = np.random.default_rng(ESTIMATOR_SEED)

    for name in ["RF", "XGB"]:
        if name not in probs:
            continue
        y_score = probs[name]
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap_point, ap_lo, ap_hi = reported_auprc(
            y_true, y_score, metrics, name, rng=rng
        )
        label = (f"{MODEL_NAMES[name]}  "
                 f"AP={ap_point:.3f} (95% CI {ap_lo:.3f}-{ap_hi:.3f})")
        ax.plot(rec, prec, color=TOL_COLOURS[name], lw=1.8, label=label)

    ax.axhline(prevalence, color="k", ls="--", lw=0.8,
               label=f"No skill (prevalence={prevalence:.2f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve: Test Set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=True, fontsize=8.5)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    sns.despine(ax=ax)
    save_fig("fig2_pr_curves", fig)


# -----------------------------------------------------------------------------
# 4. CALIBRATION
# -----------------------------------------------------------------------------

def plot_calibration(y_true, probs, metrics, lead_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_cal, ax_hist = axes

    for name in ["RF", "XGB"]:
        if name not in probs:
            continue
        y_score = probs[name]
        brier = brier_score_loss(y_true, y_score)
        ici   = compute_ici(y_true, y_score)
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10,
                                                strategy="quantile")
        ax_cal.plot(mean_pred, frac_pos,
                    "s-", color=TOL_COLOURS[name], lw=1.5, ms=5,
                    label=f"{MODEL_NAMES[name]} (Brier={brier:.3f}, ICI={ici:.3f})")

        # Overlay loess calibration curve
        smoothed = sm_lowess(y_true, y_score, frac=0.75, it=0, return_sorted=True)
        ax_cal.plot(smoothed[:, 0], smoothed[:, 1],
                    color=TOL_COLOURS[name], lw=1.0, ls=":", alpha=0.7)

    ax_cal.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Observed fraction of poor outcomes")
    ax_cal.set_title("Calibration (Reliability) Diagram: Test Set")
    ax_cal.legend(fontsize=8.5, frameon=True)
    ax_cal.set_xlim(0, 1);  ax_cal.set_ylim(0, 1)
    sns.despine(ax=ax_cal)

    # Prediction histogram for the lead model (default: highest AUROC)
    best_name = lead_name or metrics.sort_values("test_auroc", ascending=False).iloc[0]["model"]
    if best_name in probs:
        y_score = probs[best_name]
        ax_hist.hist(y_score[y_true == 0], bins=20, alpha=0.6,
                     color="#4477AA", label="Good outcome", density=True)
        ax_hist.hist(y_score[y_true == 1], bins=20, alpha=0.6,
                     color="#EE6677", label="Poor outcome", density=True)
        ax_hist.set_xlabel(f"Predicted probability ({MODEL_NAMES[best_name]})")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Score Distribution: Test Set")
        ax_hist.legend(fontsize=8.5)
        sns.despine(ax=ax_hist)

    fig.tight_layout()
    save_fig("fig3_calibration", fig)


# -----------------------------------------------------------------------------
# 5. CONFUSION MATRICES
# -----------------------------------------------------------------------------

def plot_confusion_matrices(y_true, probs, metrics):
    model_list = [n for n in ["RF", "XGB"] if n in probs]
    fig, axes = plt.subplots(1, len(model_list),
                             figsize=(3.5 * len(model_list), 3.5))
    if len(model_list) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_list):
        row = metrics[metrics["model"] == name].iloc[0]
        thr = row["opt_threshold"]
        y_pred = (probs[name] >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Good", "Poor"],
                    yticklabels=["Good", "Poor"],
                    cbar=False, square=True)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{MODEL_NAMES[name]}\n(thr={thr:.2f})")

    fig.suptitle("Confusion Matrices at Development-Set Youden Threshold", y=1.02)
    fig.tight_layout()
    save_fig("fig4_confusion_matrices", fig)


# -----------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (Random Forest)
# -----------------------------------------------------------------------------

FEATURE_DISPLAY_NAMES = {
    "age_years":          "Age (years)",
    "pres_logmar":        "Presenting VA (logMAR)",
    "gender_female":      "Female sex",
    "fundus_visible":     "Fundus visible",
    "rapd_present":       "RAPD present",
    "diabetes":           "Diabetes mellitus",
    "immune_suppressed":  "Immune suppression",
    "prior_surgery":      "Prior ophthalmic surgery",
    "etiol_cataract_sx":   "Post-cataract/IOL surgery",
    "etiol_glaucoma_sx":   "Post-glaucoma/drainage surgery",
    "etiol_trauma":        "Penetrating eye injury",
    "etiol_corneal_ulcer": "Corneal ulcer",
    "etiol_endogenous":    "Metastatic/endogenous",
    "etiol_ivi":           "Post-intravitreal injection",
    "etiol_other":         "Other ocular procedure",
    "culture_positive":   "Culture positive",
    # organism OHE columns kept as-is
}


def get_feature_names_from_pipe(pipe):
    """Extract feature names from the fitted ColumnTransformer inside a Pipeline."""
    ct = pipe.named_steps["pre"]
    names = []
    for tf_name, transformer, cols in ct.transformers_:
        if transformer == "drop":
            continue
        if tf_name == "num":
            # add_indicator=False: output has exactly len(cols) columns, in order
            names.extend(cols)
        elif tf_name == "cat":
            ohe = transformer.named_steps["ohe"]
            try:
                for i, col in enumerate(cols):
                    cats = ohe.categories_[i]
                    names.extend([f"{col}_{c}" for c in cats])
            except Exception:
                names.extend(cols)
    return names


def plot_rf_importance(models, pred_df, y_true, feat_cols):
    if "RF" not in models:
        return
    pipe = models["RF"]
    X_test_raw = pred_df[feat_cols] if all(c in pred_df for c in feat_cols) else None
    if X_test_raw is None:
        log.warning("Cannot find raw features in pred_df for RF importance.")
        return

    # RF built-in MDI importance (on training-transformed features)
    rf_clf = pipe.named_steps["clf"]
    feat_names = get_feature_names_from_pipe(pipe)
    importances = rf_clf.feature_importances_
    n = min(len(importances), len(feat_names))
    importances = importances[:n]
    feat_names  = feat_names[:n]

    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df["display"] = imp_df["feature"].map(FEATURE_DISPLAY_NAMES).fillna(
        imp_df["feature"])
    imp_df = imp_df.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    colours = ["#4477AA" if not f.endswith("_missing") else "#AAAAAA"
               for f in imp_df["feature"]]
    ax.barh(imp_df["display"][::-1], imp_df["importance"][::-1],
            color=colours[::-1])
    ax.set_xlabel("Mean Decrease in Impurity (MDI)")
    ax.set_title(f"Random Forest: Feature Importance (Top {len(imp_df)})")
    ax.axvline(0, color="k", lw=0.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_fig("fig5_feature_importance_rf", fig)


def plot_xgb_importance(models):
    if "XGB" not in models:
        return
    pipe = models["XGB"]
    xgb_clf = pipe.named_steps["clf"]
    feat_names = get_feature_names_from_pipe(pipe)

    score_dict = xgb_clf.get_booster().get_score(importance_type="gain")
    # Map internal f-names (f0, f1, etc.) to real names
    imp_df = pd.DataFrame([
        {"feature_idx": int(k[1:]), "importance": v}
        for k, v in score_dict.items()
    ]).sort_values("feature_idx")
    imp_df["feature"] = imp_df["feature_idx"].apply(
        lambda i: feat_names[i] if i < len(feat_names) else f"f{i}")
    imp_df["display"] = imp_df["feature"].map(FEATURE_DISPLAY_NAMES).fillna(
        imp_df["feature"])
    imp_df = imp_df.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.barh(imp_df["display"][::-1], imp_df["importance"][::-1],
            color="#228833")
    ax.set_xlabel("Mean Gain")
    ax.set_title(f"XGBoost: Feature Importance by Gain (Top {len(imp_df)})")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_fig("fig6_feature_importance_xgb", fig)


# -----------------------------------------------------------------------------
# 7. SHAP ANALYSIS (lead model)
# -----------------------------------------------------------------------------

def plot_shap(models, pred_df, y_true, feat_cols, metrics, lead_name=None):
    best_name = lead_name or metrics.sort_values("test_auroc", ascending=False).iloc[0]["model"]
    log.info("SHAP analysis on lead model: %s", best_name)

    if best_name not in models:
        log.warning("Best model %s not found.", best_name)
        return

    pipe = models[best_name]
    feat_names = get_feature_names_from_pipe(pipe)

    # Transform test data through preprocessor
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X_raw = pred_df[[c for c in feat_cols if c in pred_df.columns]]
    try:
        X_trans = pre.transform(X_raw)
    except Exception as e:
        log.warning("Could not transform test data for SHAP: %s", e)
        return

    X_trans_df = pd.DataFrame(X_trans, columns=feat_names[:X_trans.shape[1]])

    # Choose explainer by model type
    base_values_override = None  # set by fallback path for waterfall
    try:
        if best_name in ("RF", "XGB"):
            try:
                explainer = shap.TreeExplainer(clf)
                shap_values_raw = explainer.shap_values(X_trans_df)
            except Exception:
                # Fallback for SHAP/XGBoost 3.x TreeExplainer incompatibility
                bg = shap.sample(X_trans_df, 50, random_state=ESTIMATOR_SEED)
                exp_obj = shap.PermutationExplainer(clf.predict_proba, bg)
                exp_result = exp_obj(X_trans_df)
                shap_values_raw = exp_result.values   # shape (n, f, 2)
                base_values_override = exp_result.base_values  # shape (n, 2)
                explainer = exp_obj
            # Handle different shap output shapes:
            #   list of 2 arrays: take index 1 (class=poor)
            #   3-D array (n,f,2): take [:,:,1]
            #   2-D array (n,f): use directly (binary output)
            if isinstance(shap_values_raw, list):
                shap_values = shap_values_raw[1]
            elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
                shap_values = shap_values_raw[:, :, 1]
            else:
                shap_values = shap_values_raw
        else:
            explainer = shap.LinearExplainer(clf, X_trans_df)
            shap_values = explainer.shap_values(X_trans_df)
    except Exception as e:
        log.warning("SHAP computation failed: %s", e)
        return

    # Beeswarm / summary
    fig, ax = plt.subplots(figsize=(7, 5.5))
    display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in X_trans_df.columns]
    shap.summary_plot(
        shap_values, X_trans_df,
        feature_names=display_names,
        show=False, plot_size=None, max_display=20,
    )
    ax = plt.gca()
    ax.set_title(f"SHAP Feature Contributions: {MODEL_NAMES.get(best_name, best_name)}")
    plt.tight_layout()
    save_fig("fig7_shap_summary", plt.gcf())

    # Concise beeswarm: top 10 features only
    fig, ax = plt.subplots(figsize=(7, 5.5))
    shap.summary_plot(
        shap_values, X_trans_df,
        feature_names=display_names,
        show=False, plot_size=None, max_display=10,
    )
    ax = plt.gca()
    ax.set_title(f"SHAP Feature Contributions: {MODEL_NAMES.get(best_name, best_name)}")
    plt.tight_layout()
    save_fig("fig7_shap_summary_concise", plt.gcf())

    # Waterfall plots for two illustrative cases
    indices = {"Typical poor": np.where(y_true == 1)[0][0],
               "Typical good": np.where(y_true == 0)[0][0]}
    try:
        if base_values_override is not None:
            bv = base_values_override
            # 2-D (n, 2): take class-1 column; 1-D: use as is
            if isinstance(bv, np.ndarray) and bv.ndim == 2:
                bv = bv[:, 1]
            base_vals_arr = np.asarray(bv, dtype=float)
        else:
            ev = explainer.expected_value
            # TreeExplainer for multiclass/binary returns array or list; take class-1 value
            if hasattr(ev, '__len__'):
                ev = float(ev[1]) if len(ev) > 1 else float(ev[0])
            else:
                ev = float(ev)
            base_vals_arr = np.full(len(shap_values), ev)
        exp = shap.Explanation(
            values=shap_values,
            base_values=base_vals_arr,
            data=X_trans_df.values,
            feature_names=display_names,
        )
        fig, axes = plt.subplots(1, len(indices), figsize=(6 * len(indices), 5))
        if len(indices) == 1:
            axes = [axes]
        for ax, (label, idx) in zip(axes, indices.items()):
            plt.sca(ax)
            shap.waterfall_plot(exp[idx], max_display=12, show=False)
            ax.set_title(f"{label} (y={y_true[idx]})", fontsize=10)
        fig.suptitle("SHAP Waterfall Plots: Example Cases", y=1.01)
        fig.tight_layout()
        save_fig("fig8_shap_waterfall_examples", fig)
    except Exception as e:
        log.warning("Waterfall plot failed: %s", e)


# -----------------------------------------------------------------------------
# 8. TABLE 1: COHORT CHARACTERISTICS
# -----------------------------------------------------------------------------

def make_table1(proc_df: pd.DataFrame) -> pd.DataFrame:
    """Produce submitted Table 1, including the specified univariable tests.

    Continuous variables use two-sided Mann-Whitney U tests. Categorical
    variables use Pearson chi-squared tests with Yates continuity correction.
    Tests use available observations for the relevant variable, matching the
    denominators displayed in each row.
    """
    df      = proc_df.dropna(subset=["poor_outcome"]).copy()
    overall = df
    good    = df[df["poor_outcome"] == 0]
    poor    = df[df["poor_outcome"] == 1]
    groups  = [("Overall", overall), ("Good outcome", good), ("Poor outcome", poor)]

    # Denominators vary by row because documentation is incomplete for several
    # predictors, so every cell carries the number observed. Without it the
    # percentages cannot be reconstructed and read as though they were taken
    # over the full 1300 episodes.
    def fmt_cont(col, grp):
        v = grp[col].dropna()
        return (f"{v.median():.1f} ({v.quantile(0.25):.1f}-{v.quantile(0.75):.1f})"
                f" [n={len(v)}]")

    def fmt_cat(col, val, grp):
        n   = (grp[col] == val).sum()
        tot = grp[col].notna().sum()
        return f"{n}/{tot} ({100*n/tot:.1f})" if tot > 0 else "NA"

    def fmt_p(p):
        if p is None or not np.isfinite(p):
            return ""
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    def p_cont(col):
        good_values = good[col].dropna()
        poor_values = poor[col].dropna()
        if good_values.empty or poor_values.empty:
            return np.nan
        return float(stats.mannwhitneyu(
            good_values, poor_values, alternative="two-sided").pvalue)

    def p_cat(col, val):
        good_values = good[col].dropna()
        poor_values = poor[col].dropna()
        if good_values.empty or poor_values.empty:
            return np.nan
        table = np.array([
            [(good_values == val).sum(), (good_values != val).sum()],
            [(poor_values == val).sum(), (poor_values != val).sum()],
        ], dtype=int)
        if np.any(table.sum(axis=0) == 0):
            return np.nan
        return float(stats.chi2_contingency(table, correction=True).pvalue)

    rows = []

    def add(char, fn, p_fn=None):
        row = {"Characteristic": char}
        for gname, grp in groups:
            row[gname] = fn(grp)
        row["Univariate statistical comparison"] = (
            fmt_p(p_fn()) if p_fn is not None else ""
        )
        rows.append(row)

    add("N",                                    lambda g: str(len(g)))
    add("Follow-up duration, days, median (IQR)", lambda g: fmt_cont("days_to_final_visit", g), lambda: p_cont("days_to_final_visit"))
    add("Age, years, median (IQR)",            lambda g: fmt_cont("age_years", g), lambda: p_cont("age_years"))
    add("Female sex, n (%)",                   lambda g: fmt_cat("gender_female", 1.0, g), lambda: p_cat("gender_female", 1.0))
    add("Presenting VA, logMAR, median (IQR)", lambda g: fmt_cont("pres_logmar", g), lambda: p_cont("pres_logmar"))
    add("Presenting VA worse than 6/60, n (%)", lambda g: fmt_cat("pres_poor_va", 1.0, g), lambda: p_cat("pres_poor_va", 1.0))
    add("Fundus not visible, n (%)",           lambda g: fmt_cat("fundus_visible", 0.0, g), lambda: p_cat("fundus_visible", 0.0))
    add("RAPD present, n (%)",                 lambda g: fmt_cat("rapd_present", 1.0, g), lambda: p_cat("rapd_present", 1.0))
    add("Diabetes mellitus, n (%)",            lambda g: fmt_cat("diabetes", 1.0, g), lambda: p_cat("diabetes", 1.0))
    add("Immune suppression, n (%)",           lambda g: fmt_cat("immune_suppressed", 1.0, g), lambda: p_cat("immune_suppressed", 1.0))
    add("Prior ophthalmic surgery, n (%)",     lambda g: fmt_cat("prior_surgery", 1.0, g), lambda: p_cat("prior_surgery", 1.0))
    # All seven a priori aetiology flags, matching the specified predictor set.
    add("  Post-cataract/IOL surgery, n (%)",  lambda g: fmt_cat("etiol_cataract_sx", 1.0, g), lambda: p_cat("etiol_cataract_sx", 1.0))
    add("  Post-IVI, n (%)",                   lambda g: fmt_cat("etiol_ivi", 1.0, g), lambda: p_cat("etiol_ivi", 1.0))
    add("  Post-glaucoma/drainage surgery, n (%)", lambda g: fmt_cat("etiol_glaucoma_sx", 1.0, g), lambda: p_cat("etiol_glaucoma_sx", 1.0))
    add("  Penetrating eye injury, n (%)",     lambda g: fmt_cat("etiol_trauma", 1.0, g), lambda: p_cat("etiol_trauma", 1.0))
    add("  Corneal ulcer, n (%)",              lambda g: fmt_cat("etiol_corneal_ulcer", 1.0, g), lambda: p_cat("etiol_corneal_ulcer", 1.0))
    add("  Metastatic/endogenous, n (%)",      lambda g: fmt_cat("etiol_endogenous", 1.0, g), lambda: p_cat("etiol_endogenous", 1.0))
    add("  Other ocular procedure, n (%)",     lambda g: fmt_cat("etiol_other", 1.0, g), lambda: p_cat("etiol_other", 1.0))
    add("Culture positive, n (%)",             lambda g: fmt_cat("culture_positive", 1.0, g), lambda: p_cat("culture_positive", 1.0))

    return pd.DataFrame(rows, columns=["Characteristic", "Overall",
                                        "Good outcome", "Poor outcome",
                                        "Univariate statistical comparison"])


# -----------------------------------------------------------------------------
# 9. TABLE 2: MODEL PERFORMANCE
# -----------------------------------------------------------------------------

def make_table2(metrics: pd.DataFrame, y_true, probs) -> pd.DataFrame:
    """Academic-style performance table with 95% CI.

    CIs are read from pre-computed columns in test_metrics.csv when available
    (populated by 02_train.py); otherwise recomputed via bootstrap.
    """
    rng  = np.random.default_rng(ESTIMATOR_SEED)
    rows = []
    has_ci = ("auroc_lower_95" in metrics.columns and
              "auroc_upper_95" in metrics.columns)
    has_auprc_ci = ("auprc_lower_95" in metrics.columns and
                    "auprc_upper_95" in metrics.columns)

    for name in ["RF", "XGB"]:
        if name not in probs:
            continue
        y_score = probs[name]
        brier   = brier_score_loss(y_true, y_score)
        row     = metrics[metrics["model"] == name].iloc[0]

        auc_m = float(row["test_auroc"])
        if has_ci:
            auc_lo = float(row["auroc_lower_95"])
            auc_hi = float(row["auroc_upper_95"])
        else:
            _, auc_lo, auc_hi = bootstrap_auc(y_true, y_score, rng=rng)

        ap_m = float(row["test_auprc"])
        if has_auprc_ci:
            ap_lo = float(row["auprc_lower_95"])
            ap_hi = float(row["auprc_upper_95"])
        else:
            _, ap_lo, ap_hi = bootstrap_auprc(y_true, y_score, rng=rng)

        # Report the ICI POINT estimate (the same value plot_calibration prints on
        # Figure SF3 and 06_imbalance_sensitivity.py reports). The bootstrap mean
        # is upward biased because ICI is a mean absolute deviation, so only the
        # percentile interval is taken from the bootstrap.
        ici_m = compute_ici(y_true, y_score)
        _, ici_lo, ici_hi = bootstrap_ici(y_true, y_score, rng=rng)

        tp = float(row["tp"]); tn = float(row["tn"])
        fp = float(row["fp"]); fn = float(row["fn"])
        acc = (tp + tn) / (tp + tn + fp + fn)

        rows.append({
            "Model":       MODEL_NAMES[name],
            "AUROC":       f"{auc_m:.3f} ({auc_lo:.3f}-{auc_hi:.3f})",
            "AUPRC":       f"{ap_m:.3f} ({ap_lo:.3f}-{ap_hi:.3f})",
            "Brier score": f"{brier:.3f}",
            "ICI":         f"{ici_m:.3f} ({ici_lo:.3f}-{ici_hi:.3f})",
            "Sensitivity": f"{row['sensitivity']:.3f}",
            "Specificity": f"{row['specificity']:.3f}",
            "PPV":         f"{row['ppv']:.3f}",
            "NPV":         f"{row['npv']:.3f}",
            "Accuracy":    f"{acc:.3f}",
            "Threshold":   f"{row['opt_threshold']:.2f}",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 10. OPERATING CHARACTERISTICS: MANUSCRIPT TABLE 4 (lead model)
# -----------------------------------------------------------------------------


def make_table4(y_true, probs, metrics, lead_name=None) -> pd.DataFrame:
    """Operating-point table: multiple thresholds for the lead model."""
    best_name = lead_name or metrics.sort_values("test_auroc", ascending=False).iloc[0]["model"]
    if best_name not in probs:
        return pd.DataFrame()
    y_score = probs[best_name]

    rows = []
    for thr in np.arange(0.15, 0.91, 0.05):
        y_pred = (y_score >= thr).astype(int)
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_true, y_pred,
                                                    labels=[0, 1]).ravel()
        sens = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else np.nan
        spec = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else np.nan
        ppv  = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else np.nan
        npv  = tn_f / (tn_f + fn_f) if (tn_f + fn_f) > 0 else np.nan
        rows.append({
            "Threshold":      round(thr, 2),
            "Sensitivity":    round(sens, 3),
            "Specificity":    round(spec, 3),
            "PPV":            round(ppv,  3),
            "NPV":            round(npv,  3),
            "TP": int(tp_f), "FP": int(fp_f),
            "TN": int(tn_f), "FN": int(fn_f),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 11. MICRO vs NOMICRO AUROC COMPARISON: MANUSCRIPT TABLE 3
# -----------------------------------------------------------------------------

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
    """
    DeLong's method for comparing two AUROCs on the same test set.

    Accounts for the covariance between the two ROC curves (i.e. the
    comparison is inherently paired because both models see the same cases).
    Returns (auc_a, auc_b, diff, z, p_two_sided).

    Reference: DeLong et al. (1988). Biometrics 44(3):837-845.
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
    z        = (auc_a - auc_b) / np.sqrt(var_diff)
    p        = float(2 * stats.norm.sf(abs(z)))
    return auc_a, auc_b, auc_a - auc_b, z, p


def compare_micro_nomicro(pred_df, y_true, probs, nomicro_dir):
    """
    Compares AUROC of the full-feature (micro) vs pre-culture (nomicro) models
    using DeLong's test.  A paired bootstrap (same resampled indices applied to
    both models) provides the 95% CI on the AUROC difference.

    Saved to: <FIG_DIR>/table3_micro_nomicro_comparison.csv
    """
    nm_pred_path = nomicro_dir / "test_predictions.csv"
    if not nm_pred_path.exists():
        log.warning("Nomicro predictions not found at %s; skipping.", nm_pred_path)
        return None

    nm_pred_df = pd.read_csv(nm_pred_path)
    nm_y_true  = nm_pred_df["y_true"].values

    id_cols = ["rveeh_ur", "admission_date"]
    if all(c in pred_df.columns and c in nm_pred_df.columns for c in id_cols):
        ids_full = pred_df[id_cols].astype(str).reset_index(drop=True)
        ids_pre = nm_pred_df[id_cols].astype(str).reset_index(drop=True)
        if not ids_full.equals(ids_pre):
            log.warning("Episode identity mismatch between full and pre-culture "
                        "predictions; comparison skipped.")
            return None

    if not np.array_equal(y_true, nm_y_true):
        log.warning("y_true mismatch between micro and nomicro; comparison skipped.")
        return None

    rng  = np.random.default_rng(ESTIMATOR_SEED)
    n    = len(y_true)
    rows = []

    for name in ["RF", "XGB"]:
        col = f"prob_{name}"
        if col not in pred_df.columns or col not in nm_pred_df.columns:
            continue

        p_micro = pred_df[col].values
        p_nm    = nm_pred_df[col].values

        auc_m, auc_nm, diff, z, p_val = delong_compare(y_true, p_micro, p_nm)

        # Paired bootstrap 95% CI for the difference
        diffs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            diffs.append(
                roc_auc_score(y_true[idx], p_micro[idx]) -
                roc_auc_score(y_true[idx], p_nm[idx])
            )
        diffs   = np.array(diffs)
        diff_lo = float(np.percentile(diffs, 2.5))
        diff_hi = float(np.percentile(diffs, 97.5))

        rows.append({
            "model":         name,
            "auroc_micro":   round(auc_m,   4),
            "auroc_nomicro": round(auc_nm,  4),
            "auroc_diff":    round(diff,    4),
            "diff_lower_95": round(diff_lo, 4),
            "diff_upper_95": round(diff_hi, 4),
            "z_statistic":   round(z,       3),
            "p_value":       round(p_val,   4),
        })
        log.info(
            "%s  micro=%.4f  nomicro=%.4f  diff=%.4f (95%%CI %.4f-%.4f)  "
            "z=%.3f  p=%.4f",
            name, auc_m, auc_nm, diff, diff_lo, diff_hi, z, p_val,
        )

    return pd.DataFrame(rows) if rows else None


# -----------------------------------------------------------------------------
# 11b. DECISION-CURVE ANALYSIS (net benefit): lead model
# -----------------------------------------------------------------------------

def decision_curve(y_true, probs, lead_name, pt_grid=None):
    """Net-benefit decision-curve analysis for the lead model versus the
    treat-all and treat-none strategies (Vickers & Elkin, Med Decis Making 2006).

    Net benefit at threshold probability pt:
        NB = TP/n - FP/n * (pt / (1 - pt))
    Treat-all NB uses everyone classified positive; treat-none NB = 0.
    Saves fig9_decision_curve and returns a per-threshold table.
    """
    if lead_name not in probs:
        return None
    y = np.asarray(y_true); n = len(y); prev = y.mean()
    s = probs[lead_name]
    if pt_grid is None:
        pt_grid = np.round(np.arange(0.05, 0.61, 0.05), 2)

    rows = []
    for pt in pt_grid:
        yp = (s >= pt).astype(int)
        tp = int(((yp == 1) & (y == 1)).sum())
        fp = int(((yp == 1) & (y == 0)).sum())
        nb_model = tp / n - fp / n * (pt / (1 - pt))
        nb_all   = prev - (1 - prev) * (pt / (1 - pt))
        rows.append({"threshold_prob": pt,
                     "nb_model": round(nb_model, 4),
                     "nb_treat_all": round(nb_all, 4),
                     "nb_treat_none": 0.0})
    dca = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(dca["threshold_prob"], dca["nb_model"], "-o", color=TOL_COLOURS.get(lead_name, "#EE6677"),
            lw=1.8, ms=4, label=f"{MODEL_NAMES.get(lead_name, lead_name)} (lead model)")
    ax.plot(dca["threshold_prob"], dca["nb_treat_all"], "--", color="#777777", lw=1.2, label="Treat all")
    ax.axhline(0, color="k", lw=0.8, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-Curve Analysis: Test Set")
    ax.set_ylim(min(-0.05, dca["nb_treat_all"].min()), dca["nb_model"].max() * 1.15 + 0.01)
    ax.legend(frameon=True, fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_fig("fig9_decision_curve", fig)
    return dca


# -----------------------------------------------------------------------------
# 11c. SUBGROUP FAIRNESS (discrimination by sex and age band): lead model
# -----------------------------------------------------------------------------

def subgroup_fairness(pred_df, y_true, probs, lead_name):
    """Assess discrimination (AUROC) of the lead model within demographic
    subgroups (sex; age bands). Reported for algorithmic-fairness transparency
    (TRIPOD+AI item 14). Procedure-based subgroups are deliberately NOT reported
    (treatment-agnosticism is established by split balance, not stratified
    performance). Test-set subgroups are small; results are descriptive.
    """
    if lead_name not in probs:
        return None
    y = np.asarray(y_true); s = probs[lead_name]
    rows = []

    def add(label, mask):
        m = np.asarray(mask, dtype=bool)
        n = int(m.sum()); npos = int(y[m].sum()) if n else 0
        if n >= 20 and 0 < npos < n:
            auc = roc_auc_score(y[m], s[m])
            auc = f"{auc:.3f}"
        else:
            auc = "n/a (n<20 or single class)"
        rows.append({"subgroup": label, "n": n, "n_poor": npos, "auroc": auc})

    add("Overall", np.ones(len(y), dtype=bool))
    if "gender_female" in pred_df:
        g = pred_df["gender_female"].values
        add("Female", g == 1)
        add("Male",   g == 0)
    if "age_years" in pred_df:
        a = pred_df["age_years"].values
        add("Age < 65",     a < 65)
        add("Age 65-79",    (a >= 65) & (a < 80))
        add("Age >= 80",     a >= 80)
    fair = pd.DataFrame(rows)
    return fair


# -----------------------------------------------------------------------------
# 12. MAIN
# -----------------------------------------------------------------------------

def main():
    global DATA_DIR, RES_DIR, FIG_DIR, MDL_DIR
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise endophthalmitis outcome models.")
    parser.add_argument(
        "--strict", action="store_true",
        help="Read from results_strict/ and data_strict/ (strict outcome threshold).")
    parser.add_argument(
        "--nomicro", action="store_true",
        help="Read from results[_strict]_nomicro/ (no-microbiology feature set).")
    parser.add_argument(
        "--lead_model", choices=["RF", "XGB"], default="XGB",
        help="Force the lead/clinical model used for SHAP, the clinical-threshold "
             "table, the score histogram, decision-curve and fairness analyses. "
             "Defaults to XGB, the lead model reported in the manuscript.")
    args = parser.parse_args()

    base = "results_strict" if args.strict else "results"
    if args.nomicro:
        base += "_nomicro"

    DATA_DIR = Path("data_strict") if args.strict else Path("data")
    RES_DIR  = Path(base)
    FIG_DIR  = RES_DIR / "figures"
    MDL_DIR  = RES_DIR / "models"

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    (pred_df, metrics, cv_res, feat_cols,
     proc_df, models, y_true, probs) = load_resources()

    # Lead model: XGBoost by default; fall back only if it is unavailable.
    auroc_best = metrics.sort_values("test_auroc", ascending=False).iloc[0]["model"]
    lead_name  = args.lead_model if (args.lead_model and args.lead_model in probs) else auroc_best
    log.info("Lead model for SHAP / Table 4 / histogram / DCA / fairness: %s "
             "(highest-AUROC model: %s)", lead_name, auroc_best)

    log.info("\nGenerating figures")
    plot_roc(y_true, probs, metrics)
    plot_pr(y_true, probs, metrics)
    plot_calibration(y_true, probs, metrics, lead_name=lead_name)
    plot_confusion_matrices(y_true, probs, metrics)
    plot_rf_importance(models, pred_df, y_true, feat_cols)
    plot_xgb_importance(models)
    plot_shap(models, pred_df, y_true, feat_cols, metrics, lead_name=lead_name)

    log.info("\nGenerating tables")
    tbl1 = make_table1(proc_df)
    tbl1.to_csv(FIG_DIR / "table1_patient_characteristics.csv", index=False)
    log.info("Table 1 saved.")

    tbl2 = make_table2(metrics, y_true, probs)
    tbl2.to_csv(FIG_DIR / "table2_model_performance.csv", index=False)
    log.info("Table 2 saved.\n%s", tbl2.to_string(index=False))

    tbl4 = make_table4(y_true, probs, metrics, lead_name=lead_name)
    tbl4.to_csv(FIG_DIR / "table4_clinical_thresholds.csv", index=False)
    log.info("Table 4 saved (operating points for %s).", MODEL_NAMES.get(lead_name, lead_name))

    # Decision-curve analysis (net benefit) for the lead model
    dca = decision_curve(y_true, probs, lead_name)
    if dca is not None:
        dca.to_csv(FIG_DIR / "table_decision_curve.csv", index=False)
        log.info("Decision-curve analysis saved for %s.", MODEL_NAMES.get(lead_name, lead_name))

    # Subgroup fairness (sex, age) for the lead model
    fair = subgroup_fairness(pred_df, y_true, probs, lead_name)
    if fair is not None:
        fair.to_csv(FIG_DIR / "table_fairness_subgroups.csv", index=False)
        log.info("Subgroup fairness table saved.\n%s", fair.to_string(index=False))

    # Micro vs nomicro comparison, only for the full-feature run
    if not args.nomicro:
        nomicro_dir = Path(str(RES_DIR) + "_nomicro")
        log.info("\nMicro vs nomicro AUROC comparison (DeLong)")
        tbl3 = compare_micro_nomicro(pred_df, y_true, probs, nomicro_dir)
        if tbl3 is not None:
            tbl3.to_csv(FIG_DIR / "table3_micro_nomicro_comparison.csv", index=False)
            log.info("Table 3 saved.\n%s", tbl3.to_string(index=False))

    # Final summary printout
    log.info("\n" + "=" * 65)
    log.info("PERFORMANCE SUMMARY")
    log.info("=" * 65)
    log.info("\nCross-validation (development set, 5-fold):")
    log.info(cv_res[["model","best_cv_auroc_mean","best_cv_auroc_std"]].to_string(index=False))
    log.info("\nHeld-out test set:")
    metrics_disp = metrics.copy()
    metrics_disp["accuracy"] = (
        (metrics_disp["tp"] + metrics_disp["tn"]) /
        (metrics_disp["tp"] + metrics_disp["tn"] +
         metrics_disp["fp"] + metrics_disp["fn"])
    )
    log.info(metrics_disp[["model","test_auroc","test_auprc","brier_score",
                            "sensitivity","specificity","accuracy"]].to_string(index=False))
    log.info("\nHighest-AUROC model: %s  (AUROC=%.4f)   Lead/clinical model: %s",
             auroc_best,
             metrics.sort_values("test_auroc", ascending=False).iloc[0]["test_auroc"],
             lead_name)
    log.info("\nEvaluation complete. Figures saved to: %s", FIG_DIR)


if __name__ == "__main__":
    main()
