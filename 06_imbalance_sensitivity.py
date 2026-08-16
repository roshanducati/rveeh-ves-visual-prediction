"""
06_imbalance_sensitivity.py
===========================
Sensitivity analysis of whether class-imbalance correction materially changes
discrimination or calibration in this cohort.

Motivation
----------
The analysed cohort is only mildly imbalanced (599 poor / 701 good = 46.1% vs
53.9%; train-set positive:negative ratio 1:1.19). The pipeline nonetheless
applies imbalance correction (XGBoost scale_pos_weight = n_neg/n_pos; Random
Forest class_weight="balanced"). This script assesses the effect of that
correction by refitting the lead model, with Random Forest as a secondary check,
both with and without correction while holding all other components fixed:
  * identical patient-level train/test split (seed 61),
  * identical tuned hyperparameters (read from the fitted model pipelines),
  * identical preprocessing (single-dataset iterative imputation, scaling, and
    one-hot encoding).
The only thing toggled is the imbalance weight, so any difference in the
held-out metrics is attributable to the correction alone.

Metrics on the held-out test set
--------------------------------
  Discrimination : AUROC, AUPRC
  Calibration    : Brier score, Integrated Calibration Index (ICI),
                   calibration-in-the-large (intercept), calibration slope,
                   Expected Calibration Index (ECE, 10 bins), mean predicted p.

Outputs
-------
  results_strict_sensitivity/imbalance_sensitivity_metrics.csv
  results_strict_sensitivity/imbalance_sensitivity_deltas.csv
  results_strict_sensitivity/imbalance_sensitivity_calibration.png/.pdf
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Import the study's training and evaluation helpers
def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

train = _load("train_mod", "02_train.py")
evalm = _load("eval_mod", "03_evaluate.py")

# Point the training module at the reported strict dataset.
train.IN_FILE = Path("data_strict/processed_episodes.csv")

OUT_DIR = Path("results_strict_sensitivity")
SPLIT_SEED = train.SPLIT_SEED
ESTIMATOR_SEED = train.ESTIMATOR_SEED
RNG = np.random.default_rng(ESTIMATOR_SEED)
N_BOOT = 2000


# Calibration metrics
def calibration_intercept_slope(y, p):
    """Cox calibration. Slope: coef of logit(p) in logistic reg of y on logit(p).
    Intercept (calibration-in-the-large): intercept of logistic reg of y on an
    offset of logit(p) with slope fixed at 1."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y, int)

    slope = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    slope.fit(lp, y)
    cal_slope = float(slope.coef_[0, 0])

    # intercept with slope fixed to 1 via offset: fit intercept only on (y, offset=lp)
    # emulate offset by using sample_weight trick is messy; use statsmodels-free
    # Newton on intercept.  Solve for a: mean(sigmoid(a+lp)) matches by MLE.
    a = 0.0
    for _ in range(100):
        eta = a + lp.ravel()
        mu = 1 / (1 + np.exp(-eta))
        grad = np.sum(y - mu)
        hess = -np.sum(mu * (1 - mu))
        step = grad / hess
        a -= step
        if abs(step) < 1e-10:
            break
    return float(a), cal_slope


def ece(y, p, n_bins=10):
    """Expected Calibration Error, equal-width bins."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(total)


def all_metrics(y, p):
    ci, sl = calibration_intercept_slope(y, p)
    return dict(
        auroc=roc_auc_score(y, p),
        auprc=average_precision_score(y, p),
        brier=brier_score_loss(y, p),
        ici=evalm.compute_ici(y, p),
        ece=ece(y, p),
        cal_intercept=ci,
        cal_slope=sl,
        mean_pred=float(np.mean(p)),
        obs_rate=float(np.mean(y)),
    )


def paired_boot_delta(y, p_corr, p_uncorr, fn):
    """Bootstrap 95% CI of fn(uncorrected) - fn(corrected), paired on cases."""
    y = np.asarray(y); n = len(y)
    d = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(fn(y[idx], np.asarray(p_uncorr)[idx]) -
                 fn(y[idx], np.asarray(p_corr)[idx]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


# Model refit helpers
def refit_eval(estimator, num, cat, X_tr, y_tr, X_te):
    pipe = train.Pipeline([
        ("pre", train.make_preprocessor(num, cat)),
        ("clf", estimator),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Fixed random values: split={SPLIT_SEED}, estimation={ESTIMATOR_SEED}")
    num = train.NUMERIC_FEATURES
    cat = train.CATEGORICAL_FEATURES
    Xtr, Xte, ytr, yte, dfk, gtr, tri, tei = train.load_and_split(num, cat)
    ytr = ytr.astype(int); yte = yte.astype(int)

    spw = float((ytr == 0).sum()) / float((ytr == 1).sum())
    print(f"Train poor:{int((ytr==1).sum())} good:{int((ytr==0).sum())} "
          f"scale_pos_weight={spw:.3f}")
    print(f"Test  poor:{int((yte==1).sum())} good:{int((yte==0).sum())} "
          f"prevalence={yte.mean():.3f}\n")

    # Tuned hyperparameters from the fitted models, kept identical across arms
    xgb_hp = {k: joblib.load("results_strict/models/XGB_pipeline.joblib")
              .named_steps["clf"].get_params()[k]
              for k in ["n_estimators", "max_depth", "learning_rate", "subsample",
                        "colsample_bytree", "min_child_weight", "gamma",
                        "reg_alpha", "reg_lambda"]}
    rf_hp = {k: joblib.load("results_strict/models/RF_pipeline.joblib")
             .named_steps["clf"].get_params()[k]
             for k in ["n_estimators", "max_depth", "min_samples_leaf",
                       "max_features", "min_samples_split"]}

    ES = train.ESTIMATOR_SEED
    arms = {}

    # ---- XGBoost lead model ----
    xgb_corr = xgb.XGBClassifier(eval_metric="logloss", random_state=ES, n_jobs=-1,
                                 scale_pos_weight=spw, **xgb_hp)
    xgb_unc  = xgb.XGBClassifier(eval_metric="logloss", random_state=ES, n_jobs=-1,
                                 scale_pos_weight=1.0, **xgb_hp)
    arms["XGB_corrected"]   = refit_eval(xgb_corr, num, cat, Xtr, ytr, Xte)
    arms["XGB_uncorrected"] = refit_eval(xgb_unc,  num, cat, Xtr, ytr, Xte)

    # ---- Random Forest (secondary check) ----
    rf_corr = RandomForestClassifier(n_jobs=-1, random_state=ES,
                                     class_weight="balanced", **rf_hp)
    rf_unc  = RandomForestClassifier(n_jobs=-1, random_state=ES,
                                     class_weight=None, **rf_hp)
    arms["RF_corrected"]   = refit_eval(rf_corr, num, cat, Xtr, ytr, Xte)
    arms["RF_uncorrected"] = refit_eval(rf_unc,  num, cat, Xtr, ytr, Xte)

    # Metrics table
    rows = []
    for name, p in arms.items():
        m = all_metrics(yte, p); m = {"model": name, **m}
        rows.append(m)
    tbl = pd.DataFrame(rows).set_index("model").round(4)
    print(tbl.to_string(), "\n")

    # Paired bootstrap deltas (uncorrected - corrected)
    print("Paired bootstrap delta (uncorrected - corrected), 95% CI:")
    delta_rows = []
    for base in ["XGB", "RF"]:
        pc, pu = arms[f"{base}_corrected"], arms[f"{base}_uncorrected"]
        for label, fn in [("AUROC", roc_auc_score),
                          ("Brier", brier_score_loss),
                          ("ICI",   lambda y, p: evalm.compute_ici(y, p))]:
            mu, lo, hi = paired_boot_delta(yte, pc, pu, fn)
            print(f"  {base} d{label:6s} = {mu:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
            delta_rows.append(dict(model=base, metric=label,
                                   delta=round(mu, 4), lo=round(lo, 4),
                                   hi=round(hi, 4)))
    pd.DataFrame(delta_rows).to_csv(
        OUT_DIR / "imbalance_sensitivity_deltas.csv", index=False)
    print()

    tbl.to_csv(OUT_DIR / "imbalance_sensitivity_metrics.csv")
    print("Saved:", OUT_DIR / "imbalance_sensitivity_metrics.csv")

    # Calibration comparison figure (lead model)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="Perfect calibration")
    for name, colour in [("XGB_corrected", "#228833"),
                         ("XGB_uncorrected", "#EE6677")]:
        p = arms[name]
        fp, mp = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        b = brier_score_loss(yte, p); ic = evalm.compute_ici(yte, p)
        lbl = ("with correction" if "corrected" == name.split("_")[1]
               else "no correction")
        ax.plot(mp, fp, "o-", color=colour, ms=4, lw=1.4,
                label=f"XGBoost, {lbl}\n(Brier={b:.3f}, ICI={ic:.3f})")
    ax.set_xlabel("Predicted probability of poor outcome")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration with and without imbalance correction\n"
                 "(held-out test set)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"imbalance_sensitivity_calibration.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("Saved:", OUT_DIR / "imbalance_sensitivity_calibration.png")


if __name__ == "__main__":
    main()
