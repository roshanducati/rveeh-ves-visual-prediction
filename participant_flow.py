"""
participant_flow.py
===================
Generate Figure 1 (participant flow) for the Victorian Endophthalmitis ML study.

Reproduces the exclusion cascade directly from the raw REDCap export and the
reported strict processed dataset so the box counts cannot drift from the
reported results:

  REDCap export  ->  1700 episodes (repeat instances consolidated)
                 ->  400 excluded (no documented final visual outcome)
                 ->  1300 analysed episodes
                 ->  patient-level split (GroupShuffleSplit, seed 61, 20% test)
                     into 1038 development and 262 held-out test episodes.

Outputs (results_strict/figures/):
  fig1_participant_flow.pdf / .png

Run:  python participant_flow.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import GroupShuffleSplit

PROCESSED = Path("data_strict/processed_episodes.csv")
RAW_FILE  = Path("data/raw/VictorianEndophthalm_DATA_2026-05-17_2131.xlsx")
FIG_DIR   = Path("results_strict/figures")
SPLIT_SEED = 61          # fixed patient-level split seed (see 02_train.py)
TEST_SIZE  = 0.20

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      9.5,
    "figure.dpi":     150,
})


def gather_counts():
    """Recompute every box count from source so the figure stays in sync."""
    proc = pd.read_csv(PROCESSED)
    n_episodes   = len(proc)
    n_patients   = proc["rveeh_ur"].nunique()

    known = proc.dropna(subset=["poor_outcome"]).copy()
    n_excluded = n_episodes - len(known)
    n_model    = len(known)
    n_model_pts = known["rveeh_ur"].nunique()
    n_poor = int((known["poor_outcome"] == 1).sum())
    n_good = int((known["poor_outcome"] == 0).sum())

    # Patient-level split (identical to 02_train.py)
    groups = known["rveeh_ur"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=SPLIT_SEED)
    tr_idx, te_idx = next(gss.split(known, known["poor_outcome"], groups=groups))

    dev = known.iloc[tr_idx]
    tst = known.iloc[te_idx]

    # Admission-date span of the analysed cohort
    ad = pd.to_datetime(known["admission_date"], errors="coerce")
    date_min, date_max = ad.min(), ad.max()

    # Registry export raw arm sizes (for the top annotation only)
    try:
        raw = pd.read_excel(RAW_FILE, sheet_name=0)
        n_review = int((raw["redcap_event_name"] == "review_arm_1").sum())
    except Exception:
        n_review = None

    return dict(
        n_review=n_review,
        n_episodes=n_episodes, n_patients=n_patients,
        n_excluded=n_excluded,
        n_model=n_model, n_model_pts=n_model_pts,
        n_poor=n_poor, n_good=n_good,
        pct_poor=100 * n_poor / n_model, pct_good=100 * n_good / n_model,
        n_dev=len(dev), n_dev_pts=dev["rveeh_ur"].nunique(),
        dev_poor=int((dev["poor_outcome"] == 1).sum()),
        dev_good=int((dev["poor_outcome"] == 0).sum()),
        n_tst=len(tst), n_tst_pts=tst["rveeh_ur"].nunique(),
        tst_poor=int((tst["poor_outcome"] == 1).sum()),
        tst_good=int((tst["poor_outcome"] == 0).sum()),
        date_min=date_min, date_max=date_max,
    )


# Drawing helpers
BOX_FC   = "#EEF3F8"
BOX_EC   = "#33507A"
EXCL_FC  = "#F7EEEE"
EXCL_EC  = "#9A4A4A"
ARROW_C  = "#33507A"


def box(ax, x, y, w, h, text, fc=BOX_FC, ec=BOX_EC, fontsize=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3, facecolor=fc, edgecolor=ec, mutation_aspect=1))
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color="#12233B", zorder=5)


def v_arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.3, color=ARROW_C, shrinkA=0, shrinkB=0))


def branch_arrow(ax, x0, x1, y):
    """Horizontal connector branching off the main trunk into a side box."""
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.3, color=ARROW_C, shrinkA=0, shrinkB=0))


def build_figure(c):
    fig, ax = plt.subplots(figsize=(7.8, 7.9))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(3.6, 12.1)
    ax.axis("off")

    cx = 3.7          # central column x
    ex = 8.55         # exclusion column x
    w  = 5.2          # main box width
    ew = 3.1          # exclusion box width

    # 1. Registry source
    box(ax, cx, 11.05, w, 1.25,
        "Victorian Eye and Ear Hospital\nendophthalmitis registry (REDCap export)\n"
        f"{c['n_episodes']:,} endophthalmitis episodes  ({c['n_patients']:,} patients)",
        fontsize=9.2)

    # Trunk from the registry box down to the analysed cohort, with the
    # exclusion box branching off it partway down.
    v_arrow(ax, cx, 10.425, 8.60)
    ax.text(cx + 0.25, 10.13,
            "REDCap repeat instances consolidated\nto one row per episode",
            ha="left", va="center", fontsize=7.0, style="italic", color="#4A5A70")

    # 2. Exclusion (branches off the trunk)
    box(ax, ex, 9.35, ew, 1.1,
        "Excluded\n"
        f"{c['n_excluded']} episodes with no\ndocumented final visual outcome",
        fc=EXCL_FC, ec=EXCL_EC, fontsize=8.4)
    branch_arrow(ax, cx, ex - ew / 2 - 0.03, 9.35)

    # 3. Analysed cohort
    dmin = c["date_min"].strftime("%b %Y")
    dmax = c["date_max"].strftime("%b %Y")
    box(ax, cx, 7.85, w, 1.5,
        f"Analysed cohort: {c['n_model']:,} episodes ({c['n_model_pts']:,} patients)\n"
        f"Admitted {dmin}-{dmax}\n"
        f"{c['n_poor']} poor ({c['pct_poor']:.1f}%)   |   "
        f"{c['n_good']} good ({c['pct_good']:.1f}%)",
        fontsize=9.2, weight="bold")

    v_arrow(ax, cx, 7.10, 6.35)
    ax.text(cx + 0.25, 6.73,
            "Patient-level split\n(GroupShuffleSplit, 80/20)",
            ha="left", va="center", fontsize=7.0, style="italic", color="#4A5A70")

    # 4. Split into development and test
    lx, rx = 1.85, 5.55
    bw = 3.5
    # split junction
    ax.add_patch(FancyArrowPatch((cx, 6.35), (lx, 6.35), arrowstyle="-",
                                 linewidth=1.3, color=ARROW_C))
    ax.add_patch(FancyArrowPatch((cx, 6.35), (rx, 6.35), arrowstyle="-",
                                 linewidth=1.3, color=ARROW_C))
    v_arrow(ax, lx, 6.35, 5.65)
    v_arrow(ax, rx, 6.35, 5.65)

    box(ax, lx, 4.95, bw, 1.4,
        "Development set\n"
        f"{c['n_dev']:,} episodes  ({c['n_dev_pts']:,} patients)\n"
        f"{c['dev_poor']} poor  |  {c['dev_good']} good\n"
        "Grouped CV tuning + model fitting",
        fontsize=8.7)

    box(ax, rx, 4.95, bw, 1.4,
        "Held-out test set\n"
        f"{c['n_tst']} episodes  ({c['n_tst_pts']} patients)\n"
        f"{c['tst_poor']} poor  |  {c['tst_good']} good\n"
        "Final performance evaluation",
        fontsize=8.7)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    c = gather_counts()

    print("Participant-flow counts")
    print("-----------------------")
    for k, v in c.items():
        print(f"  {k:12s}: {v}")

    fig = build_figure(c)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig1_participant_flow.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print("Saved:", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
