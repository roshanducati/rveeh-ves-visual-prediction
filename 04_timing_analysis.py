"""
04_timing_analysis.py
=====================
Reports timing intervals between key clinical events in the database.

Intervals computed
------------------
1. Presentation -> Final review (final_visit_date - admission_date)
   "How long until the last recorded visit?"

2. Presentation -> Initial intervention (intervention_date - admission_date)
   The vitreous tap/injection date is the closest available proxy for
   when cultures were collected.  Microbiological results typically
   return 48-72 h after collection, so culture-result availability is
   approximately intervention_date + 2-3 days.

3. Presentation -> Discharge (discharge_date - admission_date)
   Inpatient stay duration.

All intervals are in days.  The episode-level flattening from
01_preprocess.py is replicated here so that the analysis runs on the
same unique-episode table.
"""

import pandas as pd
from pathlib import Path

RAW_FILE = Path("data/raw/VictorianEndophthalm_DATA_2026-05-17_2131.xlsx")
MAX_FOLLOWUP_DAYS = 3650


def load_episodes():
    df = pd.read_excel(RAW_FILE, sheet_name=0)
    # Review arm only
    rev = df[df["redcap_event_name"] == "review_arm_1"].copy()
    # Match 01_preprocess.py: most recent non-missing value per field after
    # repeat-instance sorting, including one record per patient when admission
    # date is unavailable.
    rev = rev.sort_values(["rveeh_ur", "admission_date", "redcap_repeat_instance"])
    has_date = rev.dropna(subset=["admission_date"])
    eps_with_date = (
        has_date.groupby(["rveeh_ur", "admission_date"], dropna=False)
        .last()
        .reset_index()
    )
    no_date = rev[rev["admission_date"].isna()]
    eps_without_date = no_date.groupby("rveeh_ur").last().reset_index()
    eps = pd.concat([eps_with_date, eps_without_date], ignore_index=True)
    return eps


def to_date(series):
    return pd.to_datetime(series, errors="coerce")


def days_between(a, b):
    """Signed difference in days (b - a); NaN if either is missing."""
    return (to_date(b) - to_date(a)).dt.days


def clean_followup_interval(series):
    """Remove negative and over-10-year follow-up intervals."""
    return series.mask((series < 0) | (series > MAX_FOLLOWUP_DAYS))


def summarise(series, label):
    s = series.dropna()
    print(f"\n{label}")
    print(f"  N (non-missing): {len(s)}")
    print(f"  Median  (IQR) : {s.median():.1f} days  "
          f"({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})")
    print(f"  Mean    (SD)  : {s.mean():.1f} +/- {s.std():.1f}")
    print(f"  Range         : {s.min():.0f}-{s.max():.0f} days")
    # Distribution buckets
    buckets = [
        ("Same day (0)",          (s == 0).sum()),
        ("1-7 days",              ((s >= 1)  & (s <= 7)).sum()),
        ("8-30 days",             ((s >= 8)  & (s <= 30)).sum()),
        ("31-90 days",            ((s >= 31) & (s <= 90)).sum()),
        ("91-365 days",           ((s >= 91) & (s <= 365)).sum()),
        ("> 1 year",              (s > 365).sum()),
    ]
    print("  Distribution:")
    for name, n in buckets:
        pct = 100 * n / len(s) if len(s) > 0 else 0
        print(f"    {name:20s}: {n:4d}  ({pct:.1f}%)")


def main():
    print("=" * 60)
    print("TIMING ANALYSIS: Victorian Endophthalmitis Database")
    print("=" * 60)

    eps = load_episodes()
    print(f"\nUnique episodes: {len(eps)}")

    # 1. Presentation -> Final review
    eps["days_to_final_visit"] = days_between(
        eps["admission_date"], eps["final_visit_date"])
    # Apply the same plausibility window as 01_preprocess.py. Clean the stored
    # interval as well as the descriptive series so timing_analysis.csv and the
    # printed summary cannot disagree.
    d_review = eps["days_to_final_visit"]
    n_neg = int((d_review < 0).sum())
    n_long = int((d_review > MAX_FOLLOWUP_DAYS).sum())
    if n_neg:
        print(f"\n  Note: {n_neg} episodes with final_visit_date before "
              f"admission_date excluded as implausible.")
    if n_long:
        print(f"  Note: {n_long} episodes with follow-up longer than 10 years "
              "excluded as implausible.")
    eps["days_to_final_visit"] = clean_followup_interval(d_review)
    d_review = eps["days_to_final_visit"]
    summarise(d_review, "PRESENTATION -> FINAL REVIEW (days_to_final_visit)")

    # 2. Presentation -> Intervention (culture collection proxy)
    eps["days_to_intervention"] = days_between(
        eps["admission_date"], eps["intervention_date"])
    d_intv = eps["days_to_intervention"]
    n_neg2 = (d_intv < 0).sum()
    if n_neg2:
        print(f"\n  Note: {n_neg2} episodes with intervention before "
              f"admission excluded as implausible.")
    d_intv = d_intv[d_intv >= 0]
    summarise(d_intv,
              "PRESENTATION -> INTERVENTION / CULTURE COLLECTION PROXY "
              "(days_to_intervention)")

    # Also report first intravitreal treatment date as alternative proxy
    eps["days_to_ivt1"] = days_between(
        eps["admission_date"], eps["intravitreal_tx_date_1"])
    d_ivt1 = eps["days_to_ivt1"]
    d_ivt1 = d_ivt1[d_ivt1 >= 0]
    summarise(d_ivt1,
              "PRESENTATION -> FIRST INTRAVITREAL TREATMENT "
              "(days_to_ivt1, alternative culture-collection proxy)")

    # 3. Presentation -> Discharge
    eps["days_to_discharge"] = days_between(
        eps["admission_date"], eps["discharge_date"])
    d_dc = eps["days_to_discharge"]
    d_dc = d_dc[d_dc >= 0]
    summarise(d_dc, "PRESENTATION -> DISCHARGE (days_to_discharge)")

    # 4. Cross-tab: culture available at presentation?
    print("\n" + "=" * 60)
    print("CULTURE RESULT AVAILABILITY AT PRESENTATION")
    print("=" * 60)
    print("(Assuming culture result available ~48-72 h after collection)")
    print("(Using intervention_date as culture collection date)")
    # Do not treat a missing intervention date as same-day collection.
    d_result = (eps["days_to_intervention"].dropna() + 2).clip(lower=0)
    print(f"  Estimated time to culture result, median: "
          f"{d_result.dropna().median():.1f} days after presentation")
    print(f"  => Culture data NOT available at presentation for most cases.")
    print("  Culture data are typically available within 2-4 days.")

    # Save
    timing_df = eps[["rveeh_ur", "admission_date",
                      "days_to_final_visit", "days_to_intervention",
                      "days_to_ivt1", "days_to_discharge"]].copy()
    out_dir = Path("data_strict") if Path("data_strict").exists() else Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "timing_analysis.csv"
    timing_df.to_csv(out_path, index=False)
    print(f"\nTiming data saved: {out_path}")


if __name__ == "__main__":
    main()
