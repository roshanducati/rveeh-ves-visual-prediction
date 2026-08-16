"""
01_preprocess.py
================
Preprocessing pipeline for the Victorian Endophthalmitis ML study.

Clinical question
-----------------
Given a patient with endophthalmitis, what is the probability of a poor visual
outcome from presentation features, with or without early microbiology?

Input
-----
  VictorianEndophthalm_DATA_2026-05-17_2131.xlsx: REDCap export

Outputs
-------
  data/processed_episodes.csv: one row per episode, analysis-ready features
  data/va_mapping_table.csv: VA-code to logMAR reference table
  data/missingness_report.csv: per-feature missingness summary

VA Decoding
-----------
The REDCap export stores Snellen acuities as Excel date-like values via
two encoding schemes, plus free-text for subnormal categories:

  Pattern A  YYYY-06-01: Snellen 6/(year % 100), e.g. 1960-06-01 gives 6/60
  Pattern B  2026-MM-06: Snellen 6/month, e.g. 2026-09-06 gives 6/9
  Pattern C  "6/X": Snellen 6/X string, e.g. "6/120"
  Categorical HM / CF / LP / NLP / EVIS: fixed logMAR values

logMAR = log10(denominator / 6) for standard Snellen; for subnormal categories
fixed values from Schulze-Bonsel et al. (2006) Graefes Arch 244:801-805.

Poor-outcome threshold
----------------------

Strict mode (--strict): final logMAR > 1.0 (strictly worse than 6/60, i.e.
                6/60 itself is classified as GOOD outcome).  Outputs written to
                data_strict/

REDCap structure
----------------
Each patient (rveeh_ur) has one registration_arm_1 row (demographics) and one
or more review_arm_1 rows (redcap_repeat_instance >= 1). Multiple instances for
the same (patient, admission_date) pair represent updated data entries; the last
(highest) instance is retained. Different admission dates for the same patient
represent distinct endophthalmitis episodes and are retained as separate rows.
Care is taken to ensure that train/test splits, including CV splitting for
hyperparameter tuning, occurs at the patient level.

Missing data
------------
All imputation is deferred to the training script so that the test fold is
never used to inform imputation parameters. This script only flags missingness
and documents rates.
"""

import argparse
import logging
import re as re_mod
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_FILE = Path("data/raw/VictorianEndophthalm_DATA_2026-05-17_2131.xlsx")
OUT_DIR  = Path("data")   # may be overridden to Path("data_strict") via --strict

# -----------------------------------------------------------------------------
# 1. VA DECODING
# -----------------------------------------------------------------------------

# Prespecified logMAR assignments for non-Snellen categories, matching the
# manuscript and Supplement S2.
CATEGORICAL_LOGMAR = {
    "CF":   1.85,   # Counting fingers  (~6/600 equivalent)
    "HM":   2.30,   # Hand motion       (~6/3000 equivalent)
    "LP":   2.70,   # Light perception
    "NLP":  3.00,   # No light perception
    "EVIS": 4.00,   # Evisceration / enucleation: eye anatomically lost
}

def decode_va(val) -> float:
    """Return logMAR for a single REDCap VA field value, or np.nan.

    REDCap day-of-month invariant (enforced by audit_va_dates() at preprocess time):
      Pattern A rows always have day == 1, month == 6 (year encodes denominator).
      Pattern B rows always have day == 6, year == 2026 (month encodes denominator).
    Branching on day first is therefore unambiguous for all values observed in
    the registry export.
    """
    import datetime as _dt
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan

    # Pandas Timestamp or standard library datetime (Excel date as datetime)
    if isinstance(val, (pd.Timestamp, _dt.datetime, _dt.date)):
        d = val.day if hasattr(val, 'day') else val.day
        m = val.month
        y = val.year
        if d == 6 and y == 2026:
            # Pattern B: month encodes Snellen denominator
            denom = float(m)
        elif d == 1 and m == 6 and y % 100 != 0:
            # Pattern A: year % 100 encodes Snellen denominator.
            denom = float(y % 100)
        else:
            log.warning("Unrecognised date-encoded VA: %s", val)
            return np.nan
        return float(np.log10(denom / 6.0))

    # String values
    s = str(val).strip()
    s_up = s.upper()

    # Categorical (CF / HM / LP / NLP / EVIS)
    if s_up in CATEGORICAL_LOGMAR:
        return CATEGORICAL_LOGMAR[s_up]

    # Snellen string "6/X"
    m_str = re_mod.match(r"^6/(\d+\.?\d*)$", s)
    if m_str:
        denom = float(m_str.group(1))
        return float(np.log10(denom / 6.0))

    log.warning("Unrecognised VA value: %r", val)
    return np.nan


def build_va_mapping_table():
    """Produce the reference VA to logMAR table."""
    rows = []
    # Categorical
    for k, v in CATEGORICAL_LOGMAR.items():
        rows.append({"raw_value": k,
                     "snellen_equiv": k,
                     "logmar": round(v, 3),
                     "note": "Subnormal / non-Snellen"})
    # All date-encoded and string Snellen values observed in the dataset
    observed_snellens = [
        ("2026-03-06", "6/3",   3),
        ("2026-04-06", "6/4",   4),
        ("2026-05-06", "6/5",   5),
        ("6/7.5",      "6/7.5", 7.5),
        ("2026-06-06", "6/6",   6),
        ("2026-09-06", "6/9",   9),
        ("2026-12-06", "6/12",  12),
        ("2015-06-01", "6/15",  15),
        ("2019-06-01", "6/19",  19),
        ("2024-06-01", "6/24",  24),
        ("2030-06-01", "6/30",  30),
        ("2036-06-01", "6/36",  36),
        ("2048-06-01", "6/48",  48),
        ("1960-06-01", "6/60",  60),
        ("1975-06-01", "6/75",  75),
        ("1996-06-01", "6/96",  96),
        ("6/120",      "6/120", 120),
        ("6/150",      "6/150", 150),
        ("6/192",      "6/192", 192),
        ("6/240",      "6/240", 240),
    ]
    for raw, snellen, denom in observed_snellens:
        rows.append({
            "raw_value":     raw,
            "snellen_equiv": snellen,
            "logmar":        round(np.log10(denom / 6.0), 3),
            "note":          "Snellen (date-encoded or string)",
        })
    return pd.DataFrame(rows).sort_values("logmar").reset_index(drop=True)


# -----------------------------------------------------------------------------
# 2. ORGANISM NORMALISATION
# -----------------------------------------------------------------------------

# Clinically meaningful organism groups for endophthalmitis.
#
# Ordering: WORST TO BEST approximate visual prognosis. normalise_organism()
# uses first-match-wins, so for polymicrobial free-text entries naming more than
# one organism the WORST-prognosis class is assigned (clinically the worst
# co-isolate governs the outcome). Specificity of every unique free-text value
# is verified empirically by audit_organism_mapping() and saved to
# organism_mapping_audit.csv.
#
# NOTE (clinical judgement, adjustable): the exact rank of bacillus,
# other_bacteria and the staph/strep tiers is approximate. Bacillus cereus and
# Gram-negatives (Pseudomonas) are placed worst; CoNS / Propionibacterium best.
# Patterns must stay mutually exclusive for SINGLE organisms (the audit's
# n_patterns_matched column flags any single-organism overlap); ordering only
# decides genuinely polymicrobial strings.
ORGANISM_PATTERNS = [
    ("gram_negative",     r"monas|\bps?\.?\s*(aer|fluor)|klebsiella|escherichia|\be\.?\s*coli\b|proteus|moraxella"
                          r"|gram.?neg|enterobacter|serratia|haemophil|\bh\.?\s*inf\w*enz|acinetobacter"
                          r"|citrobacter|burkholderia|morganella|chryseobacter|ochrobactrum|neisseria|\bn\.?\s*mening"),
    ("bacillus",          r"\bbacillus\b"),
    ("fungal",            r"candida|\bc\.\s*(albican|tropical|dublin|parapsil)|aspergill|fungal|fungus|yeast"
                           r"|fusarium|scedospori|trichospor|curvularia|cryptococc|penicillium|paecilomyces"
                           r"|alternaria|meyerozyma|gongronella|fil\s*fungi"),
    ("cons",              r"strep\s+(warneri|lugd\w*)"),
    ("streptococcus",     r"staph\s+mitis|strep|streptococ|pneumococ|viridans|abiotrophia|granulicatella|gemella"
                           r"|\bs\.?\s*(mitis|pneum|oral|saliv|sangui|agalact|dysgalact|gordon|pyogen|mutans"
                           r"|cristat|infant|vestibul|parasang|anginos|gallolyt|constellat)"),
    ("enterococcus",      r"enterococ|\bent\s+faecal|\be\.?\s*faecal"),
    ("staph_aureus",      r"staph\w*\W+aure|staphylococcus\s+aure|\bs\.?\s*aure|\bst\.?\s*aure|\bmssa\b|\bmrsa\b"),
    ("other_bacteria",    r"mycobacter|nocardia|coryne|cornebacter|cornybact|listeria|clostri|micrococcus"
                          r"|rothia|rhodococc|brevibacter|cellulosimicrob|fusobacter|kytococcus"),
    ("cons",              r"staph\w*\W+epi|coag.?neg|coagulase.?neg|staphylococcus\s+epid|\bs\.?\s*epid"
                          r"|\bst\.?\s+epi|\bcns\b|lugdunens|\bs\.?\s*lugd|warneri|hominis|capitis|\bs\.?\s*capit"
                          r"|haemolyticus|caprae|cohnii|simulans|schleifer|staph\s+sp|staph\s+species"),
    ("propionibacterium", r"propionibact|\bprop\w*\s+sp|p\.?\s*acnes|cutibacteri"),
]

# Non-specific / non-organism free text (no organism identified, mixed flora,
# polymorphs, bare "gram positive cocci", contaminant, insufficient sample,
# pending). These map to "unknown" rather than being forced into a specific
# bacterial class. Checked AFTER the organism patterns so a genuine isolate
# mentioned alongside non-specific words (e.g. "candida ... no growth on tap")
# is still classified by the organism.
NONSPECIFIC_PATTERN = (
    r"no\s*growth|\bng\b|\bpmn\b|\bpolys?\b|polymorph|mixed skin flora|mixed organism|mixed staph"
    r"|insufficient|contamin|gram.?pos\w*\s+(cocc|rod)|g\s*\+\s*ve|\bgpc\b|cocc?obacilli|pending"
    r"|\bnone\b"
    r"|significance uncertain"
)


def normalise_organism(raw) -> str:
    """Map free-text organism to a standardised category string.

    Order of resolution:
      1. ORGANISM_PATTERNS (worst to best prognosis, first match wins).
      2. NONSPECIFIC_PATTERN gives "unknown" (no identifiable organism).
      3. Fallback gives "other_bacteria" (text looks like an organism but matches
         no known group).
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return "unknown"
    s = str(raw).strip().lower()
    for label, pattern in ORGANISM_PATTERNS:
        if re_mod.search(pattern, s):
            return label
    if re_mod.search(NONSPECIFIC_PATTERN, s):
        return "unknown"
    return "other_bacteria"


# -----------------------------------------------------------------------------
# 3. LOAD & STRUCTURE
# -----------------------------------------------------------------------------

def load_data():
    log.info("Reading %s...", RAW_FILE)
    df = pd.read_excel(RAW_FILE, sheet_name=0)
    log.info("Shape: %s", df.shape)
    return df


def split_arms(df: pd.DataFrame):
    reg = df[df["redcap_event_name"] == "registration_arm_1"].copy()
    rev = df[df["redcap_event_name"] == "review_arm_1"].copy()
    log.info("Registration arm: %d rows, %d unique patients",
             len(reg), reg["rveeh_ur"].nunique())
    log.info("Review arm: %d rows, %d unique patients",
             len(rev), rev["rveeh_ur"].nunique())
    return reg, rev


def merge_gender(rev: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    """
    Gender is stored only in the registration arm (1=male, 2=female).
    Merge it into the review arm.
    """
    gender_map = reg.dropna(subset=["gender"]).drop_duplicates("rveeh_ur") \
                    .set_index("rveeh_ur")["gender"]
    rev = rev.copy()
    rev["gender"] = rev["rveeh_ur"].map(gender_map)   # 1=male, 2=female as per data_dict
    rev["gender_female"] = (rev["gender"] == 2).astype(float)
    rev.loc[rev["gender"].isna(), "gender_female"] = np.nan
    log.info("Gender merged: %.1f%% complete in review arm",
             100 * rev["gender_female"].notna().mean())
    return rev


def flatten_episodes(rev: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per episode (unique rveeh_ur and admission_date).

    REDCap repeat instances >= 2 for the same (patient, admission_date) are
    treated as data updates. After sorting by repeat instance, pandas
    ``GroupBy.last`` retains the most recent non-missing value in each field.
    Different admission dates for the same patient represent distinct
    endophthalmitis episodes and are kept.
    Rows with a missing admission_date are deduplicated by patient alone (last
    instance kept).
    """
    rev = rev.sort_values(["rveeh_ur", "admission_date",
                           "redcap_repeat_instance"])

    # Rows WITH a valid admission date
    has_date  = rev.dropna(subset=["admission_date"])
    dedup_date = has_date.groupby(["rveeh_ur", "admission_date"],
                                  dropna=False).last().reset_index()

    # Rows WITHOUT an admission date: deduplicate by patient only
    no_date = rev[rev["admission_date"].isna()]
    dedup_no_date = no_date.groupby("rveeh_ur").last().reset_index()

    out = pd.concat([dedup_date, dedup_no_date], ignore_index=True)
    log.info("Episodes after flattening: %d (from %d review-arm rows)",
             len(out), len(rev))
    return out


# -----------------------------------------------------------------------------
# 4. VA DECODING & AFFECTED-EYE EXTRACTION
# -----------------------------------------------------------------------------

VA_PRES_COLS = {
    "RE":  "re_va_on_presentation",
    "LE":  "le_va_on_presentation",
}
VA_FINAL_COLS = {
    "RE":  "re_va_final",
    "LE":  "le_va_final",
}


def _worse_documented_eye(row, va_cols):
    """Return the eye with worse available VA, using RE for an exact tie."""
    re_logmar = decode_va(row.get(va_cols["RE"]))
    le_logmar = decode_va(row.get(va_cols["LE"]))
    re_known = np.isfinite(re_logmar)
    le_known = np.isfinite(le_logmar)
    if re_known and le_known:
        return "LE" if le_logmar > re_logmar else "RE"
    if re_known:
        return "RE"
    if le_known:
        return "LE"
    return None


def select_analysis_eye(row):
    """Select one anatomical eye for both presenting and final VA.

    Recorded unilateral laterality is respected. For bilateral, unspecified,
    or missing laterality, the eye with worse presenting VA is selected. If the
    presenting values are tied, the worse final eye resolves the tie; both eyes
    met the presentation rule. If no presenting VA is available, the worse
    documented final eye is retained as an explicit outcome-only fallback so
    that an otherwise observed episode outcome is not discarded.

    Returns `(eye, basis)`, where `eye` is `RE`, `LE`, or `None` and `basis`
    records how the choice was made.
    """
    recorded_eye = str(row.get("affected_eye", "")).strip().upper()
    if recorded_eye in {"RE", "LE"}:
        return recorded_eye, "recorded_laterality"

    re_presenting = decode_va(row.get(VA_PRES_COLS["RE"]))
    le_presenting = decode_va(row.get(VA_PRES_COLS["LE"]))
    re_known = np.isfinite(re_presenting)
    le_known = np.isfinite(le_presenting)

    if re_known and le_known:
        if not np.isclose(re_presenting, le_presenting):
            eye = "RE" if re_presenting > le_presenting else "LE"
            return eye, "worse_presenting_va"
        final_eye = _worse_documented_eye(row, VA_FINAL_COLS)
        return final_eye or "RE", "tied_presenting_va"
    if re_known:
        return "RE", "only_presenting_va"
    if le_known:
        return "LE", "only_presenting_va"

    final_eye = _worse_documented_eye(row, VA_FINAL_COLS)
    if final_eye is not None:
        return final_eye, "final_va_fallback"
    return None, "unavailable"


def extract_eye_va(row, va_cols, selected_eye=None) -> float:
    """Return logMAR from the selected anatomical eye."""
    eye = selected_eye
    if eye not in {"RE", "LE"}:
        eye, _ = select_analysis_eye(row)
    return decode_va(row.get(va_cols[eye])) if eye is not None else np.nan


# -----------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

POOR_OUTCOME_LOGMAR_THRESHOLD = 1.0   # logMAR threshold for poor outcome


def build_features(eps: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Construct the analysis-ready feature matrix from the flattened episode
    table. Features are limited to presentation information and early
    microbiology; post-presentation treatment decisions are excluded.

    Feature groups
    --------------
    Demographics
        age_years: continuous; age at admission
        gender_female: binary (1=female, 0=male)

    Presenting visual acuity
        pres_logmar: logMAR of the affected eye at presentation.
                             The ONLY presenting-VA variable used by the model
                             (continuous; see NUMERIC_FEATURES in 02_train.py).
        pres_poor_va: binary >=6/60 flag. DESCRIPTIVE-ONLY
                             (cohort summaries); NOT a model input.

    Clinical signs at presentation
        fundus_visible: 1=fundus visible, 0=obscured (media opacification)
        rapd_present: 1=RAPD, 0=absent [61% missing; imputed in train]

    Medical co-morbidities
        diabetes: binary
        immune_suppressed: binary [93% missing; imputed in train]

    Precipitating aetiology (REDCap checkbox flags, mutually inclusive)
    Codes per data dictionary [precipitating_factor]:
        etiol_cataract_sx: post-cataract/IOL surgery (code 1)
        etiol_glaucoma_sx: post-glaucoma/drainage surgery (code 2)
        etiol_trauma: penetrating eye injury (code 3)
        etiol_corneal_ulcer: corneal ulcer (code 4)
        etiol_endogenous: metastatic/endogenous (code 5)
        etiol_ivi: post-intravitreal injection (code 7)
        etiol_other: other ocular procedure (code 98)

    Microbiology (available once cultures result, typically 48-72 h)
        culture_positive: 1=documented positive culture; 0=not documented
                             positive. The organism category distinguishes
                             confirmed no growth from unavailable status.
        organism_cat: normalised organism category (one-hot in train).
                             Assigned by normalise_organism() via worst-to-best
                             first-match (polymicrobial entries take the
                             worst-prognosis class). Non-specific/non-organism
                             free text gives "unknown"; no growth gives
                             "no_growth"; missing culture gives "unknown".
                             Per-value mapping is
                             verified by organism_mapping_audit.csv.

    Target
        poor_outcome: in strict mode, 1 if final logMAR > 1.0, else 0
                             [NaN if unknown]
        final_logmar: continuous final logMAR
    """
    out = eps.copy()

    # Select one anatomical eye once, then use it for both presenting and final
    # VA. This prevents the two measurements from silently referring to
    # different eyes in bilateral or unspecified-eye episodes.
    eye_selection = out.apply(select_analysis_eye, axis=1, result_type="expand")
    eye_selection.columns = ["analysis_eye", "analysis_eye_basis"]
    out[["analysis_eye", "analysis_eye_basis"]] = eye_selection

    # Presenting VA
    out["pres_logmar"] = out.apply(
        lambda r: extract_eye_va(r, VA_PRES_COLS, r["analysis_eye"]), axis=1)

    # DESCRIPTIVE-ONLY presenting-VA flag (Table 1 / cohort summaries). NOT a
    # model input. The model uses the continuous pres_logmar (NUMERIC_FEATURES
    # in 02_train.py). Boundary tracks the outcome definition (strict: > 1.0)
    # so descriptive tables and Methods agree.
    if strict:
        out["pres_poor_va"] = (out["pres_logmar"] > 1.0).astype(float)
    else:
        out["pres_poor_va"] = (out["pres_logmar"] >= 1.0).astype(float)
    out.loc[out["pres_logmar"].isna(), "pres_poor_va"] = np.nan

    # Final VA and outcome
    out["final_logmar"] = out.apply(
        lambda r: extract_eye_va(r, VA_FINAL_COLS, r["analysis_eye"]), axis=1)
    # Default (legacy): >= threshold (includes 6/60).  Strict: > threshold (excludes 6/60).
    if strict:
        out["poor_outcome"] = (out["final_logmar"] > POOR_OUTCOME_LOGMAR_THRESHOLD
                               ).astype(float)
    else:
        out["poor_outcome"] = (out["final_logmar"] >= POOR_OUTCOME_LOGMAR_THRESHOLD
                               ).astype(float)
    out.loc[out["final_logmar"].isna(), "poor_outcome"] = np.nan

    # Demographics
    out["age_years"] = pd.to_numeric(out["admission_age"], errors="coerce")
    # gender_female already added in merge_gender()

    # Binary clinical variables
    # _yn maps to {0, 1, NaN}. Anything outside {0, 1} (e.g. REDCap '2 = unknown'
    # if present in any y/n field) becomes NaN and is downstream-imputed by
    # IterativeImputer,
    # rather than being silently clipped to 1.
    def _yn(col):
        raw = pd.to_numeric(out[col], errors="coerce")
        cleaned = raw.where(raw.isin([0.0, 1.0]))
        n_dropped = int((raw.notna() & cleaned.isna()).sum())
        if n_dropped > 0:
            log.warning(
                "%s: %d values outside {0,1} mapped to NaN", col, n_dropped)
        return cleaned

    out["fundus_visible"]    = _yn("fundus_visible_yn")
    out["rapd_present"]      = _yn("rapd_yn")
    out["diabetes"]          = _yn("diabetes_yn")
    out["immune_suppressed"] = _yn("immune_suppressed_yn")

    # Prior ophthalmic surgery (NaN coded as unknown; treated as missing)
    out["prior_surgery"] = _yn("surgery_yn")

    # Aetiology flags
    # Code-to-name mapping verified against the REDCap data dictionary codebook
    # (documents/reference/codebook endoph study.pdf, field [precipitating_factor]):
    #   1 cataract/IOL sx; 2 glaucoma/drainage sx; 3 penetrating eye injury
    #   4 corneal ulcer; 5 metastatic/endogenous; 7 intravitreal injection
    #   98 other ocular procedure   (no code 6 in this export)
    out["etiol_cataract_sx"]   = pd.to_numeric(
        out["precipitating_factor___1"], errors="coerce")
    out["etiol_glaucoma_sx"]   = pd.to_numeric(
        out["precipitating_factor___2"], errors="coerce")
    out["etiol_trauma"]        = pd.to_numeric(
        out["precipitating_factor___3"], errors="coerce")   # penetrating eye injury
    out["etiol_corneal_ulcer"] = pd.to_numeric(
        out["precipitating_factor___4"], errors="coerce")   # corneal ulcer
    out["etiol_endogenous"]    = pd.to_numeric(
        out["precipitating_factor___5"], errors="coerce")   # metastatic/endogenous
    out["etiol_ivi"]           = pd.to_numeric(
        out["precipitating_factor___7"], errors="coerce")
    out["etiol_other"]         = pd.to_numeric(
        out["precipitating_factor___98"], errors="coerce")

    # Microbiology
    out["culture_positive"] = _yn("growths_yn")
    out["organism_cat"]     = out["microorganism"].apply(normalise_organism)
    # If an organism is explicitly documented but the culture-status checkbox is
    # blank, preserve the specific microbiology rather than erasing it as
    # "unknown". Blank/non-specific organism text remains handled below.
    organism_text_present = (
        out["microorganism"].notna()
        & out["microorganism"].astype(str).str.strip().ne("")
    )
    mask_blank_status_specific_org = (
        out["culture_positive"].isna()
        & organism_text_present
        & ~out["organism_cat"].isin(["unknown"])
    )
    if mask_blank_status_specific_org.any():
        log.warning(
            "Microbiology: %d episode(s) have organism text but blank growths_yn; "
            "setting culture_positive=1 and preserving organism_cat.",
            int(mask_blank_status_specific_org.sum()),
        )
    out.loc[mask_blank_status_specific_org, "culture_positive"] = 1.0
    # For cases with no growth, override organism category to "no_growth"
    mask_no_growth = out["culture_positive"] == 0
    out.loc[mask_no_growth, "organism_cat"] = "no_growth"
    # For unavailable culture status, label the organism category unknown and
    # encode the numeric indicator as not documented positive. The paired
    # organism category keeps unavailable status distinct from confirmed no
    # growth, while avoiding a redundant imputed value in the numeric branch.
    mask_miss_cult = out["culture_positive"].isna()
    out.loc[mask_miss_cult, "organism_cat"]     = "unknown"
    out.loc[mask_miss_cult, "culture_positive"] = 0.0

    # Primary procedure classification
    # NOT a model feature. `intervention___v` / `intervention___b` reflect
    # treatment decisions made *after* presentation and are therefore excluded
    # from `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` in 02_train.py. The
    # column is retained as a descriptive variable only (initial procedure type
    # is reported as not-a-model-input and balance-checked across the train/test
    # split in the manuscript; it is not used to stratify model performance).
    #
    # REDCap checkbox: intervention___v = vitrectomy, intervention___b = biopsy.
    # Any vitrectomy flag (with or without concurrent biopsy) gives "vitrectomy".
    # Biopsy only (v=0, b=1) gives "biopsy". Neither known gives "unknown".
    _vitx = pd.to_numeric(out["intervention___v"], errors="coerce").clip(0, 1)
    _biop = pd.to_numeric(out["intervention___b"], errors="coerce").clip(0, 1)
    out["primary_vitrectomy"] = _vitx
    out["primary_biopsy"]     = _biop
    out["procedure_group"] = np.select(
        [_vitx == 1, (_vitx == 0) & (_biop == 1)],
        ["vitrectomy", "biopsy"],
        default="unknown",
    )
    out.loc[_vitx.isna() & _biop.isna(), "procedure_group"] = np.nan

    # Follow-up duration (presentation to final review)
    out["days_to_final_visit"] = (
        pd.to_datetime(out["final_visit_date"], errors="coerce") -
        pd.to_datetime(out["admission_date"], errors="coerce")
    ).dt.days
    # Negative and >10-year values are implausible data-entry errors, so use NaN
    out.loc[out["days_to_final_visit"] < 0,    "days_to_final_visit"] = np.nan
    out.loc[out["days_to_final_visit"] > 3650,  "days_to_final_visit"] = np.nan

    # Affected eye (retained for reference)
    out["affected_eye_clean"] = out["affected_eye"].fillna("UNK").str.strip().str.upper()

    return out


# -----------------------------------------------------------------------------
# 6. MISSINGNESS REPORT
# -----------------------------------------------------------------------------

FEATURE_COLS = [
    "age_years", "gender_female",
    "pres_logmar", "pres_poor_va",
    "fundus_visible", "rapd_present",
    "diabetes", "immune_suppressed", "prior_surgery",
    "etiol_cataract_sx", "etiol_glaucoma_sx", "etiol_trauma",
    "etiol_corneal_ulcer", "etiol_endogenous", "etiol_ivi", "etiol_other",
    "culture_positive", "organism_cat",
    "final_logmar", "poor_outcome",
]

FEATURE_LABELS = {
    "age_years":          "Age at admission (years)",
    "gender_female":      "Sex (female)",
    "pres_logmar":        "Presenting VA (logMAR, affected eye)",
    "pres_poor_va":       "Presenting VA threshold flag (binary, descriptive)",
    "fundus_visible":     "Fundus visible on presentation",
    "rapd_present":       "Relative afferent pupillary defect (RAPD)",
    "diabetes":           "Diabetes mellitus",
    "immune_suppressed":  "Immune suppression",
    "prior_surgery":      "Prior ophthalmic surgery",
    "etiol_cataract_sx":   "Aetiology: post-cataract/IOL surgery",
    "etiol_glaucoma_sx":   "Aetiology: post-glaucoma/drainage surgery",
    "etiol_trauma":        "Aetiology: penetrating eye injury",
    "etiol_corneal_ulcer": "Aetiology: corneal ulcer",
    "etiol_endogenous":    "Aetiology: metastatic/endogenous",
    "etiol_ivi":           "Aetiology: post-intravitreal injection",
    "etiol_other":         "Aetiology: other ocular procedure",
    "culture_positive":   "Documented culture positivity",
    "organism_cat":       "Organism category (normalised)",
    "final_logmar":       "Final VA (logMAR, affected eye) [outcome]",
    "poor_outcome":       "Poor visual outcome (binary target)",
}


def missingness_report(feat: pd.DataFrame, total_n: int) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLS:
        if col not in feat.columns:
            continue
        n_miss = feat[col].isna().sum()
        rows.append({
            "feature":           col,
            "label":             FEATURE_LABELS.get(col, col),
            "n_total":           total_n,
            "n_missing":         int(n_miss),
            "pct_missing":       round(100 * n_miss / total_n, 1),
            "n_observed":        int(total_n - n_miss),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 7. AUDIT STEPS (decoding / data-quality verification)
# -----------------------------------------------------------------------------

VA_COLS_ALL = ["re_va_on_presentation", "le_va_on_presentation",
               "re_va_final",           "le_va_final"]


def audit_va_dates(rev: pd.DataFrame) -> pd.DataFrame:
    """
    Verify the REDCap day-of-month invariant for date-encoded VA values:
      Pattern A : day == 1 AND month == 6 AND year % 100 != 0
                  (year encodes denominator)
      Pattern B : day == 6 AND year == 2026 (month encodes denominator)
    Anything else is flagged and decoded as missing by decode_va().
    """
    import datetime as _dt
    columns = ["column", "value", "day", "month", "year", "pattern", "n"]
    rows = []
    for col in VA_COLS_ALL:
        if col not in rev.columns:
            continue
        ser = rev[col]
        for val, n in ser.value_counts(dropna=True).items():
            if isinstance(val, (pd.Timestamp, _dt.datetime, _dt.date)):
                d, m, y = val.day, val.month, val.year
                pattern_a = (d == 1 and m == 6 and y % 100 != 0)
                pattern_b = (d == 6 and y == 2026)
                if pattern_a:
                    pat = "A"
                elif pattern_b:
                    pat = "B"
                else:
                    pat = "VIOLATION"
                rows.append({"column": col, "value": str(val),
                             "day": d, "month": m, "year": y,
                             "pattern": pat, "n": int(n)})
    audit = pd.DataFrame(rows, columns=columns)
    if not audit.empty:
        audit = audit.sort_values(
            ["pattern", "column", "value"]).reset_index(drop=True)
    n_violations = int((audit["pattern"] == "VIOLATION").sum())
    log.info("VA date audit: %d distinct date values across %d VA columns; "
             "%d invariant violations.",
             len(audit), len([c for c in VA_COLS_ALL if c in rev.columns]),
             n_violations)
    if n_violations > 0:
        log.warning(
            "audit_va_dates found %d distinct VA date values that violate the "
            "documented Pattern A or Pattern B rules. See "
            "va_date_audit.csv for details.",
            n_violations)
    return audit


def audit_repeat_instances(rev: pd.DataFrame) -> pd.DataFrame:
    """
    For episodes with > 1 REDCap repeat instance (same patient + admission_date),
    compare populated-cell rates between instance == 1 and instance == max.
    If "max" instances systematically have *more* populated downstream-clinical
    fields than instance 1, this is evidence that later instances contain
    follow-up data rather than corrections to presentation-time data, which
    would mean flatten_episodes (which keeps the last instance) is importing
    follow-up information into presentation-time features.
    """
    grouped = rev.dropna(subset=["admission_date"]).groupby(
        ["rveeh_ur", "admission_date"], dropna=False)
    multi = grouped.filter(lambda g: g["redcap_repeat_instance"].nunique() > 1)
    if multi.empty:
        log.info("Repeat-instance audit: no episodes with > 1 instance.")
        return pd.DataFrame()

    audit_cols = [c for c in [
        "re_va_on_presentation", "le_va_on_presentation",
        "re_va_final", "le_va_final",
        "rapd_yn", "fundus_visible_yn", "diabetes_yn",
        "immune_suppressed_yn", "surgery_yn", "growths_yn", "microorganism",
        "intervention___v", "intervention___b",
        "final_visit_date",
    ] if c in rev.columns]

    first = multi.sort_values("redcap_repeat_instance").groupby(
        ["rveeh_ur", "admission_date"]).first()
    last  = multi.sort_values("redcap_repeat_instance").groupby(
        ["rveeh_ur", "admission_date"]).last()

    rows = []
    n_groups = len(first)
    for c in audit_cols:
        n_pop_first = int(first[c].notna().sum())
        n_pop_last  = int(last[c].notna().sum())
        rows.append({
            "field":             c,
            "n_groups":          n_groups,
            "n_populated_first": n_pop_first,
            "n_populated_last":  n_pop_last,
            "delta_last_minus_first": n_pop_last - n_pop_first,
            "pct_first":         round(100 * n_pop_first / n_groups, 1),
            "pct_last":          round(100 * n_pop_last  / n_groups, 1),
        })
    audit = pd.DataFrame(rows)
    log.info("Repeat-instance audit: %d episodes with > 1 instance.", n_groups)
    suspicious = audit[audit["delta_last_minus_first"]
                       > max(2, int(0.10 * n_groups))]
    if not suspicious.empty:
        log.warning(
            "Repeat-instance audit: %d field(s) substantially more populated "
            "in 'last' than 'first' instance; possible late-data contamination. "
            "See repeat_instance_audit.csv.", len(suspicious))
    return audit


def audit_precipitating_codes(rev: pd.DataFrame) -> pd.DataFrame:
    """
    List all precipitating_factor___N columns observed in the raw export, with
    non-zero rates. Verifies whether code 6 exists and is unused vs. exists and
    is non-trivial.
    """
    cols = sorted([c for c in rev.columns
                   if c.startswith("precipitating_factor___")])
    rows = []
    for c in cols:
        s = pd.to_numeric(rev[c], errors="coerce")
        rows.append({
            "column":   c,
            "code":     c.replace("precipitating_factor___", ""),
            "n_one":    int((s == 1).sum()),
            "n_zero":   int((s == 0).sum()),
            "n_other":  int(s.notna().sum() - (s.isin([0, 1])).sum()),
            "n_nan":    int(s.isna().sum()),
        })
    audit = pd.DataFrame(rows)
    log.info("Precipitating-factor audit: %d code columns found.", len(audit))
    if "precipitating_factor___6" in cols:
        n6 = int((pd.to_numeric(rev["precipitating_factor___6"],
                                errors="coerce") == 1).sum())
        log.info("precipitating_factor___6 present in export; n(=1) = %d.", n6)
    else:
        log.info("precipitating_factor___6 NOT present in export; "
                 "code 6 is unused; current feature set is complete.")
    return audit


def audit_organism_mapping(rev: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that every unique free-text `microorganism` value maps cleanly and
    uniquely to one organism_cat. For each distinct value the audit records:
      n: occurrences in the review arm
      organism_cat: category returned by normalise_organism()
      matched_pattern: ORGANISM_PATTERNS label that fired (or "" if none)
      n_patterns: how many ORGANISM_PATTERNS matched (>1 indicates polymicrobial;
                        for a SINGLE organism this should be 1. A single-
                        organism value with n_patterns>1 indicates overlapping,
                        non-specific regexes that should be tightened)
      is_default: fell through to the "other_bacteria" fallback
      is_nonspecific: routed to "unknown" via NONSPECIFIC_PATTERN

    Inspect is_default==True rows for recognisable organisms that were missed,
    and n_patterns>1 rows to confirm the worst-prognosis class was chosen.
    """
    columns = [
        "raw_value_lower", "n", "organism_cat", "matched_pattern",
        "n_patterns", "is_default", "is_nonspecific",
    ]
    if "microorganism" not in rev.columns:
        log.info("Organism-mapping audit: no 'microorganism' column.")
        return pd.DataFrame(columns=columns)

    vals = rev["microorganism"].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    vc = vals.str.lower().value_counts()

    rows = []
    for val, n in vc.items():
        matches = [lab for lab, pat in ORGANISM_PATTERNS
                   if re_mod.search(pat, val)]
        cat = normalise_organism(val)
        rows.append({
            "raw_value_lower": val,
            "n":               int(n),
            "organism_cat":    cat,
            "matched_pattern": matches[0] if matches else "",
            "n_patterns":      len(matches),
            "is_default":      (not matches) and cat == "other_bacteria",
            "is_nonspecific":  cat == "unknown",
        })
    audit = pd.DataFrame(rows, columns=columns)
    if not audit.empty:
        audit = (audit
                 .sort_values(["organism_cat", "n"], ascending=[True, False])
                 .reset_index(drop=True))

    n_default = int(audit["is_default"].sum())
    n_default_rows = int(audit.loc[audit["is_default"], "n"].sum())
    n_multi = int((audit["n_patterns"] > 1).sum())
    n_multi_rows = int(audit.loc[audit["n_patterns"] > 1, "n"].sum())
    log.info("Organism-mapping audit: %d unique values; %d categories.",
             len(audit), audit["organism_cat"].nunique())
    if n_default:
        log.warning(
            "Organism-mapping audit: %d unique value(s) (%d rows) hit the "
            "'other_bacteria' fallback. Review organism_mapping_audit.csv to "
            "confirm none are recognisable organisms.", n_default, n_default_rows)
    if n_multi:
        log.info(
            "Organism-mapping audit: %d unique value(s) (%d rows) are "
            "polymicrobial (matched >1 pattern); first-match assigned the "
            "worst-prognosis class.", n_multi, n_multi_rows)
    return audit


# -----------------------------------------------------------------------------
# 8. MAIN
# -----------------------------------------------------------------------------

def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(
        description="Preprocess Victorian Endophthalmitis REDCap data.")
    parser.add_argument(
        "--strict", action="store_true",
        help="Strict outcome threshold: poor = final logMAR > 1.0 (excludes 6/60). "
             "Outputs written to data_strict/ instead of data/.")
    args = parser.parse_args()

    if args.strict:
        OUT_DIR = Path("data_strict")
    OUT_DIR.mkdir(exist_ok=True)

    raw   = load_data()
    reg, rev = split_arms(raw)

    rev   = merge_gender(rev, reg)

    # Data-quality audits (run before flattening / feature build)
    va_date_audit       = audit_va_dates(rev)
    repeat_audit        = audit_repeat_instances(rev)
    precip_audit        = audit_precipitating_codes(rev)
    organism_audit      = audit_organism_mapping(rev)
    va_date_audit.to_csv(OUT_DIR / "va_date_audit.csv", index=False)
    if not repeat_audit.empty:
        repeat_audit.to_csv(OUT_DIR / "repeat_instance_audit.csv", index=False)
    precip_audit.to_csv(OUT_DIR / "precipitating_code_audit.csv", index=False)
    if not organism_audit.empty:
        organism_audit.to_csv(OUT_DIR / "organism_mapping_audit.csv", index=False)
    log.info("Saved: %s", OUT_DIR / "va_date_audit.csv")
    if not repeat_audit.empty:
        log.info("Saved: %s", OUT_DIR / "repeat_instance_audit.csv")
    log.info("Saved: %s", OUT_DIR / "precipitating_code_audit.csv")
    if not organism_audit.empty:
        log.info("Saved: %s", OUT_DIR / "organism_mapping_audit.csv")

    eps   = flatten_episodes(rev)
    feat  = build_features(eps, strict=args.strict)

    # Outcome summary
    outcome_known = feat["poor_outcome"].notna()
    n_total    = len(feat)
    n_outcome  = outcome_known.sum()
    n_poor     = (feat.loc[outcome_known, "poor_outcome"] == 1).sum()
    n_good     = (feat.loc[outcome_known, "poor_outcome"] == 0).sum()

    log.info("=" * 60)
    log.info("Total episodes:              %d", n_total)
    log.info("Episodes with known outcome: %d (%.1f%%)",
             n_outcome, 100 * n_outcome / n_total)
    threshold_label = "VA > 6/60 (strict)" if args.strict else "VA >= 6/60"
    log.info("  Poor outcome (%s):  %d (%.1f%%)",
             threshold_label, n_poor, 100 * n_poor / n_outcome)
    log.info("  Good outcome:                %d (%.1f%%)",
             n_good, 100 * n_good / n_outcome)
    log.info("=" * 60)

    # VA distribution (presenting)
    log.info("Presenting VA logMAR distribution:")
    log.info("  Median (IQR): %.2f (%.2f-%.2f)",
             feat["pres_logmar"].median(),
             feat["pres_logmar"].quantile(0.25),
             feat["pres_logmar"].quantile(0.75))

    # Save outputs
    # Select and save feature columns plus identifiers
    save_cols = (
        ["rveeh_ur", "admission_date", "affected_eye_clean",
         "analysis_eye", "analysis_eye_basis",
         "days_to_final_visit",
         "primary_vitrectomy", "primary_biopsy", "procedure_group"] +
        FEATURE_COLS
    )
    save_cols = [c for c in save_cols if c in feat.columns]

    feat[save_cols].to_csv(OUT_DIR / "processed_episodes.csv", index=False)
    log.info("Saved: %s", OUT_DIR / "processed_episodes.csv")

    # VA mapping reference table
    va_tbl = build_va_mapping_table()
    va_tbl.to_csv(OUT_DIR / "va_mapping_table.csv", index=False)
    log.info("Saved: %s", OUT_DIR / "va_mapping_table.csv")

    # Missingness report
    miss = missingness_report(feat, total_n=len(feat))
    miss.to_csv(OUT_DIR / "missingness_report.csv", index=False)
    log.info("Saved: %s", OUT_DIR / "missingness_report.csv")

    # Pretty-print missingness
    log.info("\nMissingness summary:")
    for _, row in miss.iterrows():
        log.info("  %-45s  %5.1f%% missing (%d/%d)",
                 row["label"], row["pct_missing"],
                 row["n_missing"], row["n_total"])

    return feat


if __name__ == "__main__":
    main()
