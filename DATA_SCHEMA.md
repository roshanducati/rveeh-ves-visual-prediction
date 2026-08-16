# Data schema

This repository contains no patient data. An approved user can run the pipeline
by creating `data/raw/` locally and placing the REDCap export workbook at:

```
data/raw/VictorianEndophthalm_DATA_2026-05-17_2131.xlsx
```

If your export uses a different filename, update the `RAW_FILE` constant at the
top of `01_preprocess.py` and `04_timing_analysis.py`.

## Export structure

The export is a single REDCap workbook with two event arms in the
`redcap_event_name` column:

- a **registration arm** (one row per patient), and
- a **review arm** (`review_arm_1`; one or more episode records per patient,
  distinguished by `redcap_repeat_instance` and admission date).

`01_preprocess.py` flattens the review arm to one row per episode, keying on
`rveeh_ur` (patient identifier) and `admission_date`. Within duplicated keys,
records are sorted by repeat instance and the most recent non-missing value in
each field is retained. Distinct admission dates for the same patient are
treated as separate episodes.

## Fields consumed by the pipeline

The preprocessing script reads the following registry fields (consult the study
codebook for the exact REDCap field names and coded values; the names below
describe the role each field plays in the analysis):

| Role | Description |
|------|-------------|
| Patient identifier | `rveeh_ur`: used only for patient-level grouping in the train/test split and cross-validation. Never used as a predictor. |
| Admission date | `admission_date`: episode key and the index date for timing intervals. |
| Repeat instance | `redcap_repeat_instance`: selects the most recent review record per episode. |
| Affected eye | Laterality of the affected eye. For bilateral or unspecified laterality, the worse eye at presentation is selected and the same anatomical eye is followed to final review. Tied or unavailable presenting acuity uses the documented fallback described in `01_preprocess.py`. |
| Presenting visual acuity | Snellen reference code (numeric or non-numeric category), decoded to logMAR. |
| Final visual acuity | As above, taken at the most recent documented visit; defines the outcome. |
| Demographics | Age at presentation; sex. |
| Clinical signs | Fundus visibility; relative afferent pupillary defect. |
| Comorbidities | Diabetes mellitus; immune suppression. |
| Aetiology | Precipitating-event field, mapped to seven non-exclusive flags (post-cataract/IOL surgery, post-glaucoma/drainage surgery, penetrating eye injury, corneal ulcer, metastatic/endogenous, post-intravitreal injection, other ocular procedure). |
| Microbiology | Documented culture positivity and organism free text. The numeric indicator is 1 for documented positivity and 0 when positivity is not documented. The normalised organism category distinguishes confirmed no growth (`no_growth`) from unavailable status (`unknown`), in addition to the nine organism groups. |
| Procedure / dates | Initial procedure and intervention/discharge dates are used for the timing analysis and the procedure-balance check only; the procedure type is **excluded** as a predictor. |

## Visual acuity decoding

Snellen values (including any stored in a date-encoded form by the data-capture
system) are decoded to logMAR via a fixed, audited mapping verified to reproduce
every distinct recorded value. Non-Snellen categories are mapped as:

| Category | logMAR |
|----------|--------|
| Counting fingers (CF) | 1.85 |
| Hand movements (HM) | 2.30 |
| Light perception (LP) | 2.70 |
| No light perception (NLP) | 3.00 |
| Loss of the eye (evisceration/enucleation) | 4.00 |

`01_preprocess.py` writes an audit of the full acuity mapping
(`va_mapping_table.csv`) and several data-quality audits
(`va_date_audit.csv`, `repeat_instance_audit.csv`,
`precipitating_code_audit.csv`, `organism_mapping_audit.csv`) so that the
decoding and feature derivation can be independently checked.

Date-encoded acuities must match one of two complete patterns. Pattern A is
`YYYY-06-01`, with the final two year digits encoding the Snellen denominator.
Pattern B is `2026-MM-06`, with the month encoding the denominator. Values that
do not satisfy either complete pattern are reported by the audit and decoded as
missing.

## Outcome definition

The submitted primary outcome is a **poor visual outcome**: final logMAR > 1.00 (strictly
worse than 6/60) or loss of the eye. A final acuity of exactly 6/60 (logMAR
1.00) is classified as a good outcome. This threshold is selected with the
`--strict` flag, which routes outputs to `data_strict/` and `results_strict*/`.

All preprocessing outputs are local derived data. They are excluded from version
control and are not a component public repository.
