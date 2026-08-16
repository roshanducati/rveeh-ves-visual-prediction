# `01_preprocess.py`

This script converts the private REDCap workbook into one analysis row per
endophthalmitis episode. No imputation is performed here.

It:

1. separates registration and review arms;
2. merges sex from registration records;
3. consolidates duplicated patient/admission keys using the most recent
   non-missing value in each field after repeat-instance sorting;
4. decodes the audited Snellen representation and non-Snellen categories to
   logMAR;
5. selects the worse eye at presentation for bilateral or unspecified
   laterality and follows that anatomical eye to the final visit;
6. derives the submitted outcome (`final_logmar > 1.00` or eye loss) under
   `--strict`;
7. derives the 16 model predictors, procedure descriptors, and follow-up
   duration; and
8. writes mapping, missingness, repeat-instance, aetiology-code, and organism
   audits.

The seven aetiology flags are non-exclusive. Organism free text is mapped into
nine organism groups plus `no_growth` and `unknown`. A named organism with a
blank culture-status field is retained and treated as culture-positive, matching
the performed analysis.

The numeric culture indicator means documented positive versus not documented
positive. Confirmed no growth and unavailable status are distinguished by the
paired organism-category levels `no_growth` and `unknown`, respectively.

Outputs under `data_strict/` are derived patient-level data and are git-ignored.
