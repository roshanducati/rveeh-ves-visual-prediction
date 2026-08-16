"""Data-free regression tests for the public analysis code."""

from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


preprocess = load_script("preprocess", "01_preprocess.py")
train = load_script("train", "02_train.py")
evaluate = load_script("evaluate", "03_evaluate.py")
missingness_sensitivity = load_script(
    "missingness_sensitivity", "05_sensitivity_analysis.py"
)
followup_sensitivity = load_script(
    "followup_sensitivity", "07_followup_sensitivity.py"
)
timing = load_script("timing", "04_timing_analysis.py")


def synthetic_episode(final_va: str) -> dict:
    return {
        "rveeh_ur": "synthetic",
        "admission_date": "2026-01-01",
        "affected_eye": "RE",
        "re_va_on_presentation": "6/60",
        "le_va_on_presentation": "6/6",
        "re_va_final": final_va,
        "le_va_final": "6/6",
        "admission_age": 70,
        "gender_female": 1.0,
        "fundus_visible_yn": 0,
        "rapd_yn": 1,
        "diabetes_yn": 0,
        "immune_suppressed_yn": np.nan,
        "surgery_yn": 1,
        "precipitating_factor___1": 1,
        "precipitating_factor___2": 0,
        "precipitating_factor___3": 0,
        "precipitating_factor___4": 0,
        "precipitating_factor___5": 0,
        "precipitating_factor___7": 0,
        "precipitating_factor___98": 0,
        "growths_yn": 0,
        "microorganism": "",
        "intervention___v": 0,
        "intervention___b": 1,
        "final_visit_date": "2026-02-01",
    }


class PreprocessingTests(unittest.TestCase):
    def test_visual_acuity_decoding(self):
        self.assertAlmostEqual(preprocess.decode_va("6/60"), 1.0)
        self.assertAlmostEqual(preprocess.decode_va(datetime(1960, 6, 1)), 1.0)
        self.assertAlmostEqual(
            preprocess.decode_va(datetime(2026, 9, 6)), np.log10(9 / 6)
        )
        self.assertEqual(preprocess.decode_va("CF"), 1.85)
        self.assertEqual(preprocess.decode_va("EVIS"), 4.0)
        self.assertTrue(np.isnan(preprocess.decode_va(datetime(2025, 9, 6))))
        self.assertTrue(np.isnan(preprocess.decode_va(datetime(2000, 6, 1))))

    def test_worse_eye_rule(self):
        row = {
            "affected_eye": "BE",
            "re_va_final": "6/12",
            "le_va_final": "6/60",
        }
        self.assertAlmostEqual(
            preprocess.extract_eye_va(row, preprocess.VA_FINAL_COLS), 1.0
        )

    def test_selected_presentation_eye_is_followed_to_final_visit(self):
        episode = synthetic_episode("6/75")
        episode.update({
            "affected_eye": "BE",
            "re_va_on_presentation": "6/60",
            "le_va_on_presentation": "6/12",
            "re_va_final": "6/75",
            "le_va_final": "EVIS",
        })
        result = preprocess.build_features(pd.DataFrame([episode]), strict=True)
        self.assertEqual(result.loc[0, "analysis_eye"], "RE")
        self.assertEqual(result.loc[0, "analysis_eye_basis"], "worse_presenting_va")
        self.assertAlmostEqual(result.loc[0, "final_logmar"], np.log10(75 / 6))

    def test_visual_acuity_date_audit_enforces_complete_patterns(self):
        report = preprocess.audit_va_dates(pd.DataFrame({
            "re_va_on_presentation": [
                datetime(1960, 6, 1),
                datetime(2026, 9, 6),
                datetime(2025, 9, 6),
                datetime(2000, 6, 1),
            ]
        }))
        self.assertEqual((report["pattern"] == "VIOLATION").sum(), 2)

    def test_organism_mapping(self):
        self.assertEqual(
            preprocess.normalise_organism("Pseudomonas aeruginosa"),
            "gram_negative",
        )
        self.assertEqual(
            preprocess.normalise_organism("coagulase negative Staphylococcus"),
            "cons",
        )
        self.assertEqual(preprocess.normalise_organism(""), "unknown")

    def test_submitted_outcome_boundary(self):
        episodes = pd.DataFrame(
            [
                synthetic_episode("6/60"),
                synthetic_episode("6/75"),
                synthetic_episode("EVIS"),
            ]
        )
        result = preprocess.build_features(episodes, strict=True)
        self.assertEqual(result["poor_outcome"].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(result["organism_cat"].tolist(), ["no_growth"] * 3)

    def test_empty_audits_return_schema_correct_reports(self):
        va_report = preprocess.audit_va_dates(pd.DataFrame())
        self.assertTrue(va_report.empty)
        self.assertEqual(
            va_report.columns.tolist(),
            ["column", "value", "day", "month", "year", "pattern", "n"],
        )

        organism_report = preprocess.audit_organism_mapping(
            pd.DataFrame({"microorganism": [np.nan, ""]})
        )
        self.assertTrue(organism_report.empty)
        self.assertEqual(
            organism_report.columns.tolist(),
            [
                "raw_value_lower", "n", "organism_cat", "matched_pattern",
                "n_patterns", "is_default", "is_nonspecific",
            ],
        )


class ModellingContractTests(unittest.TestCase):
    def test_reported_predictor_sets(self):
        full = train.NUMERIC_FEATURES + train.CATEGORICAL_FEATURES
        pre_culture = (
            train.NUMERIC_FEATURES_NOMICRO + train.CATEGORICAL_FEATURES_NOMICRO
        )
        self.assertEqual(len(full), 16)
        self.assertEqual(len(pre_culture), 14)
        self.assertNotIn("procedure_group", full)
        self.assertNotIn("primary_vitrectomy", full)
        self.assertNotIn("days_to_final_visit", full)
        self.assertNotIn("culture_positive", pre_culture)
        self.assertNotIn("organism_cat", pre_culture)

    def test_fixed_random_values(self):
        self.assertEqual(train.SPLIT_SEED, 61)
        self.assertEqual(train.ESTIMATOR_SEED, 42)

    def test_iterative_imputation_produces_one_completed_dataset(self):
        modules = [train, missingness_sensitivity, followup_sensitivity]
        for module in modules:
            with self.subTest(module=module.__name__):
                preprocessor = module.make_preprocessor(
                    module.NUMERIC_FEATURES, module.CATEGORICAL_FEATURES
                )
                numeric_pipeline = preprocessor.transformers[0][1]
                self.assertIn("iterative_imputer", numeric_pipeline.named_steps)
                imputer = numeric_pipeline.named_steps["iterative_imputer"]
                self.assertEqual(imputer.max_iter, 10)
                self.assertFalse(imputer.sample_posterior)
                self.assertEqual(type(imputer.estimator).__name__, "BayesianRidge")

    def test_manuscript_threshold_grid(self):
        y = np.array([0, 0, 1, 1])
        probs = {"XGB": np.array([0.1, 0.4, 0.6, 0.9])}
        metrics = pd.DataFrame([{"model": "XGB", "test_auroc": 1.0}])
        table = evaluate.make_table4(y, probs, metrics, lead_name="XGB")
        self.assertEqual(len(table), 16)
        self.assertAlmostEqual(table.iloc[0]["Threshold"], 0.15)
        self.assertAlmostEqual(table.iloc[-1]["Threshold"], 0.90)

    def test_table1_includes_submitted_univariable_tests(self):
        outcome = np.repeat([0.0, 1.0], 10)
        data = {
            "poor_outcome": outcome,
            "days_to_final_visit": np.arange(20, dtype=float) + 1,
            "age_years": np.r_[np.arange(40, 50), np.arange(70, 80)].astype(float),
            "pres_logmar": np.r_[np.linspace(0.1, 0.5, 10),
                                    np.linspace(1.2, 2.1, 10)],
        }
        binary_columns = [
            "gender_female", "pres_poor_va", "fundus_visible",
            "rapd_present", "diabetes", "immune_suppressed",
            "prior_surgery", "etiol_cataract_sx", "etiol_ivi",
            "etiol_glaucoma_sx", "etiol_trauma", "etiol_corneal_ulcer",
            "etiol_endogenous", "etiol_other", "culture_positive",
        ]
        for index, column in enumerate(binary_columns):
            data[column] = (np.arange(20) + index) % 2
        data["age_years"][0] = np.nan
        data["immune_suppressed"] = np.r_[
            np.zeros(9), np.nan, np.zeros(9), 1.0
        ]

        table = evaluate.make_table1(pd.DataFrame(data))
        comparison = "Univariate statistical comparison"
        self.assertEqual(table.columns[-1], comparison)
        self.assertEqual(
            table.loc[table["Characteristic"] == "N", comparison].iloc[0],
            "",
        )
        self.assertEqual(
            table.loc[
                table["Characteristic"] == "Age, years, median (IQR)",
                comparison,
            ].iloc[0],
            "<0.001",
        )
        age_overall = table.loc[
            table["Characteristic"] == "Age, years, median (IQR)",
            "Overall",
        ].iloc[0]
        self.assertIn("[n=19]", age_overall)
        sparse_p = table.loc[
            table["Characteristic"] == "Immune suppression, n (%)",
            comparison,
        ].iloc[0]
        self.assertNotEqual(sparse_p, "")

    def test_pr_figure_uses_table2_auprc_values(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = pd.DataFrame([{
            "model": "XGB",
            "test_auprc": 0.8123,
            "auprc_lower_95": 0.7012,
            "auprc_upper_95": 0.9012,
        }])
        reported = evaluate.reported_auprc(
            y, scores, metrics, "XGB"
        )
        self.assertEqual(reported, (0.8123, 0.7012, 0.9012))

    def test_followup_plausibility_window_matches_preprocessing(self):
        values = pd.Series([-1.0, 0.0, 3650.0, 3651.0])
        cleaned = timing.clean_followup_interval(values)
        self.assertTrue(np.isnan(cleaned.iloc[0]))
        self.assertEqual(cleaned.iloc[1], 0.0)
        self.assertEqual(cleaned.iloc[2], 3650.0)
        self.assertTrue(np.isnan(cleaned.iloc[3]))


if __name__ == "__main__":
    unittest.main()
