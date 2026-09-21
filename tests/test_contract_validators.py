from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_data_contract import DataContractValidator
from scripts.verify_model_contract import ModelContractValidator, prediction_problems


CATEGORIES = ["书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"]


def base_config() -> dict:
    return {
        "project": {"random_seed": 42},
        "data": {
            "raw_path": "data/raw/online_shopping_10_cats.csv",
            "train_ratio": 0.8,
            "dev_ratio": 0.1,
            "test_ratio": 0.1,
        },
        "task": {"categories": CATEGORIES},
    }


def valid_manifest() -> dict:
    return {
        "created_at": "2026-09-21T10:05:00+08:00",
        "random_seed": 42,
        "source": {"path": "data/raw/online_shopping_10_cats.csv", "sha256": "a" * 64},
        "cleaning": {
            "rules": {
                "empty_review": "drop",
                "duplicate_review": "drop_conflicts",
                "control_chars": "remove",
                "short_text_threshold": "none",
            },
            "rows_before": 62_774,
            "rows_after": 62_700,
        },
        "split": {
            "strategy": "stratified_by_cat",
            "ratios": {"train": 0.8, "dev": 0.1, "test": 0.1},
            "dedup_key": "normalized_text",
        },
        "leakage_check": {"based_on": "normalized_text"},
    }


class DataContractValidatorTests(unittest.TestCase):
    def test_manifest_fixed_rules_accept_valid_contract(self) -> None:
        validator = DataContractValidator(Path("unused"), base_config())
        validator.check_manifest_contract(valid_manifest())
        self.assertEqual(validator.checks[-1].status, "ok")

    def test_manifest_fixed_rules_reject_wrong_split(self) -> None:
        manifest = valid_manifest()
        manifest["random_seed"] = 7
        manifest["split"]["strategy"] = "random"
        manifest["split"]["ratios"] = {"train": 0.7, "dev": 0.2, "test": 0.1}
        validator = DataContractValidator(Path("unused"), base_config())
        validator.check_manifest_contract(manifest)
        self.assertEqual(validator.checks[-1].status, "fail")
        self.assertIn("random_seed", validator.checks[-1].detail)
        self.assertIn("split.ratios", validator.checks[-1].detail)

    def test_measured_stratification_rejects_skewed_split(self) -> None:
        columns = ["sample_id", "review", "cat", "label"]
        frames = {
            "train": pd.DataFrame(
                [[str(index), f"评论{index}", CATEGORIES[0], "1"] for index in range(10)],
                columns=columns,
            ),
            "dev": pd.DataFrame(columns=columns),
            "test": pd.DataFrame(columns=columns),
        }
        validator = DataContractValidator(Path("unused"), base_config())
        validator.check_stratification(frames)
        self.assertEqual(validator.checks[-1].status, "fail")


class ModelContractValidatorTests(unittest.TestCase):
    def test_prediction_rejects_nan_and_negative_probabilities(self) -> None:
        probabilities = {category: 0.0 for category in CATEGORIES}
        probabilities[CATEGORIES[0]] = math.nan
        probabilities[CATEGORIES[1]] = -0.1
        problems = prediction_problems(
            [probabilities, probabilities.copy()],
            [CATEGORIES[0], CATEGORIES[0]],
            CATEGORIES,
        )
        self.assertTrue(any("非法概率" in problem for problem in problems))

    def test_prediction_must_match_highest_probability(self) -> None:
        probabilities = {category: 0.0 for category in CATEGORIES}
        probabilities[CATEGORIES[0]] = 1.0
        problems = prediction_problems(
            [probabilities, probabilities.copy()],
            [CATEGORIES[1], CATEGORIES[0]],
            CATEGORIES,
        )
        self.assertTrue(any("最高概率类别" in problem for problem in problems))

    def test_inference_import_error_is_failure(self) -> None:
        validator = ModelContractValidator(Path("unused"), Path("unused"))
        with patch(
            "scripts.verify_model_contract.importlib.import_module",
            side_effect=RuntimeError("broken import"),
        ):
            validator.check_inference()
        self.assertEqual(validator.checks[-1].status, "fail")


if __name__ == "__main__":
    unittest.main()
