from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.contract import PredictorError, load_label_mapping
from src.inference.fasttext_adapter import FastTextCategoryPredictor
from src.inference.transformers_adapter import TransformersCategoryPredictor


CATEGORIES = ["书籍", "平板", "手机"]


def valid_mapping() -> dict:
    return {
        "categories": CATEGORIES,
        "label2id": {name: index for index, name in enumerate(CATEGORIES)},
        "id2label": {str(index): name for index, name in enumerate(CATEGORIES)},
    }


class LabelMappingTests(unittest.TestCase):
    def write_mapping(self, directory: Path, mapping: dict) -> None:
        (directory / "label_mapping.json").write_text(
            json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
        )

    def test_valid_mapping_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_mapping(directory, valid_mapping())
            self.assertEqual(load_label_mapping(directory), valid_mapping())

    def test_label2id_must_follow_category_order(self) -> None:
        mapping = valid_mapping()
        mapping["label2id"] = {"书籍": 1, "平板": 0, "手机": 2}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_mapping(directory, mapping)
            with self.assertRaisesRegex(PredictorError, "label2id"):
                load_label_mapping(directory)

    def test_id2label_must_follow_category_order(self) -> None:
        mapping = valid_mapping()
        mapping["id2label"] = {"0": "平板", "1": "书籍", "2": "手机"}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_mapping(directory, mapping)
            with self.assertRaisesRegex(PredictorError, "id2label"):
                load_label_mapping(directory)


class FastTextPredictionTests(unittest.TestCase):
    def predictor_with_output(self, labels: list[str], scores: list[float]):
        predictor = FastTextCategoryPredictor.__new__(FastTextCategoryPredictor)
        predictor.categories = list(CATEGORIES)
        predictor.model = Mock()
        predictor.model.predict.return_value = (labels, scores)
        return predictor

    def test_valid_output_is_returned_in_contract_order(self) -> None:
        predictor = self.predictor_with_output(
            ["__label__平板", "__label__书籍", "__label__手机"],
            [0.6, 0.3, 0.1],
        )
        result = predictor.predict_proba(["屏幕不错"])[0]
        self.assertEqual(list(result), CATEGORIES)
        self.assertAlmostEqual(sum(result.values()), 1.0)
        # 适配器会对概率按总和等比归一（见 fasttext_adapter.predict_proba 的说明），
        # 而 0.6 / (0.6 + 0.3 + 0.1) 在 IEEE-754 下等于 0.6000000000000001，
        # 用 ``assertEqual`` 比较会因末位舍入而失败。断言语义仍是「平板 ≈ 0.6」，
        # 故改用近似比较，不改变被测行为。
        self.assertAlmostEqual(result["平板"], 0.6)

    def test_unknown_label_is_rejected(self) -> None:
        predictor = self.predictor_with_output(
            ["__label__平板", "__label__书籍", "__label__未知"],
            [0.6, 0.3, 0.1],
        )
        with self.assertRaisesRegex(PredictorError, "未知标签"):
            predictor.predict_proba(["屏幕不错"])

    def test_duplicate_label_is_rejected(self) -> None:
        predictor = self.predictor_with_output(
            ["__label__平板", "__label__平板", "__label__手机"],
            [0.6, 0.3, 0.1],
        )
        with self.assertRaisesRegex(PredictorError, "重复"):
            predictor.predict_proba(["屏幕不错"])

    def test_invalid_probability_is_rejected(self) -> None:
        predictor = self.predictor_with_output(
            ["__label__平板", "__label__书籍", "__label__手机"],
            [float("nan"), 0.3, 0.7],
        )
        with self.assertRaisesRegex(PredictorError, r"\[0, 1\]"):
            predictor.predict_proba(["屏幕不错"])

    def test_probability_sum_is_checked(self) -> None:
        predictor = self.predictor_with_output(
            ["__label__平板", "__label__书籍", "__label__手机"],
            [0.4, 0.3, 0.1],
        )
        with self.assertRaisesRegex(PredictorError, "概率和"):
            predictor.predict_proba(["屏幕不错"])


class TransformersConfigurationTests(unittest.TestCase):
    def test_batch_size_must_be_positive_integer(self) -> None:
        for value in (0, -1, True, 1.5, "32"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(PredictorError, "batch_size"):
                    TransformersCategoryPredictor(Path("unused"), batch_size=value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
