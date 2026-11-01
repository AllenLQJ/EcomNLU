"""``POST /api/v1/nlu/analyze``（契约 v1.3 §3.6）。全部使用确定性假预测器，不依赖真实模型产物。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.web import create_app  # noqa: E402
from src.web.config import load_web_config  # noqa: E402

CONFIG = load_web_config()
CATEGORIES = CONFIG.categories


class TablePredictor:
    """按关键词返回预设分布的假主任务模型；没命中关键词时均匀分布（低置信）。"""

    def __init__(self, table: dict[str, tuple[str, float]]) -> None:
        self.categories = list(CATEGORIES)
        self.table = table
        self.calls: list[str] = []

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        out = []
        for text in texts:
            self.calls.append(text)
            for key, (label, probability) in self.table.items():
                if key in text:
                    rest = (1 - probability) / (len(self.categories) - 1)
                    out.append({name: (probability if name == label else rest) for name in self.categories})
                    break
            else:
                out.append({name: 1 / len(self.categories) for name in self.categories})
        return out


class SentimentTable:
    def __init__(self, negative_words: tuple[str, ...] = ("漏水", "坏了", "差")) -> None:
        self.categories = ["负向", "正向"]
        self.negative_words = negative_words

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        return [{"负向": 0.97, "正向": 0.03} if any(w in t for w in self.negative_words) else {"负向": 0.1, "正向": 0.9}
                for t in texts]


TABLE = {"热水器": ("热水器", 0.93), "牛奶": ("蒙牛", 0.91), "酒店": ("酒店", 0.88), "苹果": ("水果", 0.95),
         "平板": ("平板", 0.5)}


@contextmanager
def make_client(table=None, sentiment=True, web_config=None):
    predictor = TablePredictor(table or TABLE)
    loaded = {"primary": predictor, "sentiment": SentimentTable()}

    def fake_load(path):
        return loaded["sentiment"] if "sentiment" in str(path) else loaded["primary"]

    with patch("src.web.service.load_predictor", side_effect=fake_load):
        app = create_app(model_dir=Path("fake_model"), sentiment_model_dir=Path("fake_sentiment") if sentiment else None,
                         web_config=web_config)
        with TestClient(app) as client:
            client.predictor = predictor
            yield client


def post(client, text, scene="AFTER_SALE_FEEDBACK"):
    return client.post("/api/v1/nlu/analyze", json={"text": text, "scene_type": scene})


class AnalyzeTests(unittest.TestCase):
    def test_single_clause_category_mapping_and_negative_priority(self):
        with make_client() as client:
            body = post(client, "热水器漏水了", "aftersales").json()
        self.assertEqual(body["clauses"], [{
            "text": "热水器漏水了", "predicted_label": "热水器", "category_confidence": 0.93, "category_code": "WATER_HEATER",
            "catalog_category_hint": "HOME_APPLIANCE_HEATER", "brand_hint": None, "item_type": "PRODUCT"}])
        self.assertEqual((body["sentiment"], body["sentiment_score"], body["priority_boost"]), ("NEGATIVE", 0.97, True))
        self.assertIs(body["scene_not_applicable"], False)
        self.assertEqual(body["model_name"], "fake_model")

    def test_multi_clause_brand_service_and_category_without_catalog_id(self):
        with make_client() as client:
            body = post(client, "牛奶要一箱，酒店订一晚，苹果也来点", "shopping").json()
        rows = [(c["predicted_label"], c["category_code"], c["catalog_category_hint"], c["brand_hint"], c["item_type"])
                for c in body["clauses"]]
        self.assertEqual(rows, [("蒙牛", "DAIRY", "DAIRY_MILK", "蒙牛", "PRODUCT"),
                                ("酒店", "HOTEL", "TRAVEL_HOTEL", None, "SERVICE"),
                                ("水果", "FRUIT", None, None, "PRODUCT")])      # 水果没有目录品类：有编码、无 hint
        self.assertFalse(body["priority_boost"])                                  # 导购场景不加急

    def test_low_confidence_clause_abstains_without_any_hint(self):
        with make_client() as client:
            clause = post(client, "平板还行").json()["clauses"][0]               # 模型只有 0.5 置信
        self.assertEqual(clause["predicted_label"], "ABSTAIN")
        self.assertEqual(clause["category_confidence"], 0.5)
        self.assertIsNone(clause["category_code"])
        self.assertIsNone(clause["catalog_category_hint"])

    def test_weak_negative_does_not_boost_priority(self):
        class WeakNegative(SentimentTable):
            def predict_proba(self, texts):
                return [{"负向": 0.8, "正向": 0.2} for _ in texts]
        predictor = TablePredictor(TABLE)
        with patch("src.web.service.load_predictor",
                   side_effect=lambda p: WeakNegative() if "sentiment" in str(p) else predictor):
            with TestClient(create_app(model_dir=Path("fake_model"), sentiment_model_dir=Path("fake_sentiment"))) as client:
                body = post(client, "热水器漏水了", "aftersales").json()
        self.assertEqual((body["sentiment"], body["sentiment_score"], body["priority_boost"]), ("NEGATIVE", 0.8, False))

    def test_without_sentiment_model_is_neutral_not_guessed(self):
        with make_client(sentiment=False) as client:
            body = post(client, "热水器漏水了").json()
        self.assertEqual((body["sentiment"], body["sentiment_score"], body["priority_boost"]), ("NEUTRAL", 0.0, False))

    def test_not_applicable_scenes_never_touch_the_model(self):
        with make_client() as client:
            for scene in ("PDF_BODY", "CS_APOLOGY"):
                body = post(client, "非常抱歉给您带来不便", scene).json()
                self.assertTrue(body["scene_not_applicable"])
                self.assertEqual(body["clauses"], [])
            self.assertEqual(client.predictor.calls, [])

    def test_single_clause_reuses_whole_text_prediction(self):
        with make_client() as client:
            post(client, "热水器漏水了")
            self.assertEqual(len(client.predictor.calls), 1)                      # 整段 == 唯一子句，不重复推理

    def test_clause_limit(self):
        with make_client() as client:
            body = post(client, "热水器，牛奶，酒店，苹果，平板，苹果，苹果").json()
        self.assertEqual(len(body["clauses"]), CONFIG.analyze.max_clauses)

    def test_validation_errors_use_the_standard_error_shape(self):
        with make_client() as client:
            cases = [
                ({"scene_type": "x"}, "MISSING_TEXT"),
                ({"text": 123}, "MISSING_TEXT"),
                ({"text": "   "}, "EMPTY_TEXT"),
                ({"text": "热" * (CONFIG.analyze.max_text_chars + 1)}, "TEXT_TOO_LONG"),
            ]
            for payload, code in cases:
                response = client.post("/api/v1/nlu/analyze", json=payload)
                self.assertEqual((response.status_code, response.json()["error"]["code"]), (400, code), payload)
            malformed = client.post("/api/v1/nlu/analyze", content="{oops", headers={"content-type": "application/json"})
            self.assertEqual((malformed.status_code, malformed.json()["error"]["code"]), (400, "INVALID_JSON"))

    def test_model_not_loaded_is_503_but_not_applicable_scene_still_answers(self):
        with TestClient(create_app()) as client:                                  # 没指定模型目录
            self.assertEqual(post(client, "热水器漏水了").status_code, 503)
            self.assertEqual(post(client, "热水器漏水了").json()["error"]["code"], "MODEL_NOT_LOADED")
            self.assertTrue(post(client, "致歉", "CS_APOLOGY").json()["scene_not_applicable"])


class AnalyzeConfigTests(unittest.TestCase):
    def _load(self, mutate):
        raw = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
        mutate(raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "c.yaml"
            path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
            return load_web_config(path)

    def test_repo_config_maps_every_category_to_a_code(self):
        self.assertEqual(set(CONFIG.analyze.categories), set(CATEGORIES))
        codes = [entry["category_code"] for entry in CONFIG.analyze.categories.values()]
        self.assertEqual(len(codes), len(set(codes)))

    def test_missing_analyze_section_falls_back_to_defaults(self):
        config = self._load(lambda raw: raw.pop("analyze"))
        self.assertEqual(config.analyze.confidence_threshold, 0.65)
        self.assertEqual(config.analyze.categories, {})

    def test_bad_values_are_rejected_with_the_offending_key(self):
        cases = [
            (lambda r: r["analyze"].update(confidence_threshold=1.5), "confidence_threshold"),
            (lambda r: r["analyze"].update(max_clauses=0), "max_clauses"),
            (lambda r: r["analyze"].update(priority_boost_min_score=2), "priority_boost_min_score"),
            (lambda r: r["analyze"].update(not_applicable_scenes="PDF_BODY"), "not_applicable_scenes"),
            (lambda r: r["analyze"]["categories"].update({"不存在的类别": {"category_code": "X"}}), "未知类别"),
            (lambda r: r["analyze"]["categories"]["手机"].pop("category_code"), "category_code"),
        ]
        for mutate, needle in cases:
            with self.assertRaises(ValueError) as caught:
                self._load(mutate)
            self.assertIn(needle, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
