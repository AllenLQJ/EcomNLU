"""``src/web`` 是否满足 ``docs/contracts.md`` §3 的自动化检查。

全部用例都不依赖真实模型产物：错误路径在请求校验阶段就会返回，成功路径用
确定性假预测器替换 ``load_predictor`` 的返回值。
"""

from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.web import create_app  # noqa: E402
from src.web.config import load_web_config  # noqa: E402
from src.web.errors import ErrorCode, HTTP_STATUS  # noqa: E402


CONFIG = load_web_config()
CATEGORIES = CONFIG.categories
MAX_REVIEW_CHARS = CONFIG.max_review_chars


class FakePredictor:
    """确定性假预测器：把概率集中在第 2 个类别上，便于断言。"""

    def __init__(self) -> None:
        self.categories = list(CATEGORIES)

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        results = []
        for _ in texts:
            item = {name: 0.0 for name in self.categories}
            item[self.categories[0]] = 0.3
            item[self.categories[1]] = 0.7
            results.append(item)
        return results


class FakeSentimentPredictor:
    """确定性假情感预测器；标签集可覆盖，用于验证标签校验。"""

    def __init__(self, labels: tuple[str, ...] = ("负向", "正向")) -> None:
        self.categories = list(labels)

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        primary = self.categories[0]
        secondary = self.categories[1] if len(self.categories) > 1 else self.categories[0]
        return [{primary: 0.75, secondary: 0.25} for _ in texts]


@contextmanager
def client_with_fake_model():
    """构造一个模型已加载的客户端，不触碰真实模型产物。"""
    with patch("src.web.service.load_predictor", return_value=FakePredictor()):
        with TestClient(create_app(model_dir=Path("fake_model"))) as client:
            yield client


@contextmanager
def client_with_sentiment(labels: tuple[str, ...] = ("负向", "正向")):
    """主任务 + 情感任务都已加载的客户端。"""

    def fake_load(model_dir):
        if "sentiment" in Path(model_dir).name:
            return FakeSentimentPredictor(labels)
        return FakePredictor()

    with patch("src.web.service.load_predictor", side_effect=fake_load):
        with TestClient(
            create_app(
                model_dir=Path("fake_model"),
                sentiment_model_dir=Path("fake_sentiment_model"),
            )
        ) as client:
            yield client


class ErrorShapeMixin:
    """契约 §3.4：所有错误必须是 ``{"error": {"code", "message"}}``。"""

    def assert_error(self, response, code: str) -> None:  # type: ignore[no-untyped-def]
        self.assertEqual(response.status_code, HTTP_STATUS[code], response.text)
        body = response.json()
        self.assertEqual(set(body), {"error"}, body)
        self.assertEqual(set(body["error"]), {"code", "message"}, body)
        self.assertEqual(body["error"]["code"], code, body)
        self.assertTrue(body["error"]["message"], body)


class HealthTests(unittest.TestCase):
    def test_health_without_model_reports_degraded(self) -> None:
        with TestClient(create_app(model_dir=None)) as client:
            response = client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(
            set(body), {"status", "model_name", "model_loaded", "sentiment_enabled", "categories"}
        )
        self.assertEqual(body["status"], "degraded")
        self.assertFalse(body["model_loaded"])
        self.assertFalse(body["sentiment_enabled"])
        self.assertEqual(body["categories"], CATEGORIES)

    def test_health_with_model_reports_ok(self) -> None:
        with client_with_fake_model() as client:
            response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["model_name"], "fake_model")

    def test_categories_available_without_model(self) -> None:
        """契约 §3.1：模型未就绪时该端点仍须可用，类别取自配置而非模型。"""
        with TestClient(create_app(model_dir=None)) as client:
            response = client.get("/api/categories")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["categories"], CATEGORIES)


class ValidationTests(ErrorShapeMixin, unittest.TestCase):
    """请求校验发生在模型调用之前，因此不需要模型。"""

    def setUp(self) -> None:
        self._context = TestClient(create_app(model_dir=None))
        self.client = self._context.__enter__()

    def tearDown(self) -> None:
        self._context.__exit__(None, None, None)

    def post(self, **kwargs):  # type: ignore[no-untyped-def]
        return self.client.post("/api/predict", **kwargs)

    def test_malformed_json(self) -> None:
        response = self.post(content=b"{not json", headers={"Content-Type": "application/json"})
        self.assert_error(response, ErrorCode.INVALID_JSON)

    def test_body_is_not_an_object(self) -> None:
        self.assert_error(self.post(json=[1, 2, 3]), ErrorCode.INVALID_JSON)

    def test_missing_review(self) -> None:
        self.assert_error(self.post(json={}), ErrorCode.MISSING_REVIEW)

    def test_review_is_not_a_string(self) -> None:
        self.assert_error(self.post(json={"review": 123}), ErrorCode.MISSING_REVIEW)

    def test_empty_review(self) -> None:
        self.assert_error(self.post(json={"review": ""}), ErrorCode.EMPTY_REVIEW)

    def test_whitespace_only_review(self) -> None:
        self.assert_error(self.post(json={"review": "  \n\t "}), ErrorCode.EMPTY_REVIEW)

    def test_review_over_limit(self) -> None:
        response = self.post(json={"review": "测" * (MAX_REVIEW_CHARS + 1)})
        self.assert_error(response, ErrorCode.REVIEW_TOO_LONG)

    def test_review_at_limit_is_not_rejected(self) -> None:
        """长度恰好等于上限时不应报错（此时才会走到模型未加载）。"""
        response = self.post(json={"review": "测" * MAX_REVIEW_CHARS})
        self.assert_error(response, ErrorCode.MODEL_NOT_LOADED)

    def test_top_k_out_of_range(self) -> None:
        for value in (0, -1, CONFIG.max_top_k + 1):
            with self.subTest(top_k=value):
                self.assert_error(
                    self.post(json={"review": "手机很好用", "top_k": value}),
                    ErrorCode.INVALID_TOP_K,
                )

    def test_top_k_wrong_type(self) -> None:
        for value in ("abc", True, 1.0, None):
            with self.subTest(top_k=value):
                self.assert_error(
                    self.post(json={"review": "手机很好用", "top_k": value}),
                    ErrorCode.INVALID_TOP_K,
                )

    def test_predict_without_model(self) -> None:
        self.assert_error(
            self.post(json={"review": "手机很好用"}), ErrorCode.MODEL_NOT_LOADED
        )


class PredictTests(unittest.TestCase):
    def test_success_response_shape(self) -> None:
        with client_with_fake_model() as client:
            response = client.post("/api/predict", json={"review": "手机很好用，电池耐用"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(
            set(body), {"category", "top_k", "sentiment", "elapsed_ms", "model_name"}, body
        )
        self.assertEqual(set(body["category"]), {"label", "confidence"})
        self.assertIn(body["category"]["label"], CATEGORIES)
        self.assertEqual(body["model_name"], "fake_model")
        self.assertGreater(body["elapsed_ms"], 0)

    def test_sentiment_field_is_present_and_null_without_a_sentiment_model(self) -> None:
        """契约 §3.3：未配置情感模型时该字段为 null，但**不得省略**。"""
        with client_with_fake_model() as client:
            body = client.post("/api/predict", json={"review": "手机很好用"}).json()
        self.assertIn("sentiment", body)
        self.assertIsNone(body["sentiment"])

    def test_default_top_k_and_ordering(self) -> None:
        with client_with_fake_model() as client:
            body = client.post("/api/predict", json={"review": "手机很好用"}).json()
        self.assertEqual(len(body["top_k"]), CONFIG.default_top_k)
        probabilities = [item["probability"] for item in body["top_k"]]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertEqual(body["category"]["label"], body["top_k"][0]["label"])
        self.assertEqual(set(body["top_k"][0]), {"label", "probability"})

    def test_requested_top_k_is_respected(self) -> None:
        with client_with_fake_model() as client:
            body = client.post(
                "/api/predict", json={"review": "手机很好用", "top_k": 5}
            ).json()
        self.assertEqual(len(body["top_k"]), 5)

    def test_top_k_upper_bound_is_accepted(self) -> None:
        with client_with_fake_model() as client:
            response = client.post(
                "/api/predict", json={"review": "手机很好用", "top_k": CONFIG.max_top_k}
            )
        self.assertEqual(response.status_code, 200, response.text)


class SentimentTests(unittest.TestCase):
    """M5 情感任务接入后，契约 §3.3 的 ``sentiment`` 字段行为。"""

    def test_health_reports_sentiment_enabled(self) -> None:
        with client_with_sentiment() as client:
            response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["sentiment_enabled"])

    def test_health_reports_sentiment_disabled_without_the_model(self) -> None:
        with client_with_fake_model() as client:
            body = client.get("/api/health").json()
        self.assertFalse(body["sentiment_enabled"])

    def test_predict_returns_a_contract_shaped_sentiment_object(self) -> None:
        with client_with_sentiment() as client:
            body = client.post("/api/predict", json={"review": "质量太差了"}).json()
        sentiment = body["sentiment"]
        self.assertIsNotNone(sentiment)
        self.assertEqual(set(sentiment), {"label", "confidence"})
        self.assertIn(sentiment["label"], ("负向", "正向"))
        self.assertGreaterEqual(sentiment["confidence"], 0.0)
        self.assertLessEqual(sentiment["confidence"], 1.0)

    def test_sentiment_label_is_the_highest_probability_one(self) -> None:
        """假预测器把 0.75 给了 ``负向``，应取最高概率项。"""
        with client_with_sentiment() as client:
            sentiment = client.post("/api/predict", json={"review": "很好"}).json()["sentiment"]
        self.assertEqual(sentiment["label"], "负向")

    def test_wrong_sentiment_label_set_is_rejected(self) -> None:
        """标签集不符契约 v1.2 §2.1 时必须拒绝启用，而不是输出错位的标签。"""
        with client_with_sentiment(labels=("负面", "正面")) as client:
            self.assertFalse(client.get("/api/health").json()["sentiment_enabled"])
            body = client.post("/api/predict", json={"review": "很好"}).json()
        self.assertIsNone(body["sentiment"])

    def test_sentiment_does_not_change_the_category_result(self) -> None:
        """接入情感不得影响类别预测与 top_k 行为。"""
        with client_with_fake_model() as plain, client_with_sentiment() as enriched:
            without = plain.post("/api/predict", json={"review": "手机很好用", "top_k": 4}).json()
            with_sentiment = enriched.post(
                "/api/predict", json={"review": "手机很好用", "top_k": 4}
            ).json()
        self.assertEqual(without["category"], with_sentiment["category"])
        self.assertEqual(without["top_k"], with_sentiment["top_k"])
        self.assertIsNone(without["sentiment"])
        self.assertIsNotNone(with_sentiment["sentiment"])


if __name__ == "__main__":
    unittest.main()
