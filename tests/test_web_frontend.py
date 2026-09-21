"""首页与静态资源的守卫测试（说明书 §9.5、§10.1）。

只做结构与约束检查：页面由 FastAPI 同源托管、不依赖外部资源、只调用契约端点，
且输入长度上限与 ``configs/base.yaml`` 保持一致。
接口行为本身由 ``tests/test_web.py`` 覆盖。
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.web import create_app  # noqa: E402
from src.web.config import load_web_config  # noqa: E402
from src.web.app import STATIC_DIR, TEMPLATES_DIR  # noqa: E402


CONFIG = load_web_config()
CONTRACT_ENDPOINTS = {"/api/health", "/api/categories", "/api/predict"}

#: 页面必须存在的交互区域（说明书 §10.1）。
REQUIRED_ANCHORS = {
    "评论输入框": 'id="review"',
    "Top-K 选择": 'id="top-k"',
    "提交按钮": 'id="btn-submit"',
    "清空按钮": 'id="btn-clear"',
    "预测类别区域": 'id="category-body"',
    "Top-K 概率区域": 'id="topk-list"',
    "情感分析区域": 'id="sentiment"',
    "建议处理方式区域": 'id="routing"',
    "接口详情折叠区": 'id="api-details"',
}

#: 不得出现在页面资源里的外部依赖形式。
EXTERNAL_PATTERNS = (
    r"https?://",
    r"//cdn\.",
    r"integrity\s*=",
    r"crossorigin\s*=",
)


class FrontendTestCase(unittest.TestCase):
    """共用客户端：首页在无模型时也必须可用。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._context = TestClient(create_app(model_dir=None))
        cls.client = cls._context.__enter__()
        cls.index_html = cls.client.get("/").text

    @classmethod
    def tearDownClass(cls) -> None:
        cls._context.__exit__(None, None, None)


class IndexServingTests(FrontendTestCase):
    def test_index_returns_html_with_expected_title(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("电商评论智能分析系统", response.text)

    def test_page_loads_without_a_model(self) -> None:
        """说明书 §10.1：没有模型时首页仍可加载，不得白屏或报错。"""
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_static_css_is_served(self) -> None:
        response = self.client.get("/static/css/app.css")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text/css", response.headers.get("content-type", ""))

    def test_static_js_is_served(self) -> None:
        response = self.client.get("/static/js/app.js")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("javascript", response.headers.get("content-type", ""))

    def test_index_is_not_part_of_the_openapi_schema(self) -> None:
        """首页是演示资源，不应混进对外发布的接口文档。"""
        paths = set(self.client.get("/openapi.json").json()["paths"])
        self.assertEqual(paths, CONTRACT_ENDPOINTS, paths)

    def test_frontend_assets_exist_on_disk(self) -> None:
        for path in (
            TEMPLATES_DIR / "index.html",
            STATIC_DIR / "css" / "app.css",
            STATIC_DIR / "js" / "app.js",
        ):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), str(path))


class IndexContentTests(FrontendTestCase):
    def test_required_regions_are_present(self) -> None:
        for name, anchor in REQUIRED_ANCHORS.items():
            with self.subTest(region=name):
                self.assertIn(anchor, self.index_html)

    def test_uses_semantic_and_accessible_markup(self) -> None:
        for token in ("<header", "<nav", "<main", "<section", "<form", "<textarea"):
            with self.subTest(token=token):
                self.assertIn(token, self.index_html)
        self.assertIn('lang="zh-CN"', self.index_html)
        self.assertIn('aria-live', self.index_html)

    def test_review_input_maxlength_matches_config(self) -> None:
        """输入上限必须与 configs/base.yaml 的 api.max_review_chars 一致。"""
        match = re.search(r'id="review"[^>]*\bmaxlength="(\d+)"', self.index_html)
        self.assertIsNotNone(match, "输入框缺少 maxlength 属性")
        self.assertEqual(int(match.group(1)), CONFIG.max_review_chars)

    def test_no_external_resources(self) -> None:
        """说明书 §10.1：页面不得引用 CDN 或任何外部资源。"""
        for name, text in (
            ("index.html", self.index_html),
            ("app.css", self.client.get("/static/css/app.css").text),
            ("app.js", self.client.get("/static/js/app.js").text),
        ):
            for pattern in EXTERNAL_PATTERNS:
                with self.subTest(asset=name, pattern=pattern):
                    self.assertIsNone(re.search(pattern, text), f"{name} 命中 {pattern}")

    def test_only_self_hosted_assets_are_referenced(self) -> None:
        for match in re.finditer(r'(?:href|src)="([^"]+)"', self.index_html):
            target = match.group(1)
            with self.subTest(target=target):
                self.assertTrue(target.startswith("/static/"), target)


class ScriptBehaviourTests(FrontendTestCase):
    """对 app.js 的静态约束检查；运行期行为由浏览器手工联调覆盖。"""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.script = cls.client.get("/static/js/app.js").text

    def test_only_calls_contract_endpoints(self) -> None:
        referenced = set(re.findall(r"/api/[A-Za-z0-9_/-]+", self.script))
        self.assertTrue(referenced, "app.js 未引用任何接口，测试失去意义")
        self.assertLessEqual(referenced, CONTRACT_ENDPOINTS, referenced)

    def test_pure_helpers_are_exposed_for_testing(self) -> None:
        for name in (
            "formatPercent",
            "formatElapsedMs",
            "validatePredictResponse",
            "getRoutingAdvice",
            "clampRatio",
            "classifyHealth",
        ):
            with self.subTest(helper=name):
                self.assertIn("function " + name, self.script)
                self.assertIn(name + ":", self.script)

    def test_avoids_unsafe_dom_patterns(self) -> None:
        """说明书 §8：不使用 eval、内联处理器，也不写入未清洗的 innerHTML。"""
        for pattern in (r"\beval\s*\(", r"new\s+Function\s*\(", r"document\.write\s*\("):
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, self.script), pattern)
        self.assertIsNone(
            re.search(r"\.innerHTML\s*=", self.script), "不得对 innerHTML 赋值"
        )
        self.assertIn("textContent", self.script)

    def test_has_request_timeout(self) -> None:
        self.assertIn("REQUEST_TIMEOUT_MS", self.script)
        self.assertIn("AbortController", self.script)

    def test_does_not_predict_sentiment_locally(self) -> None:
        """说明书 §4.6：前端不得根据评论内容自行推断情感。"""
        for forbidden in ("正向评论", "负向评论", "keyword", "关键词匹配"):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, self.script)

    def test_routing_covers_all_contract_categories(self) -> None:
        """建议映射必须覆盖配置中的全部类别；未知类别走通用回退。"""
        block = re.search(r"ROUTING_BY_CATEGORY\s*=\s*\{(.*?)\n\};", self.script, re.S)
        self.assertIsNotNone(block, "未找到 ROUTING_BY_CATEGORY")
        for category in CONFIG.categories:
            with self.subTest(category=category):
                self.assertIn("'" + category + "'", block.group(1))
        self.assertIn("FALLBACK_ROUTING", self.script)


class StyleTests(FrontendTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.css = cls.client.get("/static/css/app.css").text

    def test_uses_css_variables_for_theme(self) -> None:
        """说明书 §5.1：颜色、圆角、阴影集中由 CSS 变量定义。"""
        for token in ("--color-primary", "--color-heading", "--color-bg",
                      "--radius-card", "--shadow-card"):
            with self.subTest(variable=token):
                self.assertIn(token, self.css)
        self.assertIn(":root", self.css)

    def test_defines_responsive_breakpoints(self) -> None:
        """说明书 §5.2：至少覆盖 1200 / 768 两个断点。"""
        self.assertIn("@media (max-width: 1199px)", self.css)
        self.assertIn("@media (max-width: 767px)", self.css)

    def test_respects_reduced_motion(self) -> None:
        self.assertIn("prefers-reduced-motion", self.css)

    def test_defines_visible_focus_style(self) -> None:
        self.assertIn(":focus-visible", self.css)


if __name__ == "__main__":
    unittest.main()
