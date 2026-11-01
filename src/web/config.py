"""Web 层配置。

类别列表与长度上限都从 ``configs/base.yaml`` 读取，不在代码里硬编码。
类别以配置为准而非模型产物：模型未就绪时 ``/api/categories`` 仍须可用（契约 §3.1），
而契约 §1.3 已保证 ``label_mapping.json`` 的 categories 与配置一致，并由
``scripts/verify_data_contract.py`` 强制校验。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.data.config import public_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_MAX_REVIEW_CHARS = 2000
DEFAULT_TOP_K = 3
MAX_TOP_K = 10
MODEL_DIR_ENV = "WEB_MODEL_DIR"
SENTIMENT_MODEL_DIR_ENV = "WEB_SENTIMENT_MODEL_DIR"


@dataclass(frozen=True)
class AnalyzeConfig:
    """``POST /api/v1/nlu/analyze`` 的配置（``configs/base.yaml`` 的 ``analyze`` 段）。"""

    confidence_threshold: float = 0.65
    max_clauses: int = 5
    max_text_chars: int = 500
    not_applicable_scenes: tuple[str, ...] = ("PDF_BODY", "CS_APOLOGY")
    priority_boost_scenes: tuple[str, ...] = ("AFTER_SALE_FEEDBACK", "aftersales")
    #: 类别 -> {category_code, catalog_category_id, brand_hint?, item_type?}
    categories: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class WebConfig:
    """Web 服务运行所需的配置。"""

    categories: list[str] = field(default_factory=list)
    max_review_chars: int = DEFAULT_MAX_REVIEW_CHARS
    model_dir: Path | None = None
    #: 情感任务模型目录（M5）。为 ``None`` 时契约 §3.3 要求 ``sentiment`` 返回 ``null``。
    sentiment_model_dir: Path | None = None
    default_top_k: int = DEFAULT_TOP_K
    max_top_k: int = MAX_TOP_K
    analyze: AnalyzeConfig = field(default_factory=AnalyzeConfig)

    @property
    def category_count(self) -> int:
        """类别总数。"""
        return len(self.categories)


def _resolve_model_dir(
    explicit: str | Path | None, configured: object, env_var: str = MODEL_DIR_ENV
) -> Path | None:
    """确定模型目录：显式参数 > 环境变量 > 配置文件。"""
    candidate: str | Path | None = explicit
    if candidate is None:
        candidate = os.environ.get(env_var) or None
    if candidate is None and isinstance(configured, str) and configured.strip():
        candidate = configured.strip()
    if candidate is None:
        return None
    return Path(candidate)


def _load_analyze(raw: object, categories: list[str], path: Path) -> AnalyzeConfig:
    if raw is None:
        return AnalyzeConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 的 analyze 段必须是映射")
    defaults = AnalyzeConfig()
    threshold = raw.get("confidence_threshold", defaults.confidence_threshold)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError(f"{path} 的 analyze.confidence_threshold 必须在 0 到 1 之间，实际为 {threshold!r}")
    ints = {}
    for key, default in (("max_clauses", defaults.max_clauses), ("max_text_chars", defaults.max_text_chars)):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path} 的 analyze.{key} 必须是正整数，实际为 {value!r}")
        ints[key] = value
    scenes = {}
    for key, default in (("not_applicable_scenes", defaults.not_applicable_scenes),
                         ("priority_boost_scenes", defaults.priority_boost_scenes)):
        value = raw.get(key, list(default))
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"{path} 的 analyze.{key} 必须是非空字符串列表")
        scenes[key] = tuple(value)
    mapping = raw.get("categories", {}) or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"{path} 的 analyze.categories 必须是映射")
    unknown = sorted(set(mapping) - set(categories))
    if unknown:
        raise ValueError(f"{path} 的 analyze.categories 含未知类别 {unknown}（必须是 task.categories 之一）")
    for name, entry in mapping.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("category_code"), str) or not entry["category_code"]:
            raise ValueError(f"{path} 的 analyze.categories.{name} 必须是含 category_code 的映射")
    return AnalyzeConfig(confidence_threshold=float(threshold), categories=dict(mapping), **ints, **scenes)


def load_web_config(
    config_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    sentiment_model_dir: str | Path | None = None,
) -> WebConfig:
    """读取并校验 Web 层配置。"""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{public_path(path)}")

    with path.open("r", encoding="utf-8") as file_obj:
        raw = yaml.safe_load(file_obj) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 的顶层必须是映射")

    categories = raw.get("task", {}).get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{path} 缺少非空的 task.categories")
    if not all(isinstance(name, str) and name.strip() for name in categories):
        raise ValueError(f"{path} 的 task.categories 必须全部是非空字符串")

    api_section = raw.get("api", {})
    if not isinstance(api_section, dict):
        raise ValueError(f"{path} 的 api 段必须是映射")

    max_review_chars = api_section.get("max_review_chars", DEFAULT_MAX_REVIEW_CHARS)
    if (
        not isinstance(max_review_chars, int)
        or isinstance(max_review_chars, bool)
        or max_review_chars <= 0
    ):
        raise ValueError(
            f"{path} 的 api.max_review_chars 必须是正整数，实际为 {max_review_chars!r}"
        )

    cleaned_categories = [name.strip() for name in categories]
    return WebConfig(
        analyze=_load_analyze(raw.get("analyze"), cleaned_categories, path),
        categories=cleaned_categories,
        max_review_chars=max_review_chars,
        model_dir=_resolve_model_dir(model_dir, api_section.get("model_dir")),
        sentiment_model_dir=_resolve_model_dir(
            sentiment_model_dir,
            api_section.get("sentiment_model_dir"),
            env_var=SENTIMENT_MODEL_DIR_ENV,
        ),
    )
