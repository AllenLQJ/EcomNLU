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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_MAX_REVIEW_CHARS = 2000
DEFAULT_TOP_K = 3
MAX_TOP_K = 10
MODEL_DIR_ENV = "WEB_MODEL_DIR"


@dataclass(frozen=True)
class WebConfig:
    """Web 服务运行所需的配置。"""

    categories: list[str] = field(default_factory=list)
    max_review_chars: int = DEFAULT_MAX_REVIEW_CHARS
    model_dir: Path | None = None
    default_top_k: int = DEFAULT_TOP_K
    max_top_k: int = MAX_TOP_K

    @property
    def category_count(self) -> int:
        """类别总数。"""
        return len(self.categories)


def _resolve_model_dir(
    explicit: str | Path | None, configured: object
) -> Path | None:
    """确定模型目录：显式参数 > 环境变量 > 配置文件。"""
    candidate: str | Path | None = explicit
    if candidate is None:
        candidate = os.environ.get(MODEL_DIR_ENV) or None
    if candidate is None and isinstance(configured, str) and configured.strip():
        candidate = configured.strip()
    if candidate is None:
        return None
    return Path(candidate)


def load_web_config(
    config_path: str | Path | None = None, model_dir: str | Path | None = None
) -> WebConfig:
    """读取并校验 Web 层配置。"""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")

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

    return WebConfig(
        categories=[name.strip() for name in categories],
        max_review_chars=max_review_chars,
        model_dir=_resolve_model_dir(model_dir, api_section.get("model_dir")),
    )
