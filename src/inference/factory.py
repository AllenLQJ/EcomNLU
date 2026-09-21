"""按模型目录加载出统一的预测器。

契约 §2.5 要求所有模型都能通过同一个入口加载，这样部署代码不需要知道
具体是哪种模型。分派依据是 ``metadata.json`` 里的 ``framework`` 字段。

各框架的适配器**按需导入**：只用传统模型时不会把 torch 和 transformers
拖进内存。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.inference.contract import BaseCategoryPredictor, PredictorError, load_metadata


def _resolve_adapter(framework: str) -> Callable[[Path], BaseCategoryPredictor] | None:
    """按框架名返回适配器类，未知框架返回 ``None``。"""
    if framework == "scikit-learn":
        from src.inference.sklearn_adapter import SklearnCategoryPredictor

        return SklearnCategoryPredictor
    if framework == "fasttext":
        from src.inference.fasttext_adapter import FastTextCategoryPredictor

        return FastTextCategoryPredictor
    if framework == "transformers":
        from src.inference.transformers_adapter import TransformersCategoryPredictor

        return TransformersCategoryPredictor
    return None


def supported_frameworks() -> list[str]:
    """返回当前支持的全部 framework 取值。"""
    return ["scikit-learn", "fasttext", "transformers"]


def load_predictor(model_dir: str | Path) -> BaseCategoryPredictor:
    """按模型目录加载为统一的预测器。

    ``model_dir`` 必须符合 ``docs/contracts.md`` 第二章的模型产物契约。
    """
    directory = Path(model_dir)
    if not directory.is_dir():
        raise PredictorError(f"模型目录不存在：{directory}")

    metadata = load_metadata(directory)
    framework = metadata.get("framework")
    if not isinstance(framework, str):
        raise PredictorError(f"metadata.json 的 framework 必须是字符串，实际为 {framework!r}")

    adapter = _resolve_adapter(framework)
    if adapter is None:
        raise PredictorError(
            f"不支持的 framework：{framework!r}；当前支持 {supported_frameworks()}"
        )
    return adapter(directory)
