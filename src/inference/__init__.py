"""推理层：把各种训练产物统一成同一个预测接口。

对外只需要一个入口：

    from src.inference import load_predictor

    predictor = load_predictor("models/bert_base_chinese")
    predictor.predict(["手机很好用，电池耐用"])

契约见 ``docs/contracts.md`` §2.5。本模块刻意不导入 torch / transformers /
fasttext，适配器在被选中时才加载对应依赖，因此只用传统模型时不会有额外开销。
"""

from __future__ import annotations

from src.inference.contract import (
    BaseCategoryPredictor,
    CategoryPredictor,
    PredictorError,
)
from src.inference.factory import load_predictor, supported_frameworks


__all__ = [
    "BaseCategoryPredictor",
    "CategoryPredictor",
    "PredictorError",
    "load_predictor",
    "supported_frameworks",
]
