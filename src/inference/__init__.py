"""推理层：把各种训练产物统一成同一个预测接口。

对外只需要一个入口：

    from src.inference import load_predictor

    predictor = load_predictor("models/bert_base_chinese")
    predictor.predict(["手机很好用，电池耐用"])

契约见 ``docs/contracts.md`` §2.5。本模块刻意不导入 torch / transformers /
fasttext，适配器在被选中时才加载对应依赖，因此只用传统模型时不会有额外开销。

**该入口是任务无关的**（契约 v1.2 草案 §2.5）：标签集取自模型目录自己的
``label_mapping.json``，不按任务分支。因此 M5 的情感模型也用同一个入口加载，
只是标签集变成 ``负向`` / ``正向``：

    predictor = load_predictor("models/sentiment_tfidf_svm")
    probabilities = predictor.predict_proba(["质量太差了，用两天就坏了"])
    # -> [{"负向": 0.98, "正向": 0.02}]
    # API 契约 §3.4 的 {"label": …, "confidence": …} 即取最高概率项与其概率

刻意不新增 ``SentimentPredictor`` 协议：两个任务的接口形状完全相同，只差标签集，
新增一个仅名字不同的协议属于重复（见契约 v1.2 草案 §2.5 的讨论）。
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
