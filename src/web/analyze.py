"""``POST /api/v1/nlu/analyze``：把一段用户文本拆成子句，逐句给出类别提示，并给整段的情感。

EcomPilot / EcomOps 这类上游要的是"这句话在说哪类商品、语气如何"，而不是单条评论的 top-k。
本模块只做编排：分类与情感仍由 ``PredictorService`` 里已加载的模型完成。

注意：底层是用电商**评论**训练的十分类模型，对导购需求、售后吐槽这类文本的效果没有验证过；
所以置信度低于阈值的子句一律输出 ``ABSTAIN``（不给类别提示），情感模型没配置时返回 ``NEUTRAL`` 而不是猜。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from src.web.config import AnalyzeConfig
from src.web.errors import ApiError, ErrorCode
from src.web.service import PredictorService

CLAUSE_SPLIT = re.compile(r"以及|顺便|还要|[，,。！!？?；;、和\n]")
SENTIMENT_ENUM = {"正向": "POSITIVE", "负向": "NEGATIVE"}


class AnalyzeRequest(BaseModel):
    """请求体。``scene_type`` 决定是否适用以及负向时是否加急。"""

    text: str
    scene_type: str = "AFTER_SALE_FEEDBACK"


def split_clauses(text: str, limit: int) -> list[str]:
    parts = [part.strip() for part in CLAUSE_SPLIT.split(text)]
    return [part for part in parts if part][:limit]


def _clause(text: str, label: str, confidence: float, config: AnalyzeConfig) -> dict[str, Any]:
    if confidence < config.confidence_threshold:
        return {"text": text, "predicted_label": "ABSTAIN", "category_confidence": confidence,
                "category_code": None, "catalog_category_hint": None, "brand_hint": None, "item_type": "PRODUCT"}
    entry = config.categories.get(label, {})
    return {"text": text, "predicted_label": label, "category_confidence": confidence,
            "category_code": entry.get("category_code"),
            "catalog_category_hint": entry.get("catalog_category_id"),
            "brand_hint": entry.get("brand_hint"), "item_type": entry.get("item_type", "PRODUCT")}


def analyze(request: AnalyzeRequest, config: AnalyzeConfig, service: PredictorService) -> dict[str, Any]:
    text = request.text.strip()
    if not text:
        raise ApiError(ErrorCode.EMPTY_TEXT, "text 不能为空")
    if len(text) > config.max_text_chars:
        raise ApiError(ErrorCode.TEXT_TOO_LONG, f"text 长度 {len(text)} 超过上限 {config.max_text_chars} 字符")
    if request.scene_type in config.not_applicable_scenes:
        # 不适用的场景不调用模型：既省算力，也避免对 PDF 正文、客服致歉话术给出误导性的类别/情感
        return {"clauses": [], "sentiment": "NEUTRAL", "sentiment_score": 0, "priority_boost": False,
                "scene_not_applicable": True, "model_name": service.model_name}
    if not service.is_ready:
        raise ApiError(ErrorCode.MODEL_NOT_LOADED, "模型尚未加载完成")

    parts = split_clauses(text, config.max_clauses)
    # 整段和所有子句放进同一批，模型只前向一次（路线图图 2-1："批量推理"）；与整段相同的子句不重复算
    unique = list(dict.fromkeys([text, *parts]))
    predictions = dict(zip(unique, service.classify_many(unique)))
    clauses = [_clause(part, *predictions[part], config) for part in parts]

    sentiment = service.sentiment(text)
    if sentiment and sentiment["label"] in SENTIMENT_ENUM:
        enum, score = SENTIMENT_ENUM[sentiment["label"]], float(sentiment["confidence"])
    else:
        enum, score = "NEUTRAL", 0.0
    return {"clauses": clauses, "sentiment": enum, "sentiment_score": score,
            "priority_boost": (enum == "NEGATIVE" and score >= config.priority_boost_min_score
                               and request.scene_type in config.priority_boost_scenes),
            "scene_not_applicable": False, "model_name": service.model_name}
