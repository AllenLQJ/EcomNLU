"""预测服务：持有唯一一份预测器，在服务启动时加载一次。

契约 §3.5 明确要求「模型在服务启动时加载**一次**，请求处理中不得重新加载」。
本类因此把预测器作为实例状态持有，``predict`` 只做推理不做加载。

模型加载失败**不会**阻止服务启动：服务以未就绪状态运行，``/api/health`` 返回
503 + ``degraded``，``/api/predict`` 返回 ``MODEL_NOT_LOADED``。这样在模型产物
就绪前即可联调前端与错误处理路径。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.inference import load_predictor
from src.inference.contract import PredictorError

from src.web.errors import ApiError, ErrorCode


LOGGER = logging.getLogger(__name__)

#: 情感任务在 M5 才交付。契约 §3.3 规定该字段在 M5 完成前固定为 null，
#: 且**不得省略**，以保证前端结构稳定。
SENTIMENT_ENABLED = False
PROBABILITY_PRECISION = 6


class PredictorService:
    """按契约 §3.5 加载并持有预测器。"""

    def __init__(self, model_dir: Path | None, categories: list[str]) -> None:
        self._model_dir = model_dir
        self.categories: list[str] = list(categories)
        self._category_order = {name: index for index, name in enumerate(self.categories)}
        self._predictor: Any = None
        self._load_error: str | None = None
        self.model_name: str | None = None

    @property
    def is_ready(self) -> bool:
        """模型是否已成功加载。"""
        return self._predictor is not None

    @property
    def load_error(self) -> str | None:
        """模型加载失败的原因，供服务端日志使用。"""
        return self._load_error

    def load(self) -> None:
        """加载模型；失败只记录不抛出，使服务仍可启动。"""
        if self._model_dir is None:
            self._load_error = "未指定模型目录（用 --model-dir 或 WEB_MODEL_DIR 指定）"
            LOGGER.warning("服务以未就绪状态启动：%s", self._load_error)
            return

        LOGGER.info("加载模型：%s", self._model_dir)
        started = time.perf_counter()
        try:
            self._predictor = load_predictor(self._model_dir)
        except Exception as exc:  # 加载失败不能导致服务无法启动
            self._load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("模型加载失败，服务以未就绪状态启动：%s", self._load_error)
            return
        self.model_name = self._model_dir.name
        LOGGER.info("模型加载完成：%s（耗时 %.2fs）", self.model_name, time.perf_counter() - started)

    def _rank(self, probabilities: dict[str, float]) -> list[tuple[str, float]]:
        """按概率降序排名。

        概率相同时按 ``configs/base.yaml`` 的类别顺序排列，保证同一输入的结果确定，
        并与 ``BaseCategoryPredictor.predict`` 取最高概率项的规则保持一致。
        """
        return sorted(
            probabilities.items(),
            key=lambda item: (-item[1], self._category_order.get(item[0], 0)),
        )

    def predict(self, review: str, top_k: int) -> dict[str, Any]:
        """对单条评论预测，返回契约 §3.3 规定的响应体。"""
        if self._predictor is None:
            raise ApiError(ErrorCode.MODEL_NOT_LOADED, "模型尚未加载完成")

        started = time.perf_counter()
        try:
            probabilities = self._predictor.predict_proba([review])[0]
        except PredictorError as exc:
            raise ApiError(ErrorCode.INTERNAL_ERROR, f"推理失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        ranked = self._rank(probabilities)
        best_label, best_probability = ranked[0]
        return {
            "category": {
                "label": best_label,
                "confidence": round(float(best_probability), PROBABILITY_PRECISION),
            },
            "top_k": [
                {"label": label, "probability": round(float(probability), PROBABILITY_PRECISION)}
                for label, probability in ranked[:top_k]
            ],
            "sentiment": None,
            "elapsed_ms": round(elapsed_ms, 3),
            "model_name": self.model_name,
        }
