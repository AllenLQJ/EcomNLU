"""预测服务：持有预测器，在服务启动时加载一次。

契约 §3.5 明确要求「模型在服务启动时加载**一次**，请求处理中不得重新加载」。
本类因此把预测器作为实例状态持有，``predict`` 只做推理不做加载。

主任务与情感任务各加载一个模型目录（契约 v1.2 §2.1：一个目录只能承载一个任务）。
两者相互独立：情感模型缺失或加载失败时，服务照常提供类别预测，并按契约 §3.3
让 ``sentiment`` 返回 ``null``、``sentiment_enabled`` 返回 ``false``。

模型加载失败**不会**阻止服务启动：服务以未就绪状态运行，``/api/health`` 返回
503 + ``degraded``，``/api/predict`` 返回 ``MODEL_NOT_LOADED``。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.data.config import public_path
from src.inference import load_predictor
from src.inference.contract import PredictorError

from src.web.errors import ApiError, ErrorCode


LOGGER = logging.getLogger(__name__)

PROBABILITY_PRECISION = 6

#: 契约 v1.2 §2.1 规定情感任务的标签集固定为这两个值、顺序不得调换。
#: 加载时校验，避免标签错位导致整体预测反向而没有报错。
EXPECTED_SENTIMENT_LABELS = ("负向", "正向")


class PredictorService:
    """按契约 §3.5 加载并持有预测器。"""

    def __init__(
        self,
        model_dir: Path | None,
        categories: list[str],
        sentiment_model_dir: Path | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._sentiment_model_dir = sentiment_model_dir
        self.categories: list[str] = list(categories)
        self._category_order = {name: index for index, name in enumerate(self.categories)}

        self._predictor: Any = None
        self._load_error: str | None = None
        self.model_name: str | None = None

        self._sentiment_predictor: Any = None
        self._sentiment_order: dict[str, int] = {}
        self._sentiment_error: str | None = None

    # ------------------------------------------------------------------ 状态

    @property
    def is_ready(self) -> bool:
        """主任务模型是否已成功加载。"""
        return self._predictor is not None

    @property
    def load_error(self) -> str | None:
        """主任务模型加载失败的原因，供服务端日志使用。"""
        return self._load_error

    @property
    def sentiment_enabled(self) -> bool:
        """情感任务是否可用（契约 §3.2 的 ``sentiment_enabled``）。"""
        return self._sentiment_predictor is not None

    @property
    def sentiment_error(self) -> str | None:
        """情感模型不可用的原因；可用时为 ``None``。"""
        return self._sentiment_error

    # ------------------------------------------------------------------ 加载

    def load(self) -> None:
        """加载主任务与情感模型；失败只记录不抛出，使服务仍可启动。"""
        self._load_primary()
        self._load_sentiment()

    def _load_primary(self) -> None:
        if self._model_dir is None:
            self._load_error = "未指定模型目录（用 --model-dir 或 WEB_MODEL_DIR 指定）"
            LOGGER.warning("服务以未就绪状态启动：%s", self._load_error)
            return

        LOGGER.info("加载主任务模型：%s", public_path(self._model_dir))
        started = time.perf_counter()
        try:
            self._predictor = load_predictor(self._model_dir)
        except Exception as exc:  # 加载失败不能导致服务无法启动
            self._load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("主任务模型加载失败，服务以未就绪状态启动：%s", self._load_error)
            return
        self.model_name = self._model_dir.name
        LOGGER.info(
            "主任务模型加载完成：%s（耗时 %.2fs）", self.model_name, time.perf_counter() - started
        )

    def _load_sentiment(self) -> None:
        if self._sentiment_model_dir is None:
            self._sentiment_error = (
                "未指定情感模型目录（用 --sentiment-model-dir 或 WEB_SENTIMENT_MODEL_DIR 指定）"
            )
            LOGGER.info("情感任务未启用：%s", self._sentiment_error)
            return

        LOGGER.info("加载情感模型：%s", public_path(self._sentiment_model_dir))
        started = time.perf_counter()
        try:
            predictor = load_predictor(self._sentiment_model_dir)
        except Exception as exc:
            self._sentiment_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("情感模型加载失败，情感任务将不可用：%s", self._sentiment_error)
            return

        labels = tuple(getattr(predictor, "categories", ()) or ())
        if labels != EXPECTED_SENTIMENT_LABELS:
            self._sentiment_error = (
                f"情感模型标签集应为 {list(EXPECTED_SENTIMENT_LABELS)}，实际为 {list(labels)}"
                "（契约 v1.2 §2.1）"
            )
            LOGGER.error("情感模型标签集不符，拒绝启用：%s", self._sentiment_error)
            return

        self._sentiment_predictor = predictor
        self._sentiment_order = {name: index for index, name in enumerate(labels)}
        LOGGER.info(
            "情感模型加载完成：%s（耗时 %.2fs）",
            self._sentiment_model_dir.name,
            time.perf_counter() - started,
        )

    # ------------------------------------------------------------------ 推理

    @staticmethod
    def _rank(probabilities: dict[str, float], order: dict[str, int]) -> list[tuple[str, float]]:
        """按概率降序排名。

        概率相同时按标签在模型 ``label_mapping.json`` 中的顺序排列，保证同一输入的
        结果确定，并与 ``BaseCategoryPredictor.predict`` 取最高概率项的规则一致。
        """
        return sorted(probabilities.items(), key=lambda item: (-item[1], order.get(item[0], 0)))

    def _infer(self, predictor: Any, review: str) -> dict[str, float]:
        try:
            return predictor.predict_proba([review])[0]
        except PredictorError as exc:
            raise ApiError(ErrorCode.INTERNAL_ERROR, f"推理失败：{exc}") from exc

    def _predict_sentiment(self, review: str) -> dict[str, Any] | None:
        """返回契约 §3.3 的 ``sentiment`` 结构；情感未启用时为 ``None``。"""
        if self._sentiment_predictor is None:
            return None
        ranked = self._rank(self._infer(self._sentiment_predictor, review), self._sentiment_order)
        label, probability = ranked[0]
        return {"label": label, "confidence": round(float(probability), PROBABILITY_PRECISION)}

    def classify(self, review: str) -> tuple[str, float]:
        """只做主任务的 top1 分类，返回 (类别, 置信度)；供 analyze 对子句逐句分类。"""
        if self._predictor is None:
            raise ApiError(ErrorCode.MODEL_NOT_LOADED, "模型尚未加载完成")
        label, probability = self._rank(self._infer(self._predictor, review), self._category_order)[0]
        return label, round(float(probability), PROBABILITY_PRECISION)

    def sentiment(self, review: str) -> dict[str, Any] | None:
        """情感结果 ``{"label", "confidence"}``；情感模型未启用时为 ``None``。"""
        return self._predict_sentiment(review)

    def predict(self, review: str, top_k: int) -> dict[str, Any]:
        """对单条评论预测，返回契约 §3.3 规定的响应体。"""
        if self._predictor is None:
            raise ApiError(ErrorCode.MODEL_NOT_LOADED, "模型尚未加载完成")

        started = time.perf_counter()
        probabilities = self._infer(self._predictor, review)
        sentiment = self._predict_sentiment(review)
        # 耗时覆盖本次请求的全部服务端推理（类别 + 情感），与契约 §3.3
        # 「服务端推理耗时」的口径一致。
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        ranked = self._rank(probabilities, self._category_order)
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
            "sentiment": sentiment,
            "elapsed_ms": round(elapsed_ms, 3),
            "model_name": self.model_name,
        }
