"""Web 层：按 ``docs/contracts.md`` §3 提供预测接口。

    from src.web import create_app

    app = create_app(model_dir="models/tfidf_svm")

启动服务::

    python -m src.web.app --model-dir models/tfidf_svm
"""

from __future__ import annotations

from typing import Any

from src.web.config import WebConfig, load_web_config
from src.web.errors import ApiError, ErrorCode
from src.web.service import PredictorService


def create_app(*args: Any, **kwargs: Any) -> Any:
    """惰性构建 FastAPI 应用，避免 ``python -m src.web.app`` 重复导入模块。"""
    from src.web.app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = [
    "ApiError",
    "ErrorCode",
    "PredictorService",
    "WebConfig",
    "create_app",
    "load_web_config",
]
