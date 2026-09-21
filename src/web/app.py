"""FastAPI 应用：按 ``docs/contracts.md`` §3 提供三个端点。

    GET  /api/health      健康检查
    GET  /api/categories  返回类别列表
    POST /api/predict     单条评论预测

启动::

    python -m src.web.app --model-dir models/tfidf_svm --port 8000

模型未就绪时服务仍可启动，健康检查返回 503 + ``degraded``，预测返回
``MODEL_NOT_LOADED``，便于在模型产物就绪前联调前端。
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictInt

from src.web.config import DEFAULT_CONFIG_PATH, WebConfig, load_web_config
from src.web.errors import ApiError, ErrorCode
from src.web.service import SENTIMENT_ENABLED, PredictorService


LOGGER = logging.getLogger(__name__)

ERROR_RESPONSE_EXAMPLE = {
    "error": {"code": "EMPTY_REVIEW", "message": "评论内容不能为空"}
}


class PredictRequest(BaseModel):
    """``POST /api/predict`` 的请求体（契约 §3.3）。"""

    review: str
    top_k: StrictInt = 3


def map_validation_error(exc: RequestValidationError) -> ApiError:
    """把 FastAPI 的请求校验错误映射为契约 §3.4 规定的错误码。

    这一步必须做：FastAPI 默认对请求体校验失败返回 **422** 且响应结构为
    ``{"detail": [...]}``，而契约要求 **400** + ``{"error": {"code", "message"}}``。
    不覆盖的话，畸形 JSON、缺少 review、``review`` 类型错误、``top_k`` 类型错误
    这四类请求全部违反契约。

    ``review`` 存在但类型不是字符串时归为 ``MISSING_REVIEW``：从接口语义看客户端
    没有提供可用的评论字段，契约未定义独立的类型错误码。
    """
    for error in exc.errors():
        error_type = str(error.get("type", ""))
        location = tuple(error.get("loc", ()))
        if error_type == "json_invalid":
            return ApiError(ErrorCode.INVALID_JSON, "请求体不是合法 JSON")
        if location == ("body", "review"):
            return ApiError(ErrorCode.MISSING_REVIEW, "review 字段缺失或不是字符串")
        if location == ("body", "top_k"):
            return ApiError(ErrorCode.INVALID_TOP_K, "top_k 必须是 1 到 10 的整数")
    return ApiError(ErrorCode.INVALID_JSON, "请求体必须是包含 review 字段的 JSON 对象")


def register_error_handlers(app: FastAPI) -> None:
    """注册异常处理器，保证任何异常都以契约 §3.4 的结构返回。"""

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_body())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        error = map_validation_error(exc)
        return JSONResponse(status_code=error.status_code, content=error.to_body())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # 契约 §3.5：不得把 Python 堆栈返回给前端。原因只写服务端日志。
        LOGGER.exception("未处理异常：%s %s", request.method, request.url.path)
        error = ApiError(ErrorCode.INTERNAL_ERROR, "服务内部错误")
        return JSONResponse(status_code=error.status_code, content=error.to_body())


def register_routes(app: FastAPI, config: WebConfig, service: PredictorService) -> None:
    """注册契约 §3.1 规定的三个端点。"""

    @app.get("/api/health", summary="健康检查")
    async def health() -> JSONResponse:
        loaded = service.is_ready
        body: dict[str, Any] = {
            "status": "ok" if loaded else "degraded",
            "model_name": service.model_name,
            "model_loaded": loaded,
            "sentiment_enabled": SENTIMENT_ENABLED,
            "categories": config.categories,
        }
        return JSONResponse(status_code=200 if loaded else 503, content=body)

    @app.get("/api/categories", summary="返回类别列表")
    async def categories() -> dict[str, Any]:
        return {"categories": config.categories}

    @app.post(
        "/api/predict",
        summary="单条评论预测",
        responses={
            400: {"description": "请求不合法", "content": {"application/json": {"example": ERROR_RESPONSE_EXAMPLE}}},
            503: {"description": "模型未加载"},
        },
    )
    async def predict(payload: PredictRequest) -> dict[str, Any]:
        review = payload.review
        if not review.strip():
            raise ApiError(ErrorCode.EMPTY_REVIEW, "评论内容不能为空")
        if len(review) > config.max_review_chars:
            raise ApiError(
                ErrorCode.REVIEW_TOO_LONG,
                f"评论长度 {len(review)} 超过上限 {config.max_review_chars} 字符",
            )

        top_k = payload.top_k
        if not 1 <= top_k <= config.max_top_k:
            raise ApiError(
                ErrorCode.INVALID_TOP_K, f"top_k 必须是 1 到 {config.max_top_k} 的整数"
            )

        # 训练数据已去除首尾空白（契约 §1.2），推理阶段保持一致。
        return service.predict(review.strip(), top_k)


def create_app(
    config_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    web_config: WebConfig | None = None,
) -> FastAPI:
    """构建应用。模型在 lifespan 启动阶段加载一次（契约 §3.5）。"""
    config = web_config or load_web_config(config_path, model_dir)
    service = PredictorService(config.model_dir, config.categories)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Iterator[None]:
        service.load()
        yield

    app = FastAPI(
        title="电商评论分类预测接口",
        description="接口契约见 docs/contracts.md §3。",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.web_config = config
    app.state.service = service
    register_error_handlers(app)
    register_routes(app, config, service)
    return app


app = create_app()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="电商评论分类预测服务")
    parser.add_argument("--model-dir", type=Path, default=None, help="模型产物目录")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """启动服务。"""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import uvicorn

    application = create_app(args.config, args.model_dir)
    uvicorn.run(application, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
