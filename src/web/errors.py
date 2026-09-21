"""契约 §3.4 规定的错误码与统一错误响应结构。

对外的错误响应固定为::

    {"error": {"code": "EMPTY_REVIEW", "message": "评论内容不能为空"}}

错误码取值与 HTTP 状态码一一对应，**不得随意新增或改名**——前端按 code 分支处理。
"""

from __future__ import annotations

from typing import Any


class ErrorCode:
    """契约 §3.4 错误码表。"""

    INVALID_JSON = "INVALID_JSON"
    MISSING_REVIEW = "MISSING_REVIEW"
    EMPTY_REVIEW = "EMPTY_REVIEW"
    REVIEW_TOO_LONG = "REVIEW_TOO_LONG"
    INVALID_TOP_K = "INVALID_TOP_K"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


HTTP_STATUS: dict[str, int] = {
    ErrorCode.INVALID_JSON: 400,
    ErrorCode.MISSING_REVIEW: 400,
    ErrorCode.EMPTY_REVIEW: 400,
    ErrorCode.REVIEW_TOO_LONG: 400,
    ErrorCode.INVALID_TOP_K: 400,
    ErrorCode.MODEL_NOT_LOADED: 503,
    ErrorCode.INTERNAL_ERROR: 500,
}


class ApiError(Exception):
    """可预期的接口错误，会被转换为契约 §3.4 的统一响应结构。

    服务端异常必须经由本类或全局异常处理器转换，**不得把 Python 堆栈返回给前端**（契约 §3.5）。
    """

    def __init__(self, code: str, message: str) -> None:
        if code not in HTTP_STATUS:
            raise ValueError(f"未定义的错误码：{code!r}")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

    @property
    def status_code(self) -> int:
        """该错误码对应的 HTTP 状态码。"""
        return HTTP_STATUS[self.code]

    def to_body(self) -> dict[str, Any]:
        """契约 §3.4 的统一错误响应体。"""
        return {"error": {"code": self.code, "message": self.message}}
