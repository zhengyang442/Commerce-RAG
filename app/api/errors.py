from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.request_id import new_request_id


def validation_error_handler(_: Request, error: RequestValidationError) -> JSONResponse:
    messages = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ()))
        messages.append(f"{location}: {item.get('msg', '请求参数无效')}")
    return JSONResponse(
        status_code=422,
        content={
            "request_id": new_request_id(),
            "error": {"code": "invalid_request", "message": "; ".join(messages)},
        },
    )
