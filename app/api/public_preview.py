from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from collections.abc import MutableMapping
from typing import Any

from app.core.request_id import new_request_id

PROTECTED_PATHS = {"/api/search", "/api/answer"}


class PublicPreviewGuardMiddleware:
    """Small single-process preview guard; a reverse proxy remains the outer security layer."""

    def __init__(self, app, *, max_request_bytes: int, requests_per_minute: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes
        self.requests_per_minute = requests_per_minute
        self._requests: MutableMapping[str, deque[float]] = defaultdict(deque)

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in PROTECTED_PATHS:
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            content_length = headers.get(b"content-length")
            if content_length:
                try:
                    too_large = int(content_length) > self.max_request_bytes
                except ValueError:
                    too_large = True
                if too_large:
                    await self._json_error(send, 413, "request_too_large", "请求体过大")
                    return
            client = scope.get("client") or ("unknown", 0)
            if not self._allow(str(client[0])):
                await self._json_error(send, 429, "rate_limited", "请求过于频繁，请稍后再试")
                return

        buffered: list[dict[str, Any]] = []
        total_bytes = 0
        if path in PROTECTED_PATHS:
            while True:
                message = await receive()
                buffered.append(message)
                if message.get("type") != "http.request":
                    break
                total_bytes += len(message.get("body", b""))
                if total_bytes > self.max_request_bytes:
                    await self._json_error(send, 413, "request_too_large", "请求体过大")
                    return
                if not message.get("more_body", False):
                    break

        async def guarded_receive():
            if buffered:
                return buffered.pop(0)
            return await receive()

        async def secure_send(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"x-frame-options", b"DENY"),
                    ]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, guarded_receive, secure_send)

    def _allow(self, client: str) -> bool:
        now = time.monotonic()
        window = self._requests[client]
        while window and window[0] <= now - 60:
            window.popleft()
        if len(window) >= self.requests_per_minute:
            return False
        window.append(now)
        return True

    @staticmethod
    async def _json_error(send, status: int, code: str, detail: str) -> None:
        body = json.dumps(
            {
                "request_id": new_request_id(),
                "error": {"code": code, "message": detail},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
