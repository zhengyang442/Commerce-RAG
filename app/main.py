from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.errors import validation_error_handler
from app.api.public_preview import PublicPreviewGuardMiddleware
from app.api.routes import router
from app.core.config import Settings
from app.core.logging import configure_logging
from app.core.usage_limits import DailyExternalCallBudget
from app.generation.orchestrator import AdapterFactory, build_adapter
from app.query_understanding.rewriter import ProviderQueryRewriter
from app.query_understanding.service import RewriterFactory

STATIC_DIR = Settings().project_root / "app" / "static"


def create_app(
    settings: Settings | None = None,
    *,
    llm_adapter_factory: AdapterFactory | None = None,
    query_rewriter_factory: RewriterFactory | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    configure_logging(api_key=resolved_settings.llm_api_key)
    app = FastAPI(
        title="CommerceRAG",
        version="0.4.0",
        docs_url=None if resolved_settings.public_preview else "/docs",
        redoc_url=None if resolved_settings.public_preview else "/redoc",
        openapi_url=None if resolved_settings.public_preview else "/openapi.json",
    )
    app.state.settings = resolved_settings
    app.state.llm_adapter_factory = llm_adapter_factory or build_adapter
    app.state.query_rewriter_factory = query_rewriter_factory or ProviderQueryRewriter
    app.state.external_call_semaphore = asyncio.Semaphore(
        resolved_settings.max_external_concurrency
    )
    app.state.external_call_budget = (
        DailyExternalCallBudget(resolved_settings.external_calls_per_day)
        if resolved_settings.public_preview
        else None
    )
    if resolved_settings.public_preview:
        app.add_middleware(
            PublicPreviewGuardMiddleware,
            max_request_bytes=resolved_settings.max_request_bytes,
            search_requests_per_minute=resolved_settings.rate_limit_per_minute,
            answer_requests_per_minute=resolved_settings.answer_rate_limit_per_minute,
        )
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=list(resolved_settings.allowed_hosts),
        )
        if resolved_settings.allowed_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=list(resolved_settings.allowed_origins),
                allow_credentials=False,
                allow_methods=["GET", "POST"],
                allow_headers=["content-type"],
            )
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-cache"},
        )

    return app


app = create_app()
