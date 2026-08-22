"""Application entry point.

Startup does three things beyond wiring routes:

1. **Reports the running configuration.** Which provider, which model, which
   store. Most confused bug reports about an AI app are actually "it was
   pointed at the wrong backend", and a banner ends those in one glance.
2. **Probes inference without blocking boot.** A dead vLLM server produces a
   loud warning and a degraded health endpoint, not a crash loop. The UI can
   then say "inference unavailable" instead of hanging.
3. **Connects the document service to the broadcast hub**, so ingestion
   progress reaches the browser.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import channels, documents, health, websocket
from app.core.config import settings
from app.core.exceptions import OmnivoresError
from app.core.logging import configure_logging, get_logger, set_request_id
from app.core.models import new_id
from app.inference.client import close_inference_client, get_inference_client
from app.rag.embeddings import close_embedding_provider
from app.rag.store import close_vector_store
from app.services.document_service import get_document_service
from app.services.message_service import get_message_service
from app.tools.web_search import close_search_provider

configure_logging()
logger = get_logger(__name__)


def _startup_banner() -> None:
    logger.info("=" * 68)
    logger.info("Omnivores AI backend")
    logger.info("  environment      : %s", settings.app_env)
    logger.info(
        "  inference        : %s  (model: %s)",
        settings.llm_provider,
        settings.resolved_model_name,
    )
    if settings.llm_provider == "vllm":
        logger.info("  vllm endpoint    : %s", settings.vllm_base_url)
    logger.info(
        "  embeddings       : %s  (%d dims)",
        settings.embedding_provider,
        settings.embedding_dimensions,
    )
    logger.info("  vector store     : %s", settings.vector_store)
    logger.info("  reranker         : %s", settings.reranker)
    logger.info("  python sandbox   : %s", settings.python_sandbox_mode)
    logger.info("  web search       : %s", settings.web_search_provider)
    logger.info("  auth             : %s", "enabled" if settings.auth_enabled else "OPEN")
    logger.info("=" * 68)

    if settings.llm_provider == "mock":
        logger.warning(
            "LLM_PROVIDER=mock - all model output is synthetic. Set LLM_PROVIDER=vllm "
            "and VLLM_BASE_URL to use a real model."
        )
    for warning in settings.production_warnings():
        logger.warning("PRODUCTION WARNING: %s", warning)


async def _probe_inference() -> None:
    """Check the model server once at boot. Never blocks startup."""
    try:
        client = await get_inference_client()
        status = await asyncio.wait_for(client.health(), timeout=15.0)
    except Exception as exc:  # noqa: BLE001 - startup must survive a dead backend
        logger.error("inference probe failed: %s", exc)
        return

    if status.status == "healthy":
        logger.info(
            "inference reachable (model=%s, %.0fms)",
            status.model or settings.resolved_model_name,
            status.latency_ms or 0.0,
        )
    else:
        logger.error(
            "INFERENCE %s: %s", status.status.upper(), status.detail or "no detail"
        )
        if settings.llm_provider == "vllm":
            logger.error(
                "The app will start, but agents will fail until vLLM answers at %s. "
                "See inference/README.md, or set LLM_PROVIDER=mock to develop "
                "without a GPU.",
                settings.vllm_base_url,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_banner()
    settings.ensure_directories()

    # Document progress needs to reach the browser.
    messages = get_message_service()
    get_document_service().set_event_hook(messages.broadcast)

    await _probe_inference()

    yield

    logger.info("shutting down")
    from app.services.agent_service import get_agent_service

    cancelled = get_agent_service().cancel_all()
    if cancelled:
        logger.info("cancelled %d in-flight agent run(s)", cancelled)

    await close_inference_client()
    await close_embedding_provider()
    await close_vector_store()
    await close_search_provider()


app = FastAPI(
    title="Omnivores AI",
    description=(
        "A local-first multi-agent AI workspace: agents as chat participants, "
        "document RAG with citations, streaming, tool calling, and a pluggable "
        "inference backend (vLLM, OpenAI, or a deterministic mock)."
    ),
    version=health.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a correlation id to every request and echo it back."""
    request_id = request.headers.get("X-Request-ID") or new_id("req")
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(OmnivoresError)
async def omnivores_error_handler(request: Request, exc: OmnivoresError) -> JSONResponse:
    """Turn typed errors into structured responses.

    Server-side faults are logged with a traceback; client mistakes are not,
    so a scanner probing for bad input cannot flood the logs.
    """
    if exc.http_status >= 500:
        logger.exception("%s: %s", exc.code, exc.message)
    else:
        logger.info("%s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Something went wrong.",
                # The type alone is safe and usually enough to start debugging.
                "detail": type(exc).__name__,
                "retryable": False,
            }
        },
    )


app.include_router(health.router)
app.include_router(channels.router)
app.include_router(documents.router)
app.include_router(websocket.router)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "Omnivores AI",
        "version": health.VERSION,
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws",
    }


def main() -> None:
    """Run the dev server. Production should invoke uvicorn directly."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        log_config=None,  # our own logging is already configured
    )


if __name__ == "__main__":
    main()
