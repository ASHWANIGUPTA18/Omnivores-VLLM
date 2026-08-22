"""Health and metrics endpoints.

``/health`` is the liveness probe: it answers as long as the process is up, and
it never blocks on a downstream service.

``/health/inference`` and ``/health/full`` are diagnostics. They do reach out to
the model server, the vector store and the sandbox, and they report each
component separately, because "the app is down" and "vLLM is not running" need
very different responses from whoever is on call.

The overall status is ``degraded`` rather than ``unavailable`` when a
non-essential component is down: an unreachable sandbox stops code execution
but the rest of the product still works.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import metrics_registry
from app.core.models import ComponentHealth, HealthReport
from app.inference.client import get_inference_client
from app.rag.embeddings import get_embedding_provider
from app.rag.store import get_vector_store
from app.sandbox import get_sandbox
from app.services.document_service import get_document_service
from app.services.message_service import get_message_service
from app.tools.registry import get_tool_registry

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

VERSION = "1.0.0"

#: Components whose failure makes the whole service unavailable rather than
#: merely degraded.
_ESSENTIAL = {"inference"}


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness. Cheap, never touches a downstream service."""
    return {
        "status": "healthy",
        "version": VERSION,
        "app_env": settings.app_env,
        "provider": settings.llm_provider,
        "model": settings.resolved_model_name,
    }


@router.get("/health/inference")
async def inference_health() -> dict[str, Any]:
    """Can we actually generate right now?"""
    client = await get_inference_client()
    status = await client.health()
    payload: dict[str, Any] = {"provider": client.provider, **status.to_dict()}

    try:
        info = await client.model_info()
        payload["model_info"] = info.to_dict()
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail
        payload["model_info"] = {"error": str(exc)}

    if status.status != "healthy" and settings.llm_provider == "vllm":
        payload["hint"] = (
            f"Start vLLM and confirm it answers at {settings.vllm_base_url}/models. "
            "See inference/README.md. To develop without a GPU, set LLM_PROVIDER=mock."
        )
    return payload


@router.get("/health/full", response_model=HealthReport)
async def full_health() -> HealthReport:
    """Every component, checked independently."""
    components: dict[str, ComponentHealth] = {}

    # --- inference ---------------------------------------------------------
    try:
        client = await get_inference_client()
        status = await client.health()
        components["inference"] = ComponentHealth(
            status=status.status,
            detail=status.detail,
            metadata={
                "provider": client.provider,
                "model": status.model or settings.resolved_model_name,
                "latency_ms": status.latency_ms,
            },
        )
    except Exception as exc:  # noqa: BLE001
        components["inference"] = ComponentHealth(
            status="unavailable", detail=f"{type(exc).__name__}: {exc}"
        )

    # --- embeddings --------------------------------------------------------
    try:
        provider = get_embedding_provider()
        state, detail = await provider.health()
        components["embeddings"] = ComponentHealth(
            status=state,
            detail=detail,
            metadata={"provider": provider.name, "dimensions": provider.dimensions},
        )
    except Exception as exc:  # noqa: BLE001
        components["embeddings"] = ComponentHealth(
            status="unavailable", detail=f"{type(exc).__name__}: {exc}"
        )

    # --- vector store ------------------------------------------------------
    try:
        store = get_vector_store()
        state, detail = await store.health()
        components["vector_store"] = ComponentHealth(
            status=state, detail=detail, metadata={"backend": store.name}
        )
    except Exception as exc:  # noqa: BLE001
        components["vector_store"] = ComponentHealth(
            status="unavailable", detail=f"{type(exc).__name__}: {exc}"
        )

    # --- sandbox -----------------------------------------------------------
    try:
        sandbox = get_sandbox()
        state, detail = await sandbox.health()
        components["sandbox"] = ComponentHealth(
            status=state, detail=detail, metadata={"mode": sandbox.name}
        )
    except Exception as exc:  # noqa: BLE001
        components["sandbox"] = ComponentHealth(
            status="unavailable", detail=f"{type(exc).__name__}: {exc}"
        )

    # --- tools -------------------------------------------------------------
    try:
        tool_report = await get_tool_registry().health()
        unhealthy = [n for n, r in tool_report.items() if r["status"] == "unavailable"]
        components["tools"] = ComponentHealth(
            status="degraded" if unhealthy else "healthy",
            detail=f"unavailable: {', '.join(unhealthy)}" if unhealthy else None,
            metadata=tool_report,
        )
    except Exception as exc:  # noqa: BLE001
        components["tools"] = ComponentHealth(
            status="unavailable", detail=f"{type(exc).__name__}: {exc}"
        )

    # --- in-process state --------------------------------------------------
    components["messages"] = ComponentHealth(
        status="healthy", metadata=get_message_service().stats()
    )
    components["documents"] = ComponentHealth(
        status="healthy", metadata=get_document_service().stats()
    )

    overall = "healthy"
    for name, component in components.items():
        if component.status == "unavailable":
            overall = "unavailable" if name in _ESSENTIAL else "degraded"
            if overall == "unavailable":
                break
        elif component.status == "degraded" and overall == "healthy":
            overall = "degraded"

    return HealthReport(status=overall, version=VERSION, components=components)


@router.get("/health/metrics")
async def metrics() -> dict[str, Any]:
    """Recent inference and agent metrics from the in-memory window."""
    return metrics_registry.snapshot()


@router.get("/config")
async def config() -> dict[str, Any]:
    """Non-secret configuration, so the UI can adapt and support can diagnose.

    Every value here is either a mode name or a limit. No keys, no URLs that
    could contain credentials.
    """
    return {
        "app_env": settings.app_env,
        "version": VERSION,
        "inference": {
            "provider": settings.llm_provider,
            "model": settings.resolved_model_name,
            "streaming": True,
            "max_output_tokens": settings.max_output_tokens,
            "max_context_tokens": settings.max_context_tokens,
        },
        "agents": {
            "max_tool_rounds": settings.max_tool_rounds,
            "max_tool_calls": settings.max_tool_calls,
            "timeout_seconds": settings.agent_timeout_seconds,
        },
        "rag": {
            "embedding_provider": settings.embedding_provider,
            "vector_store": settings.vector_store,
            "reranker": settings.reranker,
            "top_k": settings.rag_top_k,
            "rerank_top_k": settings.rag_rerank_top_k,
        },
        "documents": {
            "max_upload_size_mb": settings.max_upload_size_mb,
            "allowed_extensions": sorted(settings.allowed_extension_set),
        },
        "sandbox": {"mode": settings.python_sandbox_mode},
        "search": {"provider": settings.web_search_provider},
        "auth_required": settings.auth_enabled,
    }
