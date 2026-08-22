"""Inference client factory and process-wide singleton.

This module is the only place in the application that maps a provider name to a
concrete class. Everything else receives an ``InferenceClient``.

The client is created lazily and cached, because the HTTP transport underneath
it holds a connection pool that should be shared across requests.
"""

from __future__ import annotations

import asyncio

from app.core.config import settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.inference.base import InferenceClient

logger = get_logger(__name__)

_client: InferenceClient | None = None
_lock = asyncio.Lock()


def create_inference_client(provider: str | None = None) -> InferenceClient:
    """Build a fresh client for ``provider`` (defaults to ``LLM_PROVIDER``).

    Imports are deferred so that selecting one backend never imports another's
    dependencies.
    """
    name = (provider or settings.llm_provider).lower()

    if name == "mock":
        from app.inference.mock import MockInferenceClient

        return MockInferenceClient()

    if name == "vllm":
        from app.inference.vllm import VLLMClient

        return VLLMClient()

    if name == "openai":
        from app.inference.openai import OpenAIClient

        return OpenAIClient()

    raise ConfigurationError(
        f"unknown LLM_PROVIDER '{name}'",
        user_message="The configured inference provider is not supported.",
        details={"supported": ["mock", "vllm", "openai"]},
    )


async def get_inference_client() -> InferenceClient:
    """Return the shared client, creating it on first use."""
    global _client
    if _client is None:
        async with _lock:
            if _client is None:
                _client = create_inference_client()
                logger.info(
                    "inference client ready",
                    extra={
                        "provider": _client.provider,
                        "model": settings.resolved_model_name,
                    },
                )
    return _client


def set_inference_client(client: InferenceClient | None) -> None:
    """Override the shared client. Used by tests and by the app lifespan."""
    global _client
    _client = client


async def close_inference_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
