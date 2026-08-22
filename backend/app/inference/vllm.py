"""vLLM backend.

vLLM serves an OpenAI-compatible API, so almost everything lives in
``OpenAICompatibleClient``. What differs and belongs here:

* vLLM serves exactly one model per process, so ``MODEL_NAME`` may be left
  empty and auto-detected from ``GET /v1/models``.
* vLLM exposes ``POST /tokenize``, which gives an exact prompt token count
  using the served model's own tokenizer instead of our estimator.
* Health checks report the served context window, which the context manager
  uses to cap ``MAX_CONTEXT_TOKENS``.

Tool calling requires the server to be started with ``--enable-auto-tool-choice``
and a ``--tool-call-parser`` appropriate for the model. That is a deployment
concern, handled in ``inference/start-vllm.sh``; this client only detects when
it is missing and says so clearly rather than silently returning prose where a
tool call was expected.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.inference.base import ChatMessage, InferenceHealth, ModelInfo
from app.inference.openai_compatible import OpenAICompatibleClient

logger = get_logger(__name__)


class VLLMClient(OpenAICompatibleClient):
    provider = "vllm"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url or settings.vllm_base_url,
            api_key=api_key if api_key is not None else settings.vllm_api_key,
            model=model if model is not None else settings.model_name,
            **kwargs,
        )
        self._context_length: int | None = None
        self._tokenize_supported: bool | None = None

    async def count_tokens(self, messages: list[ChatMessage]) -> int:
        """Exact count via vLLM's ``/tokenize``, with graceful degradation.

        The endpoint sits at the server root, not under ``/v1``, so we address
        it relative to the base URL's parent.
        """
        if self._tokenize_supported is False:
            return await super().count_tokens(messages)

        try:
            client = await self._get_client()
            payload = {
                "model": await self._resolve_model(),
                "messages": [m.to_api() for m in messages],
            }
            # base_url ends in /v1; /tokenize is one level up.
            response = await client.post(
                self._root_path("/tokenize"),
                json=payload,
                timeout=self.connect_timeout,
            )
            if response.status_code == 404:
                self._tokenize_supported = False
                logger.debug("vLLM /tokenize not available, using estimator")
                return await super().count_tokens(messages)
            response.raise_for_status()
            body = response.json()
            count = body.get("count")
            if count is None and isinstance(body.get("tokens"), list):
                count = len(body["tokens"])
            if count is not None:
                self._tokenize_supported = True
                return int(count)
        except Exception as exc:  # noqa: BLE001 - counting must never break a request
            logger.debug("vLLM /tokenize failed, using estimator: %s", exc)
            self._tokenize_supported = False

        return await super().count_tokens(messages)

    def _root_path(self, path: str) -> str:
        """Absolute URL for a server-root endpoint such as ``/tokenize``.

        Returned absolute rather than relative because httpx joins relative
        paths onto the ``/v1`` base, which would produce ``/v1/tokenize``.
        """
        base = self.base_url
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base.rstrip("/") + "/" + path.lstrip("/")

    async def health(self) -> InferenceHealth:
        health = await super().health()
        if health.status == "unavailable":
            health.detail = (
                f"{health.detail or 'unreachable'} - is vLLM running at "
                f"{self.base_url}? See inference/README.md"
            )
        return health

    async def model_info(self) -> ModelInfo:
        info = await super().model_info()
        info.provider = "vllm"
        if info.context_length:
            self._context_length = info.context_length
        info.extra.setdefault(
            "note",
            "Tool calling requires --enable-auto-tool-choice and a "
            "--tool-call-parser matching this model.",
        )
        return info

    @property
    def served_context_length(self) -> int | None:
        """The context window reported by the server, if known.

        ``None`` until ``model_info()`` has been called at least once, which
        happens during startup health checks.
        """
        return self._context_length
