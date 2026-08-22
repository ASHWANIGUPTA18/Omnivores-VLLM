"""Hosted OpenAI backend.

Present as a bridge and a comparison baseline, not as the target. Omnivores is
local-first: the production path is vLLM. This client exists so that the
application can be validated against a known-good tool-calling implementation
while a local model is being selected, and so benchmarks have something to
compare local throughput against.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.inference.base import ModelInfo
from app.inference.openai_compatible import OpenAICompatibleClient

#: Context windows for the models we name in configuration. Only used as a
#: fallback hint - the API does not report context length.
_KNOWN_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4-turbo": 128_000,
}


class OpenAIClient(OpenAICompatibleClient):
    provider = "openai"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url or settings.openai_base_url,
            api_key=api_key if api_key is not None else settings.openai_api_key,
            model=model or settings.openai_model,
            **kwargs,
        )

    async def model_info(self) -> ModelInfo:
        model_id = self._resolved_model or self.model
        return ModelInfo(
            id=model_id,
            provider="openai",
            context_length=_KNOWN_CONTEXT_WINDOWS.get(model_id),
            supports_tools=True,
            supports_streaming=True,
        )
