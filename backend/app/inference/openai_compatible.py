"""Shared transport for OpenAI-compatible chat-completions servers.

Both supported real backends speak this protocol: vLLM exposes an
OpenAI-compatible ``/v1`` surface, and the hosted OpenAI API is the protocol by
definition. Keeping the HTTP, SSE-parsing, retry and error-mapping logic in one
place means ``vllm.py`` and ``openai.py`` only carry what genuinely differs
between them.

We speak the wire protocol directly over httpx rather than pulling in the
OpenAI SDK. It is a small, stable protocol, and the dependency would otherwise
be mandatory on a machine that may never call OpenAI.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx

from app.core.config import settings
from app.core.exceptions import (
    ContextLengthExceeded,
    InferenceError,
    InferenceTimeout,
    InferenceUnavailable,
    ModelNotLoaded,
)
from app.core.logging import get_logger
from app.core.tokens import estimate_message_tokens
from app.inference.base import (
    ChatChunk,
    ChatMessage,
    ChatResult,
    InferenceClient,
    InferenceHealth,
    ModelInfo,
    StreamAccumulator,
    ToolCall,
    ToolCallDelta,
    ToolSpec,
    Usage,
)

logger = get_logger(__name__)

#: Status codes worth retrying. 429 and 5xx are transient; 503 in particular is
#: what vLLM returns while a model is still loading.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleClient(InferenceClient):
    provider = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = "",
        timeout: float | None = None,
        connect_timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout if timeout is not None else settings.inference_timeout_seconds
        self.connect_timeout = (
            connect_timeout
            if connect_timeout is not None
            else settings.inference_connect_timeout_seconds
        )
        self.max_retries = (
            max_retries if max_retries is not None else settings.inference_max_retries
        )
        self._default_headers = default_headers or {}
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        #: Populated lazily from GET /models when MODEL_NAME is not configured.
        self._resolved_model: str | None = model or None

    # --- transport ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._default_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        base_url=self.base_url,
                        headers=self._headers(),
                        timeout=httpx.Timeout(
                            self.timeout,
                            connect=self.connect_timeout,
                            # Streaming reads must not be cut off by the overall
                            # budget between tokens; the agent runtime enforces
                            # the wall-clock limit instead.
                            read=self.timeout,
                        ),
                        follow_redirects=True,
                    )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- error mapping -----------------------------------------------------

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            return (response.text or "")[:500]
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)
            if isinstance(err, str):
                return err
            if "message" in payload:
                return str(payload["message"])
            if "detail" in payload:
                return str(payload["detail"])
        return json.dumps(payload)[:500]

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        message = self._extract_error_message(response)
        lowered = message.lower()

        if response.status_code == 400 and (
            "context" in lowered or "maximum context length" in lowered or "too long" in lowered
        ):
            raise ContextLengthExceeded(message, details={"provider": self.provider})
        if response.status_code in (401, 403):
            raise InferenceError(
                f"authentication rejected by {self.provider}: {message}",
                user_message="The inference server rejected our credentials.",
            )
        if response.status_code == 404 and "model" in lowered:
            raise ModelNotLoaded(
                f"model not served by {self.provider}: {message}",
                user_message=(
                    "The configured model is not loaded on the inference server. "
                    "Check MODEL_NAME against GET /v1/models."
                ),
            )
        if response.status_code == 503:
            raise ModelNotLoaded(message)
        if response.status_code == 429:
            raise InferenceUnavailable(
                f"rate limited by {self.provider}: {message}",
                user_message="The inference server is overloaded. Try again shortly.",
            )
        if response.status_code >= 500:
            raise InferenceUnavailable(f"{self.provider} returned {response.status_code}: {message}")
        raise InferenceError(f"{self.provider} returned {response.status_code}: {message}")

    async def _request_with_retry(
        self, method: str, path: str, *, json_body: dict[str, Any] | None = None
    ) -> httpx.Response:
        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(method, path, json=json_body)
                if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    last_error = InferenceUnavailable(
                        f"{self.provider} returned {response.status_code}"
                    )
                    await self._backoff(attempt)
                    continue
                self._raise_for_status(response)
                return response
            except httpx.TimeoutException as exc:
                last_error = InferenceTimeout(
                    f"{self.provider} timed out after {self.timeout}s"
                )
                if attempt >= self.max_retries:
                    raise last_error from exc
                await self._backoff(attempt)
            except httpx.RequestError as exc:
                last_error = InferenceUnavailable(
                    f"cannot reach {self.provider} at {self.base_url}: {exc}",
                    user_message=(
                        "AI inference is unavailable - the model server could not "
                        "be reached."
                    ),
                )
                if attempt >= self.max_retries:
                    raise last_error from exc
                await self._backoff(attempt)

        raise last_error or InferenceUnavailable("request failed")

    @staticmethod
    async def _backoff(attempt: int) -> None:
        await asyncio.sleep(min(0.5 * (2**attempt), 4.0))

    # --- payload building --------------------------------------------------

    async def _resolve_model(self) -> str:
        """Return the model id to send.

        When MODEL_NAME is unset we ask the server what it is serving. vLLM
        serves exactly one model per process, so the first entry is unambiguous.
        """
        if self._resolved_model:
            return self._resolved_model
        try:
            response = await self._request_with_retry("GET", "/models")
            data = response.json().get("data") or []
            if data:
                self._resolved_model = str(data[0].get("id"))
                logger.info(
                    "auto-detected model from inference server",
                    extra={"model": self._resolved_model, "provider": self.provider},
                )
                return self._resolved_model
        except Exception as exc:  # noqa: BLE001 - fall through to a clear error
            logger.warning("could not auto-detect model: %s", exc)
        raise ModelNotLoaded(
            "no model configured and none advertised by the server",
            user_message=(
                "No model is configured. Set MODEL_NAME to a model id served by "
                "the inference server."
            ),
        )

    async def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: list[str] | None,
        stream: bool,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": await self._resolve_model(),
            "messages": [m.to_api() for m in messages],
            "temperature": temperature if temperature is not None else settings.temperature,
            "top_p": top_p if top_p is not None else settings.top_p,
            "max_tokens": max_tokens if max_tokens is not None else settings.max_output_tokens,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = [t.to_api() for t in tools]
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        if stream:
            # vLLM and OpenAI both report usage on the final chunk when asked.
            payload["stream_options"] = {"include_usage": True}
        return payload

    # --- parsing -----------------------------------------------------------

    @staticmethod
    def _parse_usage(payload: dict[str, Any]) -> Usage:
        usage = payload.get("usage") or {}
        return Usage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )

    @staticmethod
    def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index, item in enumerate(raw or []):
            function = item.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{index}"),
                    name=str(name),
                    arguments=function.get("arguments") or "{}",
                )
            )
        return calls

    @staticmethod
    def _parse_tool_call_deltas(raw: list[dict[str, Any]] | None) -> list[ToolCallDelta]:
        deltas: list[ToolCallDelta] = []
        for position, item in enumerate(raw or []):
            function = item.get("function") or {}
            deltas.append(
                ToolCallDelta(
                    index=int(item.get("index", position)),
                    id=item.get("id"),
                    name=function.get("name"),
                    arguments=function.get("arguments") or "",
                )
            )
        return deltas

    # --- InferenceClient ---------------------------------------------------

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload = await self._build_payload(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=False,
            response_format=response_format,
        )
        response = await self._request_with_retry("POST", "/chat/completions", json_body=payload)
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise InferenceError("inference server returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        return ChatResult(
            content=message.get("content") or "",
            tool_calls=self._parse_tool_calls(message.get("tool_calls")),
            finish_reason=choice.get("finish_reason"),
            usage=self._parse_usage(body),
            model=str(body.get("model") or self._resolved_model or ""),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        payload = await self._build_payload(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=True,
        )
        client = await self._get_client()

        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    # The body has not been read yet on a streaming response.
                    await response.aread()
                    self._raise_for_status(response)

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        # SSE comment / keepalive.
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("skipping malformed SSE payload: %.120s", data)
                        continue

                    chunk = self._chunk_from_event(event)
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise InferenceTimeout(f"{self.provider} stream timed out") from exc
        except httpx.RequestError as exc:
            raise InferenceUnavailable(
                f"stream to {self.provider} failed: {exc}",
                user_message="Lost connection to the inference server mid-response.",
            ) from exc

    def _chunk_from_event(self, event: dict[str, Any]) -> ChatChunk | None:
        choices = event.get("choices") or []
        usage = self._parse_usage(event) if event.get("usage") else None

        if not choices:
            # A usage-only final frame (stream_options.include_usage).
            return ChatChunk(usage=usage) if usage else None

        choice = choices[0]
        delta = choice.get("delta") or {}
        return ChatChunk(
            content=delta.get("content") or "",
            tool_calls=self._parse_tool_call_deltas(delta.get("tool_calls")),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )

    async def collect(
        self, messages: list[ChatMessage], **kwargs: Any
    ) -> ChatResult:
        """Stream a completion but return it as a single result.

        Useful in tests and benchmarks that want to exercise the streaming path
        without handling increments.
        """
        accumulator = StreamAccumulator(model=self._resolved_model or self.model)
        async for chunk in self.stream(messages, **kwargs):
            accumulator.add(chunk)
        return accumulator.result()

    async def count_tokens(self, messages: list[ChatMessage]) -> int:
        return sum(estimate_message_tokens(m.role, m.content) for m in messages)

    async def health(self) -> InferenceHealth:
        import time

        started = time.perf_counter()
        try:
            client = await self._get_client()
            response = await client.get("/models", timeout=self.connect_timeout)
            latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code >= 400:
                return InferenceHealth(
                    status="unavailable",
                    detail=f"GET /models returned {response.status_code}",
                    latency_ms=latency_ms,
                )
            data = response.json().get("data") or []
            served = [str(m.get("id")) for m in data]
            if self.model and self.model not in served:
                return InferenceHealth(
                    status="degraded",
                    detail=(
                        f"configured model '{self.model}' is not served; "
                        f"available: {', '.join(served) or 'none'}"
                    ),
                    model=self.model,
                    latency_ms=latency_ms,
                )
            return InferenceHealth(
                status="healthy",
                model=self.model or (served[0] if served else None),
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return InferenceHealth(
                status="unavailable",
                detail=f"{type(exc).__name__}: {exc}",
                model=self.model or None,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    async def model_info(self) -> ModelInfo:
        model_id = self._resolved_model or self.model or "unknown"
        info = ModelInfo(id=model_id, provider=self.provider)
        try:
            response = await self._request_with_retry("GET", "/models")
            for entry in response.json().get("data") or []:
                if str(entry.get("id")) == model_id or not self._resolved_model:
                    info.id = str(entry.get("id"))
                    # vLLM reports the served context window here.
                    length = entry.get("max_model_len") or entry.get("context_window")
                    if length:
                        info.context_length = int(length)
                    break
        except Exception as exc:  # noqa: BLE001 - info is best-effort
            info.extra["detail"] = f"model metadata unavailable: {exc}"
        return info
