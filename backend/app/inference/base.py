"""The inference contract.

Everything above this module - agents, tools, RAG, the WebSocket layer - talks
to an ``InferenceClient`` and nothing else. No application code imports vLLM,
the OpenAI SDK, or a tokenizer. Swapping the backend is a config change.

The message and tool-call shapes mirror the OpenAI chat-completions schema
because that is the interface vLLM exposes, which makes the vLLM and OpenAI
backends thin and keeps the mock backend honest about what a real model does.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


# --- Messages --------------------------------------------------------------


@dataclass
class ToolCall:
    """A fully assembled tool call emitted by the model."""

    id: str
    name: str
    #: Raw JSON string exactly as the model produced it. Kept verbatim so that
    #: a malformed call can be reported back to the model for repair.
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        """Parse ``arguments``. Raises ``ValueError`` on malformed JSON."""
        if not self.arguments or not self.arguments.strip():
            return {}
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool arguments are not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be a JSON object")
        return value

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class ChatMessage:
    role: Role
    content: str | None = None
    #: Assistant messages only.
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Tool messages only - the id of the call being answered.
    tool_call_id: str | None = None
    #: Optional speaker label. Used to keep multi-user channel history legible.
    name: str | None = None

    def to_api(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        # The API requires `content` to be present even when null on an
        # assistant message that only carries tool calls.
        payload["content"] = self.content if self.content is not None else ""
        if self.tool_calls:
            payload["tool_calls"] = [tc.to_api() for tc in self.tool_calls]
            if not self.content:
                payload["content"] = None
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str, name: str | None = None) -> ChatMessage:
        return cls(role="user", content=content, name=name)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> ChatMessage:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str | None = None) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ToolSpec:
    """A tool as advertised to the model."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_api(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --- Results ---------------------------------------------------------------


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ToolCallDelta:
    """A fragment of a streamed tool call.

    Providers stream tool calls piecewise: the index identifies which call the
    fragment belongs to, and ``arguments`` fragments must be concatenated in
    arrival order. ``StreamAccumulator`` handles the reassembly.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass
class ChatChunk:
    """One streamed increment."""

    content: str = ""
    tool_calls: list[ToolCallDelta] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None


@dataclass
class ChatResult:
    """A completed model turn."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class StreamAccumulator:
    """Reassembles a chunk stream into a ``ChatResult``.

    Kept here rather than in each backend so that the mock client exercises the
    exact same reassembly path the real ones do - a streaming bug in tool-call
    reconstruction will show up in CI on a laptop.
    """

    def __init__(self, model: str = "") -> None:
        self.model = model
        self._content: list[str] = []
        self._tool_fragments: dict[int, dict[str, Any]] = {}
        self._finish_reason: str | None = None
        self._usage: Usage | None = None

    def add(self, chunk: ChatChunk) -> None:
        if chunk.content:
            self._content.append(chunk.content)
        for delta in chunk.tool_calls:
            slot = self._tool_fragments.setdefault(
                delta.index, {"id": None, "name": None, "arguments": []}
            )
            if delta.id:
                slot["id"] = delta.id
            if delta.name:
                slot["name"] = delta.name
            if delta.arguments:
                slot["arguments"].append(delta.arguments)
        if chunk.finish_reason:
            self._finish_reason = chunk.finish_reason
        if chunk.usage:
            self._usage = chunk.usage

    @property
    def content(self) -> str:
        return "".join(self._content)

    def result(self) -> ChatResult:
        tool_calls: list[ToolCall] = []
        for index in sorted(self._tool_fragments):
            slot = self._tool_fragments[index]
            name = slot["name"]
            if not name:
                # A fragment stream that never named a function is unusable;
                # dropping it is better than inventing a tool name.
                continue
            tool_calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=name,
                    arguments="".join(slot["arguments"]) or "{}",
                )
            )
        return ChatResult(
            content=self.content,
            tool_calls=tool_calls,
            finish_reason=self._finish_reason,
            usage=self._usage or Usage(),
            model=self.model,
        )


# --- Health / model info ---------------------------------------------------


@dataclass
class ModelInfo:
    id: str
    provider: str
    context_length: int | None = None
    supports_tools: bool = True
    supports_streaming: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "context_length": self.context_length,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
            **self.extra,
        }


@dataclass
class InferenceHealth:
    status: Literal["healthy", "degraded", "unavailable"]
    detail: str | None = None
    model: str | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms else None,
        }


# --- The client ------------------------------------------------------------


class InferenceClient(ABC):
    """Backend-agnostic language model access."""

    #: Short provider name, used in metrics and health output.
    provider: str = "base"

    @abstractmethod
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
        """Run one non-streaming completion."""

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Run one streaming completion, yielding increments as they arrive."""

    @abstractmethod
    async def count_tokens(self, messages: list[ChatMessage]) -> int:
        """Count prompt tokens. Backends fall back to estimation."""

    @abstractmethod
    async def health(self) -> InferenceHealth:
        """Check whether the backend can serve requests right now."""

    @abstractmethod
    async def model_info(self) -> ModelInfo:
        """Describe the served model."""

    async def close(self) -> None:
        """Release transport resources. Safe to call more than once."""
        return None
