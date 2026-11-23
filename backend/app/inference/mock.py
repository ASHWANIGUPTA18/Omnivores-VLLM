"""Deterministic offline backend.

This is what makes the repository developable on a laptop with no GPU. It is
not a stub: it implements the full contract, including streamed tool-call
fragments, usage reporting and finish reasons, so the agent runtime, the
WebSocket protocol and the frontend all exercise their real code paths.

Behaviour is a pure function of the conversation, so tests assert on exact
output and two identical runs produce identical text.

Three ways it decides what to do
--------------------------------
1. An explicit directive in the user message (see below) - used by tests.
2. The conversation state: if the last message is a tool result, answer using
   it; otherwise consider calling a tool.
3. A keyword heuristic over the available tools.

Test directives
---------------
Embed any of these in a user message::

    [[mock:tool=search_knowledge_base]]        force a call to a named tool
    [[mock:tool=run_python args={"code":"1"}]] force a call with exact arguments
    [[mock:text=hello world]]                  force a plain text reply
    [[mock:error]]                             raise InferenceError
    [[mock:timeout]]                           raise InferenceTimeout
    [[mock:tokens=400]]                        pad the reply to ~400 tokens
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import settings
from app.core.exceptions import InferenceError, InferenceTimeout
from app.core.tokens import estimate_message_tokens, estimate_tokens
from app.inference.base import (
    ChatChunk,
    ChatMessage,
    ChatResult,
    InferenceClient,
    InferenceHealth,
    ModelInfo,
    ToolCallDelta,
    ToolSpec,
    Usage,
)

_DIRECTIVE_RE = re.compile(r"\[\[mock:(?P<body>[^\]]+)\]\]")

#: Which tool a message is asking for, when the model has that tool.
_TOOL_HEURISTICS: list[tuple[str, tuple[str, ...]]] = [
    (
        "run_python",
        ("calculate", "compute", "run this", "python", "script", "plot",
         "sum of", "factorial", "fibonacci", "sort", "algorithm", "code"),
    ),
    (
        "web_search",
        ("latest", "news", "current", "today", "search the web", "who is",
         "recent", "release", "announced"),
    ),
    (
        "search_knowledge_base",
        ("document", "policy", "handbook", "pdf", "uploaded", "according to",
         "what does", "leave", "notice period", "contract", "report", "says"),
    ),
]


class MockInferenceClient(InferenceClient):
    provider = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-omnivores-1",
        stream_delay: float = 0.012,
        context_length: int = 32768,
    ) -> None:
        self.model = model
        #: Per-token delay while streaming. Set to 0.0 in tests for speed.
        self.stream_delay = stream_delay
        self.context_length = context_length
        #: Every request seen, for assertions in tests.
        self.calls: list[dict[str, Any]] = []

    # --- directives --------------------------------------------------------

    @staticmethod
    def _last_user_text(messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content or ""
        return ""

    @staticmethod
    def _parse_directives(text: str) -> dict[str, str]:
        directives: dict[str, str] = {}
        for match in _DIRECTIVE_RE.finditer(text):
            body = match.group("body").strip()
            if "=" in body:
                key, _, value = body.partition("=")
                directives[key.strip()] = value.strip()
            else:
                directives[body] = "true"
        return directives

    # --- planning ----------------------------------------------------------

    def _plan(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None
    ) -> dict[str, Any]:
        """Decide this turn's output. Pure and deterministic."""
        tool_names = {t.name for t in (tools or [])}
        user_text = self._last_user_text(messages)
        directives = self._parse_directives(user_text)

        if "error" in directives:
            return {"kind": "error", "error": "mock inference error (requested)"}
        if "timeout" in directives:
            return {"kind": "timeout"}
        if "text" in directives:
            return {"kind": "text", "text": directives["text"]}

        already_called = {
            call.name
            for message in messages
            if message.role == "assistant"
            for call in message.tool_calls
        }
        last_is_tool_result = bool(messages) and messages[-1].role == "tool"

        if "tool" in directives and not last_is_tool_result:
            spec = directives["tool"]
            name, args = self._split_tool_directive(spec)
            if name in tool_names and name not in already_called:
                return {"kind": "tool", "name": name, "arguments": args}

        if last_is_tool_result:
            return {
                "kind": "text",
                "text": self._answer_from_tool_results(user_text, messages),
                "pad_tokens": int(directives.get("tokens", 0) or 0),
            }

        chosen = self._choose_tool(user_text, tool_names, already_called)
        if chosen:
            return {
                "kind": "tool",
                "name": chosen,
                "arguments": self._arguments_for(chosen, user_text),
            }

        return {
            "kind": "text",
            "text": self._plain_answer(user_text, messages),
            "pad_tokens": int(directives.get("tokens", 0) or 0),
        }

    @staticmethod
    def _split_tool_directive(spec: str) -> tuple[str, dict[str, Any]]:
        if " args=" in spec:
            name, _, raw = spec.partition(" args=")
            try:
                return name.strip(), json.loads(raw)
            except json.JSONDecodeError:
                return name.strip(), {}
        return spec.strip(), {}

    @staticmethod
    def _choose_tool(
        text: str, available: set[str], already_called: set[str]
    ) -> str | None:
        lowered = text.lower()
        for name, keywords in _TOOL_HEURISTICS:
            if name not in available or name in already_called:
                continue
            if any(keyword in lowered for keyword in keywords):
                return name
        # A question mark with a knowledge base available is a document query
        # often enough that answering it blind would make the mock useless for
        # exercising the RAG path.
        if "?" in text and "search_knowledge_base" in available:
            if "search_knowledge_base" not in already_called:
                return "search_knowledge_base"
        return None

    @staticmethod
    def _arguments_for(tool_name: str, user_text: str) -> dict[str, Any]:
        query = _DIRECTIVE_RE.sub("", user_text).strip()
        query = re.sub(r"@\w+", "", query).strip() or "the uploaded documents"
        if tool_name == "search_knowledge_base":
            return {"query": query}
        if tool_name == "web_search":
            return {"query": query}
        if tool_name == "run_python":
            return {
                "code": (
                    "# Generated by the mock model for: "
                    + query.replace("\n", " ")[:80]
                    + "\nresult = sum(range(1, 11))\nprint('result =', result)\n"
                )
            }
        return {"query": query}

    # --- answer synthesis --------------------------------------------------

    def _answer_from_tool_results(
        self, user_text: str, messages: list[ChatMessage]
    ) -> str:
        """Compose an answer that actually references the tool output.

        Quoting real retrieved text matters: it is what lets citation rendering
        and the RAG pipeline be developed and demoed without a GPU.
        """
        results = [m.content or "" for m in messages if m.role == "tool"]
        combined = "\n".join(results).strip()
        question = _DIRECTIVE_RE.sub("", user_text).strip() or "your question"

        if not combined or combined.lower().startswith("no results"):
            return (
                f"I looked into {question!r} but the tools returned nothing usable. "
                "If this should have matched an uploaded document, check that the "
                "document finished indexing."
            )

        excerpt = combined[:900].strip()
        lines = [line.strip() for line in excerpt.splitlines() if line.strip()]
        # Drop the retriever's instruction header and the source labels: the
        # substance is what follows them, and quoting the header back would
        # make every mock answer look identical.
        substance = [
            line
            for line in lines
            if not line.startswith("Excerpts from")
            and not re.match(r"^\[\d+\]\s", line)
            and not line.startswith("#")
        ]
        bullets = (substance or lines)[:5] or [excerpt[:200]]

        parts = [f"Here is what I found for {question!r}:", ""]
        parts.extend(f"- {line[:220]}" for line in bullets)
        parts.extend(
            [
                "",
                "(Generated by the mock inference backend. Set LLM_PROVIDER=vllm "
                "and point VLLM_BASE_URL at a running vLLM server for real "
                "answers.)",
            ]
        )
        return "\n".join(parts)

    def _plain_answer(self, user_text: str, messages: list[ChatMessage]) -> str:
        cleaned = _DIRECTIVE_RE.sub("", user_text).strip()
        system_prompt = next(
            (m.content or "" for m in messages if m.role == "system"), ""
        )
        persona = "an Omnivores agent"
        match = re.search(r"You are ([^.\n]+)", system_prompt)
        if match:
            persona = match.group(1).strip()

        seed = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        variant = int(seed[:2], 16) % 3

        openers = [
            f"Understood. Speaking as {persona}, here is my take.",
            f"Good question. As {persona}, I would frame it this way.",
            f"Happy to help with that. From the perspective of {persona}:",
        ]
        body = cleaned[:400] if cleaned else "your message"
        return (
            f"{openers[variant]}\n\n"
            f"You asked: {body}\n\n"
            "This response comes from the deterministic mock backend, so the "
            "reasoning is simulated. The full pipeline around it - routing, "
            "context assembly, streaming, tool dispatch and citations - is real. "
            "Switch LLM_PROVIDER to vllm to get genuine answers."
        )

    @staticmethod
    def _pad(text: str, target_tokens: int) -> str:
        if target_tokens <= 0:
            return text
        filler = " lorem ipsum dolor sit amet consectetur adipiscing elit"
        while estimate_tokens(text) < target_tokens:
            text += filler
        return text

    # --- InferenceClient ---------------------------------------------------

    def _record(self, messages: list[ChatMessage], tools: list[ToolSpec] | None,
                streaming: bool) -> None:
        self.calls.append(
            {
                "messages": [m.to_api() for m in messages],
                "tools": [t.name for t in (tools or [])],
                "streaming": streaming,
            }
        )

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
        self._record(messages, tools, streaming=False)
        plan = self._plan(messages, tools)

        if plan["kind"] == "error":
            raise InferenceError(plan["error"])
        if plan["kind"] == "timeout":
            raise InferenceTimeout("mock inference timeout (requested)")

        input_tokens = await self.count_tokens(messages)

        if plan["kind"] == "tool":
            from app.inference.base import ToolCall

            call = ToolCall(
                id=self._call_id(plan["name"], messages),
                name=plan["name"],
                arguments=json.dumps(plan["arguments"]),
            )
            return ChatResult(
                content="",
                tool_calls=[call],
                finish_reason="tool_calls",
                usage=Usage(input_tokens, estimate_tokens(call.arguments)),
                model=self.model,
            )

        text = self._pad(plan["text"], plan.get("pad_tokens", 0))
        return ChatResult(
            content=text,
            finish_reason="stop",
            usage=Usage(input_tokens, estimate_tokens(text)),
            model=self.model,
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
        self._record(messages, tools, streaming=True)
        plan = self._plan(messages, tools)

        if plan["kind"] == "error":
            raise InferenceError(plan["error"])
        if plan["kind"] == "timeout":
            await asyncio.sleep(0.05)
            raise InferenceTimeout("mock inference timeout (requested)")

        input_tokens = await self.count_tokens(messages)

        if plan["kind"] == "tool":
            arguments = json.dumps(plan["arguments"])
            call_id = self._call_id(plan["name"], messages)

            # Fragment the call the way a real server does: id and name first,
            # then arguments in pieces. This keeps StreamAccumulator honest.
            yield ChatChunk(
                tool_calls=[ToolCallDelta(index=0, id=call_id, name=plan["name"])]
            )
            for piece in self._fragments(arguments, size=18):
                if self.stream_delay:
                    await asyncio.sleep(self.stream_delay)
                yield ChatChunk(tool_calls=[ToolCallDelta(index=0, arguments=piece)])

            yield ChatChunk(
                finish_reason="tool_calls",
                usage=Usage(input_tokens, estimate_tokens(arguments)),
            )
            return

        text = self._pad(plan["text"], plan.get("pad_tokens", 0))
        for piece in self._word_chunks(text):
            if self.stream_delay:
                await asyncio.sleep(self.stream_delay)
            yield ChatChunk(content=piece)

        yield ChatChunk(
            finish_reason="stop", usage=Usage(input_tokens, estimate_tokens(text))
        )

    @staticmethod
    def _call_id(name: str, messages: list[ChatMessage]) -> str:
        digest = hashlib.sha256(
            (name + str(len(messages))).encode("utf-8")
        ).hexdigest()[:12]
        return f"call_{digest}"

    @staticmethod
    def _fragments(text: str, size: int) -> list[str]:
        return [text[i : i + size] for i in range(0, len(text), size)] or [""]

    @staticmethod
    def _word_chunks(text: str) -> list[str]:
        """Split into word-ish pieces so deltas look like real token streams."""
        pieces = re.findall(r"\s*\S+", text)
        return pieces or [text]

    async def count_tokens(self, messages: list[ChatMessage]) -> int:
        return sum(estimate_message_tokens(m.role, m.content) for m in messages)

    async def health(self) -> InferenceHealth:
        return InferenceHealth(
            status="healthy",
            detail="mock backend - responses are synthetic",
            model=self.model,
            latency_ms=0.0,
        )

    async def model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self.model,
            provider="mock",
            context_length=self.context_length,
            supports_tools=True,
            supports_streaming=True,
            extra={"deterministic": True, "app_env": settings.app_env},
        )
