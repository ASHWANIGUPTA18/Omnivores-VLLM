"""The agent runtime.

One loop, shared by every agent:

    build context
        -> stream from the model
        -> did it call tools?
             yes: execute them, append results, loop
             no : done

Bounded on four axes so a confused model cannot run forever or bankrupt a
context window: ``MAX_TOOL_ROUNDS``, ``MAX_TOOL_CALLS``, ``TOOL_TIMEOUT_SECONDS``
and ``AGENT_TIMEOUT_SECONDS``.

Two design points worth stating explicitly:

**Streaming is not optional.** Text reaches the user as it is generated, which
is what makes a 40-token/second local model feel usable. Tool calls stream too,
and are reassembled by ``StreamAccumulator``.

**Running out of rounds is not a failure.** When the loop hits its limit, the
runtime makes one final call with tools withheld, forcing the model to answer
with what it has. The user gets an answer rather than a timeout.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.agents.context import BuiltContext, ContextManager, default_context_manager
from app.agents.registry import AgentDefinition
from app.core.config import settings
from app.core.exceptions import (
    AgentTimeout,
    InferenceError,
    OmnivoresError,
    RetrievalError,
)
from app.core.logging import get_logger
from app.core.metrics import AgentMetrics, InferenceMetrics, metrics_registry
from app.core.models import Citation, Message, ToolInvocation
from app.inference.base import (
    ChatMessage,
    ChatResult,
    InferenceClient,
    StreamAccumulator,
    ToolCall,
    ToolSpec,
)
from app.inference.client import get_inference_client
from app.rag.retriever import Retriever, default_retriever
from app.tools.base import ToolContext
from app.tools.registry import ToolExecutor, ToolRegistry, get_tool_registry

logger = get_logger(__name__)

#: Called with each WebSocket event the run produces.
EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]
#: Called with each text increment.
DeltaEmitter = Callable[[str], Awaitable[None]]


@dataclass
class AgentRunResult:
    content: str = ""
    citations: list[Citation] = field(default_factory=list)
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    metrics: AgentMetrics | None = None
    error: OmnivoresError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class AgentRuntime:
    def __init__(
        self,
        *,
        inference_client: InferenceClient | None = None,
        tool_registry: ToolRegistry | None = None,
        context_manager: ContextManager | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self._inference_client = inference_client
        self._tool_registry = tool_registry
        self._context_manager = context_manager or default_context_manager
        self._retriever = retriever or default_retriever

    async def _client(self) -> InferenceClient:
        return self._inference_client or await get_inference_client()

    @property
    def tools(self) -> ToolRegistry:
        return self._tool_registry or get_tool_registry()

    # --- entry point -------------------------------------------------------

    async def run(
        self,
        *,
        definition: AgentDefinition,
        channel_id: str,
        message_id: str,
        user_message: Message,
        history: list[Message],
        document_ids: list[str] | None = None,
        on_event: EventEmitter | None = None,
        on_delta: DeltaEmitter | None = None,
    ) -> AgentRunResult:
        """Run one agent turn. Never raises - failures come back on the result."""
        timeout = settings.agent_timeout_seconds
        try:
            return await asyncio.wait_for(
                self._run(
                    definition=definition,
                    channel_id=channel_id,
                    message_id=message_id,
                    user_message=user_message,
                    history=history,
                    document_ids=document_ids or [],
                    on_event=on_event,
                    on_delta=on_delta,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            error = AgentTimeout(
                f"agent '{definition.id}' exceeded {timeout:.0f}s",
                user_message=(
                    f"@{definition.id} took longer than {timeout:.0f}s and was "
                    "stopped."
                ),
            )
            logger.warning("agent timeout", extra={"agent_id": definition.id})
            return AgentRunResult(error=error)
        except asyncio.CancelledError:
            raise
        except OmnivoresError as exc:
            logger.error("agent failed: %s", exc, extra={"agent_id": definition.id})
            return AgentRunResult(error=exc)
        except Exception as exc:  # noqa: BLE001 - the loop must not leak
            logger.exception("unexpected agent failure")
            return AgentRunResult(
                error=OmnivoresError(
                    f"{type(exc).__name__}: {exc}",
                    user_message="The agent hit an unexpected error.",
                )
            )

    # --- the loop ----------------------------------------------------------

    async def _run(
        self,
        *,
        definition: AgentDefinition,
        channel_id: str,
        message_id: str,
        user_message: Message,
        history: list[Message],
        document_ids: list[str],
        on_event: EventEmitter | None,
        on_delta: DeltaEmitter | None,
    ) -> AgentRunResult:
        metrics = AgentMetrics(
            agent_id=definition.id, channel_id=channel_id, message_id=message_id
        )
        result = AgentRunResult(metrics=metrics)

        async def emit(event: dict[str, Any]) -> None:
            if on_event:
                await on_event(event)

        from app.core import events as ev

        await emit(ev.agent_started(definition.id, channel_id, message_id))

        client = await self._client()
        executor = ToolExecutor(self.tools)
        allowed_tools = set(definition.tools)
        tool_specs: list[ToolSpec] = self.tools.specs_for(definition.tools)

        tool_context = ToolContext(
            channel_id=channel_id,
            message_id=message_id,
            agent_id=definition.id,
            user_id=user_message.author.id,
            document_ids=document_ids,
        )

        # --- optional up-front retrieval -----------------------------------
        document_context = ""
        if definition.auto_retrieve or document_ids:
            document_context = await self._auto_retrieve(
                definition=definition,
                query=user_message.content,
                channel_id=channel_id,
                document_ids=document_ids,
                result=result,
                metrics=metrics,
            )

        # --- build the prompt ----------------------------------------------
        model_context_length = await self._model_context_length(client)
        built: BuiltContext = self._context_manager.build(
            system_prompt=definition.full_system_prompt(),
            history=history,
            current_message=user_message,
            agent_id=definition.id,
            document_context=document_context,
            tool_specs=tool_specs,
            model_context_length=model_context_length,
        )
        metrics.context_tokens = built.total_tokens
        conversation: list[ChatMessage] = list(built.messages)

        max_rounds = definition.max_tool_rounds or settings.max_tool_rounds
        temperature = (
            definition.temperature
            if definition.temperature is not None
            else settings.temperature
        )
        max_output = definition.max_output_tokens or settings.max_output_tokens

        content_parts: list[str] = []
        tool_calls_made = 0

        for round_index in range(1, max_rounds + 1):
            metrics.rounds = round_index
            await emit(
                ev.agent_thinking(definition.id, channel_id, message_id, round_index)
            )

            # Withhold tools once the call budget is spent, so the next turn
            # must be an answer.
            offer_tools = bool(tool_specs) and tool_calls_made < settings.max_tool_calls

            turn = await self._one_turn(
                client=client,
                conversation=conversation,
                tools=tool_specs if offer_tools else None,
                temperature=temperature,
                max_tokens=max_output,
                agent_id=definition.id,
                metrics=metrics,
                on_delta=on_delta,
                emit_leading_newline=bool(content_parts),
            )

            if turn.content:
                content_parts.append(turn.content)

            if not turn.tool_calls:
                break

            # Record the assistant turn verbatim, including its tool calls -
            # providers reject a tool result whose call is not in the history.
            conversation.append(
                ChatMessage.assistant(
                    content=turn.content or None, tool_calls=turn.tool_calls
                )
            )

            remaining_calls = max(0, settings.max_tool_calls - tool_calls_made)
            calls = turn.tool_calls[:remaining_calls]
            skipped = turn.tool_calls[remaining_calls:]

            for call in calls:
                tool_calls_made += 1
                await self._execute_tool_call(
                    call=call,
                    executor=executor,
                    allowed_tools=allowed_tools,
                    tool_context=tool_context,
                    conversation=conversation,
                    result=result,
                    metrics=metrics,
                    channel_id=channel_id,
                    message_id=message_id,
                    agent_id=definition.id,
                    emit=emit,
                )

            for call in skipped:
                conversation.append(
                    ChatMessage.tool(
                        call.id,
                        f"Not executed: this turn's limit of {settings.max_tool_calls} "
                        "tool calls has been reached. Answer with what you have.",
                        name=call.name,
                    )
                )

            if round_index == max_rounds:
                # Out of rounds with tool results pending: force a final answer
                # rather than returning the user a bare tool trace.
                logger.info(
                    "agent hit the round limit, forcing a final answer",
                    extra={"agent_id": definition.id, "rounds": max_rounds},
                )
                conversation.append(
                    ChatMessage.system(
                        "You have used all available tool calls. Answer the user "
                        "now using the information you have gathered. Do not "
                        "request any more tools."
                    )
                )
                final = await self._one_turn(
                    client=client,
                    conversation=conversation,
                    tools=None,
                    temperature=temperature,
                    max_tokens=max_output,
                    agent_id=definition.id,
                    metrics=metrics,
                    on_delta=on_delta,
                    emit_leading_newline=bool(content_parts),
                )
                if final.content:
                    content_parts.append(final.content)

        result.content = "\n\n".join(part for part in content_parts if part.strip()).strip()

        if not result.content:
            result.content = (
                "I was not able to produce an answer for that. Try rephrasing, or "
                "check the server logs if this keeps happening."
            )

        result.citations = _dedupe_citations(result.citations)
        metrics.finish()
        metrics_registry.record_agent_run(metrics)
        logger.info("agent run complete: %s", metrics.summary_line())

        await emit(
            ev.agent_completed(
                definition.id, channel_id, message_id, result.citations, metrics.to_dict()
            )
        )
        return result

    # --- one model turn ----------------------------------------------------

    async def _one_turn(
        self,
        *,
        client: InferenceClient,
        conversation: list[ChatMessage],
        tools: list[ToolSpec] | None,
        temperature: float,
        max_tokens: int,
        agent_id: str,
        metrics: AgentMetrics,
        on_delta: DeltaEmitter | None,
        emit_leading_newline: bool,
    ) -> ChatResult:
        """Stream one completion, forwarding text deltas as they arrive."""
        inference_metrics = InferenceMetrics(
            model=settings.resolved_model_name,
            provider=client.provider,
            agent_id=agent_id,
            streaming=True,
        )
        accumulator = StreamAccumulator()
        emitted_any = False

        try:
            async for chunk in client.stream(
                conversation,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                accumulator.add(chunk)

                if chunk.content:
                    inference_metrics.mark_first_token()
                    if on_delta:
                        text = chunk.content
                        if emit_leading_newline and not emitted_any:
                            # Separate this round's prose from the previous
                            # round's in the rendered message.
                            text = "\n\n" + text
                        await on_delta(text)
                    emitted_any = True
                elif chunk.tool_calls:
                    inference_metrics.mark_first_token()

                if chunk.usage:
                    inference_metrics.input_tokens = chunk.usage.input_tokens
                    inference_metrics.output_tokens = chunk.usage.output_tokens
                    inference_metrics.tokens_reported = True

        except InferenceError as exc:
            inference_metrics.error = exc.code
            inference_metrics.finish()
            metrics.inference_calls.append(inference_metrics)
            metrics_registry.record_inference(inference_metrics)
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise to our error type
            inference_metrics.error = type(exc).__name__
            inference_metrics.finish()
            metrics.inference_calls.append(inference_metrics)
            metrics_registry.record_inference(inference_metrics)
            raise InferenceError(f"{type(exc).__name__}: {exc}") from exc

        turn = accumulator.result()
        inference_metrics.finish_reason = turn.finish_reason

        if not inference_metrics.tokens_reported:
            # Providers that do not report usage still need a number, or every
            # throughput metric reads zero.
            from app.core.tokens import estimate_tokens

            inference_metrics.input_tokens = await client.count_tokens(conversation)
            inference_metrics.output_tokens = estimate_tokens(turn.content) + sum(
                estimate_tokens(c.arguments) for c in turn.tool_calls
            )

        inference_metrics.finish()
        metrics.inference_calls.append(inference_metrics)
        metrics_registry.record_inference(inference_metrics)

        if turn.finish_reason == "length":
            logger.warning(
                "generation hit max_tokens (%d) - the answer is cut off", max_tokens
            )
        return turn

    # --- tool dispatch -----------------------------------------------------

    async def _execute_tool_call(
        self,
        *,
        call: ToolCall,
        executor: ToolExecutor,
        allowed_tools: set[str],
        tool_context: ToolContext,
        conversation: list[ChatMessage],
        result: AgentRunResult,
        metrics: AgentMetrics,
        channel_id: str,
        message_id: str,
        agent_id: str,
        emit: EventEmitter,
    ) -> None:
        from app.core import events as ev

        try:
            arguments = call.parsed_arguments()
        except ValueError as exc:
            # Malformed JSON from the model: tell it precisely, and let it retry
            # on the next round.
            conversation.append(
                ChatMessage.tool(
                    call.id,
                    f"Could not parse your arguments: {exc}. Send valid JSON matching "
                    "the tool's schema.",
                    name=call.name,
                )
            )
            metrics.tool_calls.append(
                _failed_tool_metric(call.name, "invalid_json")
            )
            return

        invocation = ToolInvocation(
            id=call.id, name=call.name, arguments=arguments, status="running"
        )
        result.tool_invocations.append(invocation)
        await emit(
            ev.agent_tool_started(agent_id, channel_id, message_id, invocation)
        )

        tool_result, tool_metrics = await executor.execute(
            call.name, arguments, tool_context, allowed=allowed_tools
        )
        metrics.tool_calls.append(tool_metrics)

        invocation.status = "completed" if tool_result.ok else "failed"
        invocation.duration_ms = int(tool_metrics.duration_ms)
        invocation.result_preview = (tool_result.summary or tool_result.content)[:200]
        if not tool_result.ok:
            invocation.error = tool_result.content[:300]

        if tool_result.citations:
            result.citations.extend(tool_result.citations)
            metrics.retrieved_chunks += len(tool_result.citations)

        conversation.append(
            ChatMessage.tool(call.id, tool_result.content, name=call.name)
        )
        await emit(
            ev.agent_tool_completed(agent_id, channel_id, message_id, invocation)
        )

    # --- retrieval ---------------------------------------------------------

    async def _auto_retrieve(
        self,
        *,
        definition: AgentDefinition,
        query: str,
        channel_id: str,
        document_ids: list[str],
        result: AgentRunResult,
        metrics: AgentMetrics,
    ) -> str:
        """Fetch document context before the first model call.

        Used when the agent is document-first (``auto_retrieve``) or when the
        user explicitly scoped the question to documents. Failure here is not
        fatal: the agent still has ``search_knowledge_base`` and can try again.
        """
        started = time.perf_counter()
        try:
            retrieval = await self._retriever.retrieve(
                query,
                document_ids=document_ids or None,
                channel_id=channel_id,
            )
        except RetrievalError as exc:
            logger.error("auto-retrieval failed: %s", exc)
            return ""

        metrics.retrieval_ms += (time.perf_counter() - started) * 1000

        if retrieval.is_empty:
            return ""

        metrics.retrieved_chunks += len(retrieval.chunks)
        result.citations.extend(retrieval.citations)
        return retrieval.context

    @staticmethod
    async def _model_context_length(client: InferenceClient) -> int | None:
        try:
            info = await client.model_info()
            return info.context_length
        except Exception as exc:  # noqa: BLE001 - fall back to configuration
            logger.debug("model_info unavailable: %s", exc)
            return None


def _failed_tool_metric(name: str, error: str):
    from app.core.metrics import ToolMetrics

    return ToolMetrics(name=name, duration_ms=0.0, ok=False, error=error)


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    """Collapse repeats, keeping first-seen order and the best score."""
    by_chunk: dict[str, Citation] = {}
    for citation in citations:
        existing = by_chunk.get(citation.chunk_id)
        if existing is None:
            by_chunk[citation.chunk_id] = citation
        elif (citation.score or 0) > (existing.score or 0):
            by_chunk[citation.chunk_id] = citation
    return list(by_chunk.values())


default_runtime = AgentRuntime()
