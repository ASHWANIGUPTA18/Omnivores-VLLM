"""The inference contract: stream reassembly and mock backend behaviour.

``StreamAccumulator`` is shared by every backend, so a bug here breaks vLLM and
OpenAI identically - which is exactly why it is tested against the mock on a
laptop.
"""

from __future__ import annotations

import json

import pytest

from app.core.exceptions import InferenceError, InferenceTimeout
from app.inference.base import (
    ChatChunk,
    ChatMessage,
    StreamAccumulator,
    ToolCall,
    ToolCallDelta,
    ToolSpec,
)

SEARCH_TOOL = ToolSpec(
    name="search_knowledge_base",
    description="search documents",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
PYTHON_TOOL = ToolSpec(
    name="run_python",
    description="run code",
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
)


# --- StreamAccumulator -----------------------------------------------------


def test_accumulator_joins_text():
    acc = StreamAccumulator()
    for piece in ("Hello", ", ", "world"):
        acc.add(ChatChunk(content=piece))
    assert acc.result().content == "Hello, world"


def test_accumulator_reassembles_fragmented_tool_calls():
    acc = StreamAccumulator()
    acc.add(ChatChunk(tool_calls=[ToolCallDelta(index=0, id="c1", name="search")]))
    acc.add(ChatChunk(tool_calls=[ToolCallDelta(index=0, arguments='{"que')]))
    acc.add(ChatChunk(tool_calls=[ToolCallDelta(index=0, arguments='ry": "leave"}')]))
    acc.add(ChatChunk(finish_reason="tool_calls"))

    result = acc.result()
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "c1"
    assert call.name == "search"
    assert call.parsed_arguments() == {"query": "leave"}
    assert result.finish_reason == "tool_calls"


def test_accumulator_handles_parallel_tool_calls():
    acc = StreamAccumulator()
    acc.add(ChatChunk(tool_calls=[ToolCallDelta(index=1, id="b", name="second")]))
    acc.add(ChatChunk(tool_calls=[ToolCallDelta(index=0, id="a", name="first")]))
    acc.add(ChatChunk(tool_calls=[ToolCallDelta(index=0, arguments="{}")]))
    acc.add(ChatChunk(tool_calls=[ToolCallDelta(index=1, arguments="{}")]))

    # Ordered by index, not arrival.
    assert [c.name for c in acc.result().tool_calls] == ["first", "second"]


def test_accumulator_drops_unnamed_tool_fragments():
    """A fragment stream that never names a function is unusable; inventing a
    name would send the runtime looking for a tool that does not exist."""
    acc = StreamAccumulator()
    acc.add(ChatChunk(tool_calls=[ToolCallDelta(index=0, arguments="{}")]))
    assert acc.result().tool_calls == []


def test_tool_call_rejects_malformed_json():
    call = ToolCall(id="x", name="t", arguments="{not json")
    with pytest.raises(ValueError):
        call.parsed_arguments()


def test_tool_call_rejects_non_object_arguments():
    with pytest.raises(ValueError):
        ToolCall(id="x", name="t", arguments="[1, 2]").parsed_arguments()


def test_empty_arguments_parse_to_empty_dict():
    assert ToolCall(id="x", name="t", arguments="").parsed_arguments() == {}


# --- ChatMessage serialization ---------------------------------------------


def test_assistant_message_with_tool_calls_has_null_content():
    message = ChatMessage.assistant(
        tool_calls=[ToolCall(id="c", name="t", arguments="{}")]
    )
    payload = message.to_api()
    assert payload["content"] is None
    assert payload["tool_calls"][0]["function"]["name"] == "t"


def test_tool_message_carries_its_call_id():
    payload = ChatMessage.tool("call_1", "result text", name="search").to_api()
    assert payload == {
        "role": "tool",
        "content": "result text",
        "tool_call_id": "call_1",
        "name": "search",
    }


# --- MockInferenceClient ---------------------------------------------------


async def test_mock_is_deterministic(mock_client):
    messages = [ChatMessage.user("Explain the architecture")]
    first = await mock_client.chat(messages)
    second = await mock_client.chat(messages)
    assert first.content == second.content


async def test_mock_streams_to_the_same_text(mock_client):
    messages = [ChatMessage.user("Describe the system")]
    non_streaming = await mock_client.chat(messages)

    acc = StreamAccumulator()
    async for chunk in mock_client.stream(messages):
        acc.add(chunk)

    assert acc.result().content == non_streaming.content


async def test_mock_reports_usage(mock_client):
    result = await mock_client.chat([ChatMessage.user("hello")])
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


async def test_mock_calls_a_tool_when_the_question_suits_it(mock_client):
    result = await mock_client.chat(
        [ChatMessage.user("What does the policy document say about leave?")],
        tools=[SEARCH_TOOL],
    )
    assert result.wants_tools
    assert result.tool_calls[0].name == "search_knowledge_base"
    assert result.finish_reason == "tool_calls"


async def test_mock_streams_tool_calls_in_fragments(mock_client):
    chunks = []
    async for chunk in mock_client.stream(
        [ChatMessage.user("check the handbook policy")], tools=[SEARCH_TOOL]
    ):
        chunks.append(chunk)

    fragments = [c for c in chunks if c.tool_calls]
    assert len(fragments) > 1, "tool calls should arrive in pieces, as a real server sends them"

    acc = StreamAccumulator()
    for chunk in chunks:
        acc.add(chunk)
    call = acc.result().tool_calls[0]
    assert call.name == "search_knowledge_base"
    assert "query" in call.parsed_arguments()


async def test_mock_answers_after_a_tool_result(mock_client):
    conversation = [
        ChatMessage.user("What is the leave policy in the document?"),
        ChatMessage.assistant(
            tool_calls=[
                ToolCall(id="c1", name="search_knowledge_base", arguments='{"query":"leave"}')
            ]
        ),
        ChatMessage.tool("c1", "[1] handbook.pdf, page 2\n20 days of paid annual leave"),
    ]
    result = await mock_client.chat(conversation, tools=[SEARCH_TOOL])

    assert not result.wants_tools, "must not loop on the same tool after getting a result"
    assert "20 days" in result.content


async def test_mock_does_not_repeat_a_tool_it_already_called(mock_client):
    conversation = [
        ChatMessage.user("search the policy document"),
        ChatMessage.assistant(
            tool_calls=[ToolCall(id="c1", name="search_knowledge_base", arguments="{}")]
        ),
        ChatMessage.tool("c1", "no results"),
    ]
    result = await mock_client.chat(conversation, tools=[SEARCH_TOOL])
    assert not result.wants_tools


async def test_mock_picks_python_for_computation(mock_client):
    result = await mock_client.chat(
        [ChatMessage.user("calculate the sum of the first ten integers")],
        tools=[PYTHON_TOOL, SEARCH_TOOL],
    )
    assert result.tool_calls[0].name == "run_python"
    assert "code" in json.loads(result.tool_calls[0].arguments)


async def test_mock_never_calls_a_tool_it_was_not_offered(mock_client):
    result = await mock_client.chat(
        [ChatMessage.user("calculate the sum of the first ten integers")]
    )
    assert not result.wants_tools


async def test_directive_forces_an_error(mock_client):
    with pytest.raises(InferenceError):
        await mock_client.chat([ChatMessage.user("[[mock:error]] hello")])


async def test_directive_forces_a_timeout(mock_client):
    with pytest.raises(InferenceTimeout):
        await mock_client.chat([ChatMessage.user("[[mock:timeout]] hello")])


async def test_directive_forces_specific_text(mock_client):
    result = await mock_client.chat([ChatMessage.user("[[mock:text=exact answer]]")])
    assert result.content == "exact answer"


async def test_directive_forces_a_specific_tool(mock_client):
    result = await mock_client.chat(
        [ChatMessage.user('[[mock:tool=run_python args={"code":"print(1)"}]]')],
        tools=[PYTHON_TOOL, SEARCH_TOOL],
    )
    assert result.tool_calls[0].name == "run_python"
    assert json.loads(result.tool_calls[0].arguments) == {"code": "print(1)"}


async def test_mock_health_and_model_info(mock_client):
    health = await mock_client.health()
    assert health.status == "healthy"

    info = await mock_client.model_info()
    assert info.supports_tools and info.supports_streaming
    assert info.context_length > 0


async def test_mock_records_calls_for_assertions(mock_client):
    await mock_client.chat([ChatMessage.user("one")])
    await mock_client.chat([ChatMessage.user("two")], tools=[SEARCH_TOOL])
    assert len(mock_client.calls) == 2
    assert mock_client.calls[1]["tools"] == ["search_knowledge_base"]
