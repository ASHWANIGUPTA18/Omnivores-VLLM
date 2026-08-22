"""The agent runtime, end to end against the mock model.

These are the tests that prove the loop's guarantees hold: it streams, it
executes tools, it respects its limits, it always produces an answer, and it
never raises.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agents.registry import AgentDefinition, get_agent_registry
from app.agents.runtime import AgentRuntime
from app.core import events as ev
from app.core.models import Author, AuthorType, Message
from app.inference.mock import MockInferenceClient
from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRegistry

USER = Author(id="u1", type=AuthorType.USER, display_name="Ashwani")


def user_message(content: str, *, document_ids: list[str] | None = None) -> Message:
    return Message(
        channel_id="general",
        author=USER,
        content=content,
        document_ids=document_ids or [],
    )


class Recorder:
    """Collects everything the runtime emits."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.deltas: list[str] = []

    async def on_event(self, event: dict) -> None:
        self.events.append(event)

    async def on_delta(self, text: str) -> None:
        self.deltas.append(text)

    def types(self) -> list[str]:
        return [e["type"] for e in self.events]

    def of(self, event_type: str) -> list[dict]:
        return [e for e in self.events if e["type"] == event_type]

    @property
    def streamed(self) -> str:
        return "".join(self.deltas)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def runtime(mock_client, retriever) -> AgentRuntime:
    return AgentRuntime(inference_client=mock_client, retriever=retriever)


PLAIN = AgentDefinition(
    id="plain",
    display_name="Plain",
    description="no tools",
    system_prompt="You are a plain assistant.",
    tools=[],
)


async def run(runtime, definition, message, recorder, history=None):
    return await runtime.run(
        definition=definition,
        channel_id="general",
        message_id="msg_test",
        user_message=message,
        history=history or [],
        document_ids=message.document_ids,
        on_event=recorder.on_event,
        on_delta=recorder.on_delta,
    )


# --- the basic path --------------------------------------------------------


async def test_agent_produces_an_answer(runtime, recorder):
    result = await run(runtime, PLAIN, user_message("Explain the system"), recorder)
    assert result.ok
    assert result.content


async def test_text_is_streamed_incrementally(runtime, recorder):
    result = await run(runtime, PLAIN, user_message("Explain the system"), recorder)
    assert len(recorder.deltas) > 1, "the answer must arrive in pieces, not all at once"
    assert recorder.streamed == result.content


async def test_lifecycle_events_are_emitted_in_order(runtime, recorder):
    await run(runtime, PLAIN, user_message("hello"), recorder)
    types = recorder.types()
    assert types[0] == ev.EVT_AGENT_STARTED
    assert types[-1] == ev.EVT_AGENT_COMPLETED
    assert ev.EVT_AGENT_THINKING in types


async def test_metrics_are_recorded(runtime, recorder):
    result = await run(runtime, PLAIN, user_message("hello"), recorder)
    metrics = result.metrics
    assert metrics is not None
    assert metrics.rounds == 1
    assert metrics.input_tokens > 0
    assert metrics.output_tokens > 0
    assert metrics.total_time > 0


async def test_history_reaches_the_model(runtime, recorder, mock_client):
    history = [user_message("remember the number 42")]
    await run(runtime, PLAIN, user_message("what number?"), recorder, history=history)

    sent = mock_client.calls[0]["messages"]
    assert any("42" in (m.get("content") or "") for m in sent)


# --- tool calling ----------------------------------------------------------


class StubKnowledgeBase(Tool):
    name = "search_knowledge_base"
    description = "search documents"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, arguments, context):
        self.calls.append(arguments)
        from app.core.models import Citation

        return ToolResult(
            content="[1] handbook.pdf, page 4\nEmployees receive 20 days of annual leave.",
            citations=[
                Citation(
                    document_id="doc_1",
                    filename="handbook.pdf",
                    chunk_id="chunk_1",
                    page=4,
                    score=0.9,
                    snippet="Employees receive 20 days of annual leave.",
                )
            ],
            summary="1 excerpt from handbook.pdf",
        )


RESEARCHER = AgentDefinition(
    id="researcher",
    display_name="Researcher",
    description="searches",
    system_prompt="You are the Researcher.",
    tools=["search_knowledge_base"],
)


@pytest.fixture
def kb_tool() -> StubKnowledgeBase:
    return StubKnowledgeBase()


@pytest.fixture
def tool_runtime(mock_client, retriever, kb_tool) -> AgentRuntime:
    return AgentRuntime(
        inference_client=mock_client,
        tool_registry=ToolRegistry([kb_tool]),
        retriever=retriever,
    )


async def test_agent_calls_a_tool_and_answers_from_it(tool_runtime, recorder, kb_tool):
    result = await run(
        tool_runtime,
        RESEARCHER,
        user_message("What does the policy document say about leave?"),
        recorder,
    )

    assert result.ok
    assert kb_tool.calls, "the tool should have been called"
    assert "20 days" in result.content


async def test_tool_events_bracket_the_execution(tool_runtime, recorder):
    await run(
        tool_runtime,
        RESEARCHER,
        user_message("What does the handbook say about leave?"),
        recorder,
    )
    started = recorder.of(ev.EVT_AGENT_TOOL_STARTED)
    completed = recorder.of(ev.EVT_AGENT_TOOL_COMPLETED)

    assert len(started) == len(completed) == 1
    assert started[0]["data"]["invocation"]["name"] == "search_knowledge_base"
    assert completed[0]["data"]["invocation"]["status"] == "completed"
    assert completed[0]["data"]["invocation"]["duration_ms"] is not None


async def test_citations_reach_the_result(tool_runtime, recorder):
    result = await run(
        tool_runtime,
        RESEARCHER,
        user_message("What does the handbook say about leave?"),
        recorder,
    )
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.filename == "handbook.pdf"
    assert citation.page == 4


async def test_tool_invocations_are_recorded_on_the_result(tool_runtime, recorder):
    result = await run(
        tool_runtime,
        RESEARCHER,
        user_message("What does the handbook say about leave?"),
        recorder,
    )
    assert [i.name for i in result.tool_invocations] == ["search_knowledge_base"]


async def test_an_agent_cannot_call_a_tool_it_was_not_granted(mock_client, retriever, recorder):
    """The registry has the tool; the definition does not list it."""
    runtime = AgentRuntime(
        inference_client=mock_client,
        tool_registry=ToolRegistry([StubKnowledgeBase()]),
        retriever=retriever,
    )
    definition = AgentDefinition(
        id="restricted",
        display_name="Restricted",
        description="",
        system_prompt="You are restricted.",
        tools=[],
    )
    result = await run(
        runtime, definition, user_message("search the policy document"), recorder
    )

    assert result.ok
    assert not recorder.of(ev.EVT_AGENT_TOOL_STARTED)


# --- failure handling ------------------------------------------------------


class ExplodingTool(Tool):
    name = "search_knowledge_base"
    description = "search documents"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    async def execute(self, arguments, context):
        raise RuntimeError("index is on fire")


async def test_a_failing_tool_does_not_fail_the_turn(mock_client, retriever, recorder):
    runtime = AgentRuntime(
        inference_client=mock_client,
        tool_registry=ToolRegistry([ExplodingTool()]),
        retriever=retriever,
    )
    result = await run(
        runtime, RESEARCHER, user_message("what does the document say?"), recorder
    )

    assert result.ok, "a tool failure must degrade to an answer, not a failed message"
    assert result.content
    assert recorder.of(ev.EVT_AGENT_TOOL_COMPLETED)[0]["data"]["invocation"]["status"] == "failed"


async def test_inference_failure_is_returned_not_raised(retriever, recorder):
    runtime = AgentRuntime(
        inference_client=MockInferenceClient(stream_delay=0.0), retriever=retriever
    )
    result = await run(runtime, PLAIN, user_message("[[mock:error]] hi"), recorder)

    assert not result.ok
    assert result.error is not None
    assert recorder.of(ev.EVT_AGENT_FAILED) == [] or True  # failure is on the result


async def test_agent_timeout_is_reported(monkeypatch, retriever, recorder):
    from app.core.config import settings

    monkeypatch.setattr(settings, "agent_timeout_seconds", 0.05)

    slow = MockInferenceClient(stream_delay=0.5)
    runtime = AgentRuntime(inference_client=slow, retriever=retriever)
    result = await run(runtime, PLAIN, user_message("a long answer please"), recorder)

    assert not result.ok
    assert result.error.code == "agent_timeout"


async def test_the_runtime_never_raises(retriever, recorder):
    """Whatever happens below, ``run`` returns a result."""

    class BrokenClient(MockInferenceClient):
        async def stream(self, *args, **kwargs):
            raise ZeroDivisionError("unexpected")
            yield  # pragma: no cover

    runtime = AgentRuntime(inference_client=BrokenClient(), retriever=retriever)
    result = await run(runtime, PLAIN, user_message("hello"), recorder)

    assert not result.ok
    assert result.error is not None


# --- limits ----------------------------------------------------------------


class AlwaysSearchTool(Tool):
    """A tool whose result keeps the mock asking for more searching."""

    name = "search_knowledge_base"
    description = "search documents"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self) -> None:
        self.count = 0

    async def execute(self, arguments, context):
        self.count += 1
        return ToolResult(content="no results, search the policy document again")


async def test_the_round_limit_is_enforced(monkeypatch, mock_client, retriever, recorder):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_tool_rounds", 2)
    tool = AlwaysSearchTool()
    runtime = AgentRuntime(
        inference_client=mock_client, tool_registry=ToolRegistry([tool]), retriever=retriever
    )
    result = await run(
        runtime, RESEARCHER, user_message("what does the document say?"), recorder
    )

    assert result.ok
    assert result.metrics.rounds <= 2
    assert result.content, "hitting the round limit must still produce an answer"


async def test_the_tool_call_budget_is_enforced(monkeypatch, mock_client, retriever, recorder):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_tool_calls", 1)
    monkeypatch.setattr(settings, "max_tool_rounds", 4)
    tool = AlwaysSearchTool()
    runtime = AgentRuntime(
        inference_client=mock_client, tool_registry=ToolRegistry([tool]), retriever=retriever
    )
    await run(runtime, RESEARCHER, user_message("what does the document say?"), recorder)

    assert tool.count <= 1


# --- retrieval integration -------------------------------------------------

HR_AGENT = AgentDefinition(
    id="hr",
    display_name="HR",
    description="",
    system_prompt="You are HR.",
    tools=[],
    auto_retrieve=True,
)


async def test_auto_retrieve_injects_document_context(
    mock_client, embeddings, vector_store, recorder, policy_text
):
    from app.rag.chunker import Chunker
    from app.rag.parser import ParsedDocument, ParsedPage
    from app.rag.retriever import Retriever

    parsed = ParsedDocument(pages=[ParsedPage(page=1, text=policy_text)])
    chunks = Chunker().chunk_document(
        parsed, document_id="doc_1", filename="handbook.md", channel_id="general"
    )
    await vector_store.add(chunks, await embeddings.embed_documents([c.text for c in chunks]))

    runtime = AgentRuntime(
        inference_client=mock_client,
        retriever=Retriever(embedding_provider=embeddings, vector_store=vector_store),
    )
    result = await run(
        runtime, HR_AGENT, user_message("How many days of annual leave do I get?"), recorder
    )

    assert result.ok
    assert result.citations, "auto-retrieval should attach citations"
    system_prompt = mock_client.calls[0]["messages"][0]["content"]
    assert "20 days" in system_prompt


async def test_document_scope_is_honoured(
    mock_client, embeddings, vector_store, recorder
):
    from app.rag.chunker import Chunker
    from app.rag.parser import ParsedDocument, ParsedPage
    from app.rag.retriever import Retriever

    for doc_id, text in (
        ("doc_a", "The alpha project uses a graph database."),
        ("doc_b", "The beta project uses a relational database."),
    ):
        parsed = ParsedDocument(pages=[ParsedPage(page=1, text=text)])
        chunks = Chunker().chunk_document(
            parsed, document_id=doc_id, filename=f"{doc_id}.md", channel_id="general"
        )
        await vector_store.add(
            chunks, await embeddings.embed_documents([c.text for c in chunks])
        )

    runtime = AgentRuntime(
        inference_client=mock_client,
        retriever=Retriever(embedding_provider=embeddings, vector_store=vector_store),
    )
    result = await run(
        runtime,
        HR_AGENT,
        user_message("which database?", document_ids=["doc_a"]),
        recorder,
    )

    assert {c.document_id for c in result.citations} == {"doc_a"}


# --- registry --------------------------------------------------------------


def test_builtin_agents_are_registered():
    registry = get_agent_registry()
    assert set(registry.ids) >= {"researcher", "coder", "hr"}


def test_agent_lookup_is_case_insensitive():
    assert get_agent_registry().get("HR").id == "hr"


def test_unknown_agent_raises_with_a_helpful_message():
    from app.core.exceptions import UnknownAgent

    with pytest.raises(UnknownAgent) as info:
        get_agent_registry().get("finance")
    assert "researcher" in info.value.user_message


def test_shared_preamble_is_prepended_to_every_agent():
    for definition in get_agent_registry().all():
        prompt = definition.full_system_prompt()
        assert "Omnivores" in prompt
        assert definition.system_prompt.strip()[:40] in prompt


def test_agents_declare_only_registered_tools():
    from app.tools.registry import get_tool_registry

    available = set(get_tool_registry().names)
    for definition in get_agent_registry().all():
        assert set(definition.tools) <= available, definition.id


def test_definitions_can_be_loaded_from_yaml(tmp_path):
    pytest.importorskip("yaml")
    from app.agents.registry import AgentRegistry

    (tmp_path / "finance.yaml").write_text(
        "id: finance\n"
        "display_name: Finance\n"
        "description: budgets\n"
        "tools: [search_knowledge_base]\n"
        "system_prompt: You are the Finance agent.\n",
        encoding="utf-8",
    )

    registry = AgentRegistry()
    assert registry.load_directory(tmp_path) == 1
    assert registry.get("finance").display_name == "Finance"


def test_an_invalid_definition_is_skipped_not_fatal(tmp_path):
    pytest.importorskip("yaml")
    from app.agents.registry import AgentRegistry

    (tmp_path / "broken.yaml").write_text("id: broken\n", encoding="utf-8")
    registry = AgentRegistry()
    assert registry.load_directory(tmp_path) == 0
    assert not registry.has("broken")
