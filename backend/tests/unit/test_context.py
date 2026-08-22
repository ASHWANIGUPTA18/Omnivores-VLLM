"""Context assembly and token estimation."""

from __future__ import annotations

from app.agents.context import ContextManager
from app.core.models import Author, AuthorType, Message
from app.core.tokens import estimate_tokens, truncate_to_tokens
from app.inference.base import ToolSpec

USER = Author(id="u1", type=AuthorType.USER, display_name="Ashwani")
OTHER_USER = Author(id="u2", type=AuthorType.USER, display_name="Priya")
CODER = Author(id="coder", type=AuthorType.AGENT, display_name="Coder")
HR = Author(id="hr", type=AuthorType.AGENT, display_name="HR Assistant")


def message(content: str, author: Author = USER) -> Message:
    return Message(channel_id="general", author=author, content=content)


# --- token estimation ------------------------------------------------------


def test_estimate_is_zero_for_empty_text():
    assert estimate_tokens("") == 0


def test_estimate_grows_with_length():
    assert estimate_tokens("word " * 100) > estimate_tokens("word " * 10)


def test_estimate_never_underestimates_word_count_badly():
    """Underestimating causes a context rejection mid-conversation, which is a
    far worse failure than sending slightly less history."""
    text = "a b c d e f g h i j"
    assert estimate_tokens(text) >= 7


def test_truncation_respects_the_limit():
    text = "word " * 500
    assert estimate_tokens(truncate_to_tokens(text, 50)) <= 55


def test_truncation_leaves_short_text_alone():
    assert truncate_to_tokens("short", 100) == "short"


# --- context building ------------------------------------------------------


def test_context_starts_with_system_and_ends_with_the_question():
    built = ContextManager().build(
        system_prompt="You are the Coder.",
        history=[],
        current_message=message("write a sort function"),
        agent_id="coder",
    )
    assert built.messages[0].role == "system"
    assert built.messages[-1].role == "user"
    assert "write a sort function" in built.messages[-1].content


def test_document_context_is_appended_to_the_system_prompt():
    built = ContextManager().build(
        system_prompt="You are HR.",
        history=[],
        current_message=message("leave policy?"),
        agent_id="hr",
        document_context="[1] handbook.pdf, page 2\n20 days of leave",
    )
    assert "handbook.pdf" in built.messages[0].content
    assert built.rag_tokens > 0


def test_history_is_included_oldest_first():
    history = [message(f"message {i}") for i in range(5)]
    built = ContextManager().build(
        system_prompt="prompt",
        history=history,
        current_message=message("latest"),
        agent_id="coder",
    )
    contents = [m.content for m in built.messages[1:-1]]
    assert "message 0" in contents[0]
    assert "message 4" in contents[-1]


def test_the_agents_own_replies_become_assistant_turns():
    built = ContextManager().build(
        system_prompt="prompt",
        history=[message("earlier answer", CODER)],
        current_message=message("follow up"),
        agent_id="coder",
    )
    assert built.messages[1].role == "assistant"
    assert built.messages[1].content == "earlier answer"


def test_another_agents_replies_are_not_presented_as_our_own():
    """Presenting another model's output as this model's history makes it
    defend positions it never took."""
    built = ContextManager().build(
        system_prompt="prompt",
        history=[message("HR said something", HR)],
        current_message=message("what do you think?"),
        agent_id="coder",
    )
    assert built.messages[1].role == "user"
    assert "HR Assistant (agent)" in built.messages[1].content


def test_speakers_are_attributed_in_a_multi_user_channel():
    built = ContextManager().build(
        system_prompt="prompt",
        history=[message("first point", USER), message("second point", OTHER_USER)],
        current_message=message("what do you think?"),
        agent_id="coder",
    )
    assert "Ashwani:" in built.messages[1].content
    assert "Priya:" in built.messages[2].content


def test_the_current_message_is_never_duplicated_from_history():
    current = message("the question")
    built = ContextManager().build(
        system_prompt="prompt",
        history=[current],
        current_message=current,
        agent_id="coder",
    )
    assert sum(1 for m in built.messages if "the question" in (m.content or "")) == 1


def test_empty_messages_are_skipped():
    built = ContextManager().build(
        system_prompt="prompt",
        history=[message(""), message("real content")],
        current_message=message("q"),
        agent_id="coder",
    )
    assert built.included_messages == 1


def test_history_is_dropped_to_fit_the_budget():
    manager = ContextManager(max_context_tokens=900, output_reserve=300, recent_messages=100)
    history = [message("filler " * 60) for _ in range(40)]

    built = manager.build(
        system_prompt="prompt",
        history=history,
        current_message=message("the question"),
        agent_id="coder",
    )

    assert built.dropped_messages > 0
    assert built.included_messages < len(history)
    assert built.total_tokens <= 900


def test_dropped_history_is_announced_to_the_model():
    manager = ContextManager(max_context_tokens=900, output_reserve=300, recent_messages=100)
    built = manager.build(
        system_prompt="prompt",
        history=[message("filler " * 60) for _ in range(40)],
        current_message=message("q"),
        agent_id="coder",
    )
    notes = [m for m in built.messages if m.role == "system" and "omitted" in (m.content or "")]
    assert notes, "the model must be told history was truncated, not silently lose it"


def test_the_question_survives_even_when_everything_else_cannot():
    manager = ContextManager(max_context_tokens=700, output_reserve=200)
    built = manager.build(
        system_prompt="prompt",
        history=[message("filler " * 200) for _ in range(20)],
        current_message=message("the critical question"),
        agent_id="coder",
    )
    assert "the critical question" in built.messages[-1].content


def test_recent_message_cap_is_applied():
    manager = ContextManager(recent_messages=3)
    built = manager.build(
        system_prompt="prompt",
        history=[message(f"m{i}") for i in range(20)],
        current_message=message("q"),
        agent_id="coder",
    )
    assert built.included_messages == 3


def test_tool_schemas_are_counted_against_the_budget():
    spec = ToolSpec(
        name="t",
        description="d" * 500,
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    manager = ContextManager()
    without = manager.build(
        system_prompt="p", history=[], current_message=message("q"), agent_id="a"
    )
    with_tools = manager.build(
        system_prompt="p",
        history=[],
        current_message=message("q"),
        agent_id="a",
        tool_specs=[spec],
    )
    assert with_tools.total_tokens > without.total_tokens


def test_model_context_length_overrides_a_larger_configured_limit():
    """Never promise the model more room than it has, whatever config says."""
    manager = ContextManager(max_context_tokens=100_000, output_reserve=500)
    built = manager.build(
        system_prompt="prompt",
        history=[message("filler " * 100) for _ in range(50)],
        current_message=message("q"),
        agent_id="coder",
        model_context_length=2048,
    )
    assert built.total_tokens <= 2048


def test_oversized_document_context_is_trimmed_not_the_question():
    manager = ContextManager(max_context_tokens=1200, output_reserve=400)
    built = manager.build(
        system_prompt="prompt",
        history=[],
        current_message=message("the question"),
        agent_id="hr",
        document_context="excerpt " * 2000,
    )
    assert "the question" in built.messages[-1].content
    assert built.total_tokens <= 1200
