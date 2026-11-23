"""Context assembly.

Sending "the last 15 messages" is the naive approach and it fails in two
opposite directions: it overflows the window when messages are long, and it
wastes the window when they are short. This module builds the prompt against a
token budget instead.

Budget layout, highest priority first:

    [ system prompt        ] always included
    [ tool schemas         ] reserved, not sent by us but counted
    [ document context     ] up to CONTEXT_RAG_TOKEN_BUDGET
    [ current user message ] always included
    [ recent history       ] fills whatever is left, newest first
    [ output reserve       ] MAX_OUTPUT_TOKENS held back

Anything that does not fit is dropped from the oldest end, and a one-line
digest records that it happened - a model that silently loses the start of a
conversation gives confusingly amnesiac answers, whereas one told "12 earlier
messages were omitted" behaves sensibly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.core.models import Author, AuthorType, Message
from app.core.tokens import (
    estimate_message_tokens,
    estimate_tokens,
    estimate_tool_schema_tokens,
)
from app.inference.base import ChatMessage, ToolSpec

logger = get_logger(__name__)


@dataclass
class BuiltContext:
    messages: list[ChatMessage] = field(default_factory=list)
    total_tokens: int = 0
    #: How many history messages were dropped for budget reasons.
    dropped_messages: int = 0
    included_messages: int = 0
    rag_tokens: int = 0

    def metrics(self) -> dict[str, int]:
        return {
            "context_tokens": self.total_tokens,
            "included_messages": self.included_messages,
            "dropped_messages": self.dropped_messages,
            "rag_tokens": self.rag_tokens,
        }


class ContextManager:
    def __init__(
        self,
        *,
        max_context_tokens: int | None = None,
        output_reserve: int | None = None,
        recent_messages: int | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens or settings.max_context_tokens
        self.output_reserve = output_reserve or settings.max_output_tokens
        self.recent_messages = recent_messages or settings.context_recent_messages

    def build(
        self,
        *,
        system_prompt: str,
        history: list[Message],
        current_message: Message,
        agent_id: str,
        document_context: str = "",
        tool_specs: list[ToolSpec] | None = None,
        model_context_length: int | None = None,
    ) -> BuiltContext:
        """Assemble the message list for one agent turn."""
        ceiling = self.max_context_tokens
        if model_context_length:
            # Never promise the model more room than it has, whatever the
            # configuration says.
            ceiling = min(ceiling, model_context_length)

        budget = ceiling - self.output_reserve
        if budget <= 0:
            logger.warning(
                "MAX_OUTPUT_TOKENS (%d) leaves no room in a %d-token context",
                self.output_reserve,
                ceiling,
            )
            budget = max(512, ceiling // 2)

        built = BuiltContext()

        # --- fixed costs ---------------------------------------------------
        system_content = system_prompt
        if document_context:
            system_content = f"{system_prompt}\n\n{document_context}"
            built.rag_tokens = estimate_tokens(document_context)

        system_tokens = estimate_message_tokens("system", system_content)
        tool_tokens = sum(
            estimate_tool_schema_tokens(spec.to_api()) for spec in (tool_specs or [])
        )

        current_content = self._render(current_message, agent_id)
        current_tokens = estimate_message_tokens("user", current_content)

        remaining = budget - system_tokens - tool_tokens - current_tokens

        if remaining < 0:
            # The fixed parts alone overflow. Shrink the one elastic piece we
            # control - the document context - rather than dropping the
            # question.
            logger.warning(
                "context overflow before history: system=%d tools=%d current=%d "
                "budget=%d - trimming document context",
                system_tokens,
                tool_tokens,
                current_tokens,
                budget,
            )
            if document_context:
                from app.core.tokens import truncate_to_tokens

                allowance = max(256, built.rag_tokens + remaining - 64)
                document_context = truncate_to_tokens(document_context, allowance)
                system_content = f"{system_prompt}\n\n{document_context}"
                built.rag_tokens = estimate_tokens(document_context)
                system_tokens = estimate_message_tokens("system", system_content)
                remaining = budget - system_tokens - tool_tokens - current_tokens
            remaining = max(0, remaining)

        # --- history, newest first ------------------------------------------
        candidates = history[-self.recent_messages :] if self.recent_messages else list(history)
        # Never include the message we are answering twice.
        candidates = [m for m in candidates if m.id != current_message.id]

        selected: list[ChatMessage] = []
        used = 0
        for message in reversed(candidates):
            rendered = self._to_chat_message(message, agent_id)
            if rendered is None:
                continue
            cost = estimate_message_tokens(rendered.role, rendered.content)
            if used + cost > remaining:
                break
            selected.append(rendered)
            used += cost

        selected.reverse()

        if len(candidates) > len(selected):
            # The loop above stopped because the budget ran out, so whatever
            # slack is left is smaller than the message that did not fit - and
            # usually smaller than the note itself. Pay for the note by giving
            # back the oldest messages we selected: being told that history was
            # truncated is worth more than one more turn of it, because without
            # the note the model answers as though those turns never happened.
            note = _omission_note(len(candidates) - len(selected))
            note_tokens = estimate_message_tokens("system", note)
            while selected and note_tokens > remaining - used:
                evicted = selected.pop(0)
                used -= estimate_message_tokens(evicted.role, evicted.content)
                note = _omission_note(len(candidates) - len(selected))
                note_tokens = estimate_message_tokens("system", note)

            built.included_messages = len(selected)
            built.dropped_messages = len(candidates) - len(selected)

            if note_tokens <= remaining - used:
                selected.insert(0, ChatMessage.system(note))
                used += note_tokens
        else:
            built.included_messages = len(selected)
            built.dropped_messages = 0

        # --- assemble --------------------------------------------------------
        built.messages = [
            ChatMessage.system(system_content),
            *selected,
            ChatMessage.user(current_content),
        ]
        built.total_tokens = system_tokens + used + current_tokens + tool_tokens
        return built

    # --- rendering ---------------------------------------------------------

    @staticmethod
    def _render(message: Message, agent_id: str) -> str:
        """Render a message body, attributing the speaker when it is not us.

        Channels are multi-party. Without the name prefix the model reads three
        people's messages as one person changing their mind.
        """
        content = (message.content or "").strip()
        if message.author.type == AuthorType.USER:
            return f"{message.author.display_name}: {content}" if content else content
        return content

    def _to_chat_message(self, message: Message, agent_id: str) -> ChatMessage | None:
        """Map a stored message onto a chat-completions message.

        This agent's own past replies become ``assistant`` turns. Other agents'
        replies become ``user`` turns tagged with their name - not ``assistant``,
        because presenting another model's output as this model's own history
        makes it defend positions it never took.
        """
        content = (message.content or "").strip()
        if not content:
            return None

        author: Author = message.author

        if author.type == AuthorType.AGENT:
            if author.id == agent_id:
                return ChatMessage.assistant(content)
            return ChatMessage.user(f"{author.display_name} (agent): {content}")

        if author.type == AuthorType.SYSTEM:
            return ChatMessage.system(content)

        return ChatMessage.user(f"{author.display_name}: {content}")


def _omission_note(dropped: int) -> str:
    return (
        f"[{dropped} earlier message(s) in this channel were "
        "omitted to fit the context window. Ask the user if you need "
        "details from earlier in the conversation.]"
    )


default_context_manager = ContextManager()
