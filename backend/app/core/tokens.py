"""Token estimation.

We deliberately do not depend on ``tiktoken`` or a model tokenizer here. The
backend needs a cheap, dependency-free estimate to budget context and size
chunks; the authoritative count comes from the inference server's usage report.

``estimate_tokens`` is intentionally conservative - it rounds up - because the
failure mode of underestimating (a context-length rejection mid-conversation) is
much worse than the failure mode of overestimating (slightly less history).

When an exact count matters, ``InferenceClient.count_tokens`` is the right call:
the vLLM and OpenAI backends can use the real tokenizer.
"""

from __future__ import annotations

import math
import re

#: Empirically close to BPE tokenizers for English prose and code alike:
#: roughly 4 characters per token, but never fewer tokens than 3/4 of the
#: whitespace-delimited word count.
_CHARS_PER_TOKEN = 4.0
_WORD_RE = re.compile(r"\S+")

#: Per-message overhead for chat templates (role markers, separators).
MESSAGE_OVERHEAD_TOKENS = 4
#: Tool schemas are serialized into the prompt; JSON is token-dense.
TOOL_SCHEMA_OVERHEAD_TOKENS = 8


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``. Never returns less than 1 for
    non-empty input."""
    if not text:
        return 0
    char_estimate = len(text) / _CHARS_PER_TOKEN
    word_estimate = len(_WORD_RE.findall(text)) * 0.75
    return max(1, math.ceil(max(char_estimate, word_estimate)))


def estimate_message_tokens(role: str, content: str | None) -> int:
    return MESSAGE_OVERHEAD_TOKENS + estimate_tokens(role) + estimate_tokens(content or "")


def estimate_tool_schema_tokens(schema: dict) -> int:
    import json

    return TOOL_SCHEMA_OVERHEAD_TOKENS + estimate_tokens(
        json.dumps(schema, separators=(",", ":"))
    )


_ELLIPSIS = "..."


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut ``text`` to at most ``max_tokens``, on a word boundary.

    This is a hard upper bound, not an approximation: callers budget context
    with it, so a result one token over the limit defeats the point. The
    ellipsis is charged against the budget rather than added after it.
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = int(max_tokens * _CHARS_PER_TOKEN) - len(_ELLIPSIS)
    if max_chars <= 0:
        return ""

    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars * 0.6:
        cut = cut[:last_space]

    # estimate_tokens also floors on word count, so a character budget alone
    # is not sufficient. Shrink until the estimate genuinely fits.
    result = cut.rstrip() + _ELLIPSIS
    while cut and estimate_tokens(result) > max_tokens:
        cut = cut[: len(cut) - max(1, len(cut) // 16)]
        result = cut.rstrip() + _ELLIPSIS
    return result if cut else ""
