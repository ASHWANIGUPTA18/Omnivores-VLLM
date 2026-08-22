"""Mention parsing.

Routing depends entirely on this, and the failure modes are user-visible: an
email address that summons an agent, or a pasted code snippet that triggers
three of them.
"""

from __future__ import annotations

from app.agents.mentions import find_mentions, parse_mentions, strip_mentions

KNOWN = ["researcher", "coder", "hr"]


def test_finds_single_mention():
    assert find_mentions("@coder please help") == ["coder"]


def test_is_case_insensitive():
    assert find_mentions("@HR and @Coder") == ["hr", "coder"]


def test_preserves_order_and_deduplicates():
    assert find_mentions("@coder @hr @coder again") == ["coder", "hr"]


def test_ignores_email_addresses():
    assert find_mentions("write to lakshya.gupta@devslane.com about it") == []


def test_ignores_mid_word_at():
    assert find_mentions("user@host and a@b") == []


def test_ignores_mentions_in_inline_code():
    assert find_mentions("use the `@decorator` syntax") == []


def test_ignores_mentions_in_fenced_code():
    text = "Here:\n```python\n@property\ndef x(self): ...\n```\nthoughts?"
    assert find_mentions(text) == []


def test_finds_mention_outside_a_code_block():
    text = "@coder look at\n```\n@staticmethod\n```\nplease"
    assert find_mentions(text) == ["coder"]


def test_separates_known_from_unknown():
    result = parse_mentions("@coder and @nobody", KNOWN)
    assert result.agents == ["coder"]
    assert result.unknown == ["nobody"]


def test_strips_only_known_mentions():
    result = parse_mentions("@coder fix @nobody's bug", KNOWN)
    assert "@coder" not in result.cleaned_text
    assert "@nobody" in result.cleaned_text


def test_strip_leaves_code_mentions_intact():
    text = "@coder explain `@property`"
    assert strip_mentions(text, ["coder"]) == "explain `@property`"


def test_mention_at_start_of_line_after_punctuation():
    assert find_mentions("Hey, @hr - what is the leave policy?") == ["hr"]


def test_empty_and_none_are_safe():
    assert find_mentions("") == []
    assert find_mentions(None) == []
    assert parse_mentions("", KNOWN).agents == []


def test_no_mentions_means_no_agents():
    result = parse_mentions("just talking to the team", KNOWN)
    assert not result.has_agents
    assert result.cleaned_text == "just talking to the team"
