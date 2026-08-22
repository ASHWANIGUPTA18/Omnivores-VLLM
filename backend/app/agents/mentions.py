"""@mention parsing.

Mentions are how a message is routed to an agent, so the parser has to be
precise about what counts. Rules:

* ``@name`` matches only at a word boundary, so ``email@example.com`` and
  ``a@b`` do not trigger anything.
* Mentions inside fenced or inline code are ignored - people paste code
  containing decorators and shell variables constantly.
* Matching is case-insensitive; ``@HR`` and ``@hr`` are the same agent.
* Order is preserved and duplicates removed, so "@coder @hr @coder" runs each
  agent once, in the order first mentioned.
* An unknown ``@name`` is reported separately rather than silently ignored, so
  the user finds out they typed ``@reseacher``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Fenced blocks first, then inline spans, so ``` wins over `.
_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)

#: A mention must start at a non-word character and name may contain letters,
#: digits, underscore and hyphen.
_MENTION_RE = re.compile(r"(?<![\w@.])@([A-Za-z][\w-]{0,31})\b")


@dataclass
class MentionParse:
    #: Agent ids that exist, in first-mention order.
    agents: list[str] = field(default_factory=list)
    #: Mentioned names with no matching agent.
    unknown: list[str] = field(default_factory=list)
    #: The message with mentions of known agents removed, for readability.
    cleaned_text: str = ""

    @property
    def has_agents(self) -> bool:
        return bool(self.agents)


def _mask_code(text: str) -> str:
    """Replace code spans with same-length filler so offsets stay valid."""
    return _CODE_RE.sub(lambda m: " " * len(m.group(0)), text)


def find_mentions(text: str) -> list[str]:
    """Every @name in ``text``, lowercased, deduplicated, in order."""
    seen: set[str] = set()
    names: list[str] = []
    for match in _MENTION_RE.finditer(_mask_code(text or "")):
        name = match.group(1).lower()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def parse_mentions(text: str, known_agent_ids: set[str] | list[str]) -> MentionParse:
    """Split mentions into known agents and unknown names."""
    known = {a.lower() for a in known_agent_ids}
    result = MentionParse()

    for name in find_mentions(text):
        if name in known:
            result.agents.append(name)
        else:
            result.unknown.append(name)

    result.cleaned_text = strip_mentions(text, result.agents)
    return result


def strip_mentions(text: str, agent_ids: list[str]) -> str:
    """Remove mentions of the given agents, leaving other text intact."""
    if not agent_ids:
        return (text or "").strip()

    targets = {a.lower() for a in agent_ids}
    masked = _mask_code(text or "")

    # Rebuild from the original string using match spans found on the masked
    # copy, so mentions inside code survive untouched.
    out: list[str] = []
    cursor = 0
    for match in _MENTION_RE.finditer(masked):
        if match.group(1).lower() not in targets:
            continue
        out.append(text[cursor : match.start()])
        cursor = match.end()
    out.append(text[cursor:])

    cleaned = "".join(out)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(" \t,:;")
