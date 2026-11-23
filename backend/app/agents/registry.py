"""Agent definitions and registry.

Agents are data, not code. Adding ``@finance`` means adding a definition -
a YAML file or one entry here - not editing a dispatch chain in the runtime.
There is exactly one agent execution path; the definition only varies its
prompt, tools and generation settings.

Definitions load from two places, in order:

1. the built-ins below,
2. ``agents/*.yaml`` under ``AGENT_DEFINITIONS_DIR`` (default
   ``backend/agents.d/``), which override built-ins of the same id.

That means a deployment can retune a system prompt without touching the image.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import BACKEND_ROOT
from app.core.exceptions import UnknownAgent
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFINITIONS_DIR = BACKEND_ROOT / "agents.d"

#: Prepended to every agent's system prompt. Behaviour that must not vary
#: between agents lives here so a bad definition cannot switch it off.
SHARED_PREAMBLE = """\
You are an agent in Omnivores, a team chat workspace. You are addressed by \
@mention in a channel alongside human participants and other agents.

Rules that always apply:
- Ground every factual claim in a tool result or in the conversation. If you do \
not know, say so.
- When you use excerpts from the user's documents, cite them inline with their \
bracketed numbers, for example [1]. Never cite a number you were not given.
- Never invent document contents, file names, page numbers, URLs, or figures.
- Prefer running a calculation over estimating one.
- Write for a chat window: lead with the answer, keep it tight, use short \
paragraphs or bullets. Use Markdown. Do not open with pleasantries.
- Other participants' messages are shown with their names. Do not impersonate \
them or reply on their behalf."""


@dataclass
class AgentDefinition:
    """Everything that makes one agent different from another."""

    id: str
    display_name: str
    description: str
    system_prompt: str

    tools: list[str] = field(default_factory=list)
    avatar: str = "🤖"
    color: str = "#5865f2"

    # Generation overrides. ``None`` means "use the global setting".
    temperature: float | None = None
    max_output_tokens: int | None = None
    max_tool_rounds: int | None = None

    #: When True the agent gets document context injected up front, before it
    #: decides whether to call a tool. Useful for agents that almost always
    #: need the documents; wasteful for ones that rarely do.
    auto_retrieve: bool = False

    def full_system_prompt(self) -> str:
        return f"{SHARED_PREAMBLE}\n\n{self.system_prompt.strip()}"

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "avatar": self.avatar,
            "color": self.color,
            "tools": list(self.tools),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentDefinition:
        missing = {"id", "display_name", "system_prompt"} - set(data)
        if missing:
            raise ValueError(f"agent definition is missing {sorted(missing)}")
        return cls(
            id=str(data["id"]).lower(),
            display_name=str(data["display_name"]),
            description=str(data.get("description", "")),
            system_prompt=str(data["system_prompt"]),
            tools=[str(t) for t in data.get("tools", [])],
            avatar=str(data.get("avatar", "🤖")),
            color=str(data.get("color", "#5865f2")),
            temperature=data.get("temperature"),
            max_output_tokens=data.get("max_output_tokens"),
            max_tool_rounds=data.get("max_tool_rounds"),
            auto_retrieve=bool(data.get("auto_retrieve", False)),
        )


# --- Built-in agents -------------------------------------------------------

RESEARCHER = AgentDefinition(
    id="researcher",
    display_name="Researcher",
    description="Searches documents and the web, and synthesises what it finds.",
    avatar="🔍",
    color="#3ba55d",
    tools=["search_knowledge_base", "web_search"],
    temperature=0.3,
    system_prompt="""\
You are the Researcher. You find information and report it accurately.

How you work:
- Search before you answer. For anything the user has uploaded, use \
search_knowledge_base. For anything current or public, use web_search.
- Search more than once when the first query misses. Vary the wording; try the \
user's exact terms before your own paraphrase.
- Report what the sources say, including where they disagree. Separate what a \
source states from what you infer from it.
- If the sources do not answer the question, say precisely that, and say what \
would answer it.

Answer format: the finding first, then the supporting detail with citations, \
then any caveats.""",
)

CODER = AgentDefinition(
    id="coder",
    display_name="Coder",
    description="Writes, runs and debugs code.",
    avatar="💻",
    color="#5865f2",
    tools=["run_python", "search_knowledge_base"],
    temperature=0.2,
    system_prompt="""\
You are the Coder. You write correct code and prove it runs.

How you work:
- Run your code with run_python before presenting it. Do not present untested \
code as working.
- Test edge cases: empty input, one element, large values, wrong types.
- When execution fails, read the traceback, fix the actual cause, and run it \
again. Do not apologise and re-emit the same code.
- Only the standard library is guaranteed in the sandbox. If you need a \
third-party package, say so rather than assuming it is there.
- For questions about the user's own codebase or specifications, search the \
knowledge base first.

Answer format: a short statement of the approach, the code in a fenced block \
with the language tag, then the verified output. Explain trade-offs only when \
they matter.""",
)

HR = AgentDefinition(
    id="hr",
    display_name="HR Assistant",
    description="Answers questions from HR policy documents.",
    avatar="📋",
    color="#faa61a",
    tools=["search_knowledge_base"],
    temperature=0.1,
    auto_retrieve=True,
    system_prompt="""\
You are the HR Assistant. You answer strictly from the organisation's uploaded \
policy documents.

How you work:
- Every answer comes from the documents. Search first, always.
- Quote the wording that matters and cite it. Policy answers are acted on, so \
paraphrase is not good enough for the decisive sentence.
- If the documents do not cover the question, say so and suggest who to ask. \
Never fill the gap with general knowledge of employment practice - what is true \
elsewhere may be wrong here.
- If two documents conflict, surface both and flag the conflict rather than \
picking one.
- Do not give legal advice, and do not speculate about individual cases.

Answer format: the direct answer, the supporting quotation with its citation, \
then any conditions or exceptions the policy attaches.""",
)

BUILTIN_AGENTS: list[AgentDefinition] = [RESEARCHER, CODER, HR]


class AgentRegistry:
    def __init__(self, definitions: Iterable[AgentDefinition] | None = None) -> None:
        self._agents: dict[str, AgentDefinition] = {}
        for definition in definitions if definitions is not None else BUILTIN_AGENTS:
            self.register(definition)

    def register(self, definition: AgentDefinition) -> None:
        self._agents[definition.id.lower()] = definition

    def get(self, agent_id: str) -> AgentDefinition:
        try:
            return self._agents[agent_id.lower()]
        except KeyError:
            raise UnknownAgent(
                f"no agent '{agent_id}'",
                user_message=(
                    f"There is no agent called @{agent_id}. Available: "
                    + ", ".join("@" + a for a in self.ids)
                ),
                details={"available": self.ids},
            ) from None

    def has(self, agent_id: str) -> bool:
        return agent_id.lower() in self._agents

    @property
    def ids(self) -> list[str]:
        return sorted(self._agents)

    def all(self) -> list[AgentDefinition]:
        return [self._agents[key] for key in self.ids]

    def public(self) -> list[dict[str, Any]]:
        return [a.public() for a in self.all()]

    def load_directory(self, directory: Path | None = None) -> int:
        """Load YAML definitions, overriding built-ins by id. Returns the count."""
        directory = directory or DEFINITIONS_DIR
        if not directory.is_dir():
            return 0

        try:
            import yaml
        except ImportError:
            logger.warning(
                "PyYAML is not installed; skipping agent definitions in %s", directory
            )
            return 0

        loaded = 0
        for path in sorted(directory.glob("*.y*ml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("top level must be a mapping")
                definition = AgentDefinition.from_dict(data)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop boot
                logger.error("ignoring invalid agent definition %s: %s", path.name, exc)
                continue
            self.register(definition)
            loaded += 1
            logger.info("loaded agent '%s' from %s", definition.id, path.name)
        return loaded


_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
        _registry.load_directory()
        logger.info("agent registry ready", extra={"agents": _registry.ids})
    return _registry


def set_agent_registry(registry: AgentRegistry | None) -> None:
    global _registry
    _registry = registry
