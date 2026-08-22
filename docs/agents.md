# Agents

## Agents are data

There is exactly one agent execution path. A definition varies its prompt,
tools and generation settings - nothing else. Adding `@finance` is a YAML file,
not a branch in the runtime.

```python
@dataclass
class AgentDefinition:
    id: str
    display_name: str
    description: str
    system_prompt: str

    tools: list[str]
    avatar: str
    color: str

    temperature: float | None          # None = use the global setting
    max_output_tokens: int | None
    max_tool_rounds: int | None

    auto_retrieve: bool                # fetch documents before the first call
```

---

## Built-in agents

| Agent | Tools | Temp | Notes |
|---|---|---|---|
| `@researcher` | `search_knowledge_base`, `web_search` | 0.3 | Searches before answering |
| `@coder` | `run_python`, `search_knowledge_base` | 0.2 | Runs code before presenting it |
| `@hr` | `search_knowledge_base` | 0.1 | `auto_retrieve`, quotes policy verbatim |

Temperature descends with how much the answer must be grounded. `@hr` answers
questions people act on, so it is nearly deterministic and refuses to fill gaps
from general knowledge - what is true elsewhere may be wrong here.

---

## Adding an agent

Create `backend/agents.d/finance.yaml`:

```yaml
id: finance
display_name: Finance
description: Answers budget and expense questions from finance documents.
avatar: "💰"
color: "#00b8d4"

tools:
  - search_knowledge_base

temperature: 0.1
auto_retrieve: true

system_prompt: |
  You are the Finance agent. You answer from the organisation's finance
  documents and nothing else.

  How you work:
  - Search before you answer, always.
  - Quote the figure and cite it. Numbers people act on need the source.
  - If a document does not cover the question, say so and say who to ask.
    Never estimate a figure that ought to come from a document.
  - Flag it explicitly when two documents disagree.

  Answer format: the figure first, then the supporting quotation with its
  citation, then any conditions attached to it.
```

Restart the backend. `@finance` is now mentionable, appears in the roster, and
autocompletes in the composer.

To restrict it to one channel, add its id to that channel's `agent_ids` in
`services/message_service.py`.

---

## The shared preamble

Every agent's system prompt is prefixed with `SHARED_PREAMBLE` from
`agents/registry.py`. Behaviour that must not vary between agents lives there,
so a badly written definition cannot switch it off:

* ground claims in tool results or the conversation
* cite excerpts by their bracketed number, and never cite a number you were
  not given
* never invent document contents, filenames, pages, URLs or figures
* prefer running a calculation to estimating one
* write for a chat window
* do not impersonate other participants

---

## Writing a system prompt

What has actually worked, in order of impact:

**Say when *not* to use each tool.** "Use `web_search` for current events; do
not use it for the user's own documents" prevents more wrong tool calls than
any amount of describing what a tool does.

**Specify the answer's shape.** "The figure first, then the quotation with its
citation, then conditions" produces consistent answers. "Be helpful" does not.

**Give the failure behaviour explicitly.** "If the documents do not cover it,
say so and suggest who to ask" is what stops a model inventing a policy.

**Describe process, not personality.** "Run your code before presenting it"
changes behaviour. "You are a brilliant engineer" does not.

**Keep it under ~400 words.** It is prepended to every request in the
conversation; the cost is paid on every turn.

---

## Tools

| Tool | Does | Notes |
|---|---|---|
| `search_knowledge_base` | Searches uploaded documents | Returns numbered excerpts and citations |
| `run_python` | Executes code in a sandbox | Disabled by default |
| `web_search` | Searches the public web | Mock provider unless Tavily is configured |

### Adding one

```python
from app.tools.base import Tool, ToolContext, ToolResult

class CalculatorTool(Tool):
    name = "calculate"
    description = (
        "Evaluate an arithmetic expression exactly. Use this instead of doing "
        "mental arithmetic - never state a computed number you did not compute."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A Python arithmetic expression, e.g. '1250 * 0.18'",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    async def execute(self, arguments, context: ToolContext) -> ToolResult:
        ...
        return ToolResult(content=str(value), summary=f"= {value}")
```

Register it in `tools/registry.py::build_default_registry`, then list
`calculate` in an agent's `tools`.

Write the description **for the model**: when to use it, when not to, and what
it returns. That text is the entire basis on which the model decides.

---

## Permissions

Two independent checks:

1. The runtime only advertises tools listed in the agent's definition.
2. `ToolExecutor` re-checks on execution, so a model that hallucinates a tool
   name it was never offered is denied and told what it may use.

The second check is not redundant. Models do invent tool names, and the failure
should be a legible message the model can recover from - not an exception.

---

## Bounds

| Setting | Default | Bounds |
|---|---|---|
| `MAX_TOOL_ROUNDS` | 5 | Model↔tool exchanges per turn |
| `MAX_TOOL_CALLS` | 10 | Total tool executions per turn |
| `TOOL_TIMEOUT_SECONDS` | 30 | One tool execution |
| `AGENT_TIMEOUT_SECONDS` | 120 | The whole turn |
| `MAX_CONTEXT_TOKENS` | 16384 | Prompt size |
| `MAX_OUTPUT_TOKENS` | 2048 | Reply length |

When the call budget is spent, tools are withheld from the next request, so the
model *must* answer. When the round limit is hit, one final tools-withheld call
forces an answer from what was gathered. Either way the user gets a reply.

---

## Failure handling

Every tool failure becomes a message the model reads:

| Failure | What the model is told |
|---|---|
| Invalid arguments | The exact validation errors, and to retry |
| Unknown tool | That it does not exist, plus the list that does |
| Timeout | That it timed out, and to continue without it |
| Exception | The exception type and message |
| Permission denied | That it is not available, plus what is |
| Sandbox disabled | To answer without running code, and to say so |

None of these fail the turn. This is why a broken tool degrades an answer
instead of destroying it.

---

## Debugging

```bash
# what the agent is configured with, including the full prompt
curl localhost:8000/api/agents/hr | python -m json.tool

# watch context assembly and tool dispatch
LOG_LEVEL=DEBUG uvicorn app.main:app --reload

# drive a specific path deterministically
@researcher [[mock:tool=search_knowledge_base]] find the leave policy
```

Per-message metrics are in the UI (click the metrics line under any agent
reply) and aggregated at `GET /health/metrics`.

---

## Not supported yet

**Agent-to-agent collaboration.** Mentioning two agents runs both against the
user's message, sequentially. There is no handoff, no shared scratchpad, and no
agent invoking another.

That is deliberate for the first milestone: one reliable agent is worth more
than three that hand each other half-formed conclusions. The seam is
`AgentService.handle_user_message`, which already iterates over a list of
targets.
