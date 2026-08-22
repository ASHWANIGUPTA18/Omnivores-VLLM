# Custom agent definitions

Drop a `.yaml` file here to add an agent, or to override a built-in one by
reusing its `id`. Loaded at startup; no code change and no restart of anything
but the backend.

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

  - Search before you answer, always.
  - Quote the figure and cite it.
  - If the documents do not cover it, say so and say who to ask. Never
    estimate a figure that ought to come from a document.
```

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Lowercased. This is the `@mention`. |
| `display_name` | yes | Shown in the UI |
| `system_prompt` | yes | Appended to the shared preamble |
| `description` | no | Shown in the roster and autocomplete |
| `tools` | no | Names from the tool registry; unknown ones are skipped |
| `avatar` / `color` | no | Emoji and hex colour |
| `temperature` | no | Overrides the global setting |
| `max_output_tokens` | no | Overrides the global setting |
| `max_tool_rounds` | no | Overrides the global setting |
| `auto_retrieve` | no | Fetch documents before the first model call |

An invalid file is logged and skipped - one bad definition never stops the
backend from starting.

See [`docs/agents.md`](../../docs/agents.md) for prompt-writing guidance.
