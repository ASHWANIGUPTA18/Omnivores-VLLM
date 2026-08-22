# Architecture

## The one decision everything else follows from

The application must never depend on vLLM.

```
        NOT THIS                              THIS
   ┌──────────────┐                   ┌──────────────────┐
   │  Omnivores   │                   │    Omnivores     │
   │              │                   └────────┬─────────┘
   │  vLLM calls  │                            ▼
   │  everywhere  │                   ┌──────────────────┐
   └──────────────┘                   │ InferenceClient  │
                                      └────────┬─────────┘
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                                  vLLM      OpenAI      Mock
```

Everything above `InferenceClient` - agents, tools, RAG, the WebSocket layer -
knows only that interface. `grep -r "import vllm" backend/` returns nothing,
and `openai` appears only inside `app/inference/openai.py`.

This is what makes the repository developable on a laptop and deployable on a
GPU workstation without a rewrite, and it is the property most worth protecting
in review.

---

## Layers

```
┌───────────────────────────────────────────────────────────────┐
│  api/          WebSocket · REST · auth · rate limiting        │
├───────────────────────────────────────────────────────────────┤
│  services/     messages · documents · agent dispatch          │
├───────────────────────────────────────────────────────────────┤
│  agents/       registry · runtime loop · context · mentions   │
├───────────────────────────────────────────────────────────────┤
│  tools/        contract · validation · guarded executor       │
├───────────────────────────────────────────────────────────────┤
│  inference/    │  rag/              │  sandbox/               │
│  the model     │  the documents     │  the code execution     │
├───────────────────────────────────────────────────────────────┤
│  core/         config · models · events · errors · metrics    │
└───────────────────────────────────────────────────────────────┘
```

Dependencies point downward only. `core/` imports nothing from the application;
`api/` imports from everything.

---

## Request path

A user sends `@hr what is the leave policy?` in `#hr`.

```
 browser
   │  {"type":"client.send_message", "data":{...}}
   ▼
 api/websocket.py
   │  rate limit → validate → store → broadcast message.created
   │  dispatch as a task, so the receive loop stays responsive
   ▼
 services/agent_service.py
   │  parse @mentions → check channel membership
   │  create the placeholder message the agent streams into
   ▼
 agents/runtime.py
   │  auto-retrieve (hr has auto_retrieve: true)
   │      └──▶ rag/retriever.py ──▶ embed → search → rerank → context
   │  build context within the token budget
   │  stream from InferenceClient
   │      ├── text delta ──▶ message.delta ──▶ browser
   │      └── tool call  ──▶ tools/registry.py
   │                            validate args → check permission
   │                            → execute with timeout
   │                            → agent.tool_started/completed
   │  loop until no tool calls, or the round limit
   ▼
 services/agent_service.py
   │  complete the stored message, attach citations and metrics
   │  broadcast message.completed
   ▼
 browser
```

---

## The event protocol

One envelope crosses the socket:

```json
{"type": "message.delta", "ts": "...", "data": {...}}
```

Defining it explicitly in `core/events.py` - rather than scattering
`json.dumps` calls - is what lets the frontend be a single reducer over a
closed set of cases.

| Direction | Events |
|---|---|
| server → client | `connection.ready` `error` `pong` |
| | `message.created` `.delta` `.completed` `.failed` |
| | `agent.started` `.thinking` `.tool_started` `.tool_completed` `.completed` `.failed` |
| | `document.uploaded` `.processing` `.ready` `.failed` `.deleted` |
| | `typing.started` `.stopped` · `channel.history` |
| client → server | `client.send_message` `.subscribe` `.typing` `.ping` `.cancel` |

---

## The agent loop

```
                    ┌──────────────────┐
                    │  build context   │  system + RAG + history + question,
                    └────────┬─────────┘  inside MAX_CONTEXT_TOKENS
                             ▼
                    ┌──────────────────┐
             ┌─────▶│ stream from model│──── text ────▶ message.delta
             │      └────────┬─────────┘
             │               ▼
             │        ┌─────────────┐
             │        │ tool calls? │──── no ────▶ done
             │        └──────┬──────┘
             │             yes
             │               ▼
             │      ┌──────────────────┐
             │      │ validate · permit│
             │      │ execute · timeout│
             │      └────────┬─────────┘
             └────────────────┘
                  round < MAX_TOOL_ROUNDS
```

Four independent bounds: `MAX_TOOL_ROUNDS`, `MAX_TOOL_CALLS`,
`TOOL_TIMEOUT_SECONDS`, `AGENT_TIMEOUT_SECONDS`.

Two behaviours worth knowing:

**Hitting the round limit produces an answer, not an error.** The runtime makes
one final call with tools withheld, forcing the model to answer from what it
gathered.

**A tool failure is not a turn failure.** Every failure mode - an exception, a
timeout, a missing tool, invalid arguments, a permission denial - becomes a
tool message the model reads and can recover from on the next round.

---

## Context budgeting

Sending "the last N messages" overflows when messages are long and wastes the
window when they are short. `agents/context.py` budgets instead:

```
  MAX_CONTEXT_TOKENS
  ├── system prompt          always
  ├── tool schemas           reserved
  ├── document context       up to CONTEXT_RAG_TOKEN_BUDGET
  ├── current question       always
  ├── recent history         fills the remainder, newest first
  └── output reserve         MAX_OUTPUT_TOKENS held back
```

Two details that matter more than they look:

* When history is dropped, the model is **told** - `[12 earlier messages were
  omitted]`. A model that silently loses the start of a conversation gives
  confusingly amnesiac answers.
* Another agent's replies enter as `user` turns labelled with their name, not
  as `assistant` turns. Presenting another model's output as this model's own
  history makes it defend positions it never took.

---

## Retrieval

```
  query
    │
    ├── scope: explicit documents > channel > everything
    ▼
  embed ──▶ vector search (RAG_TOP_K)
              │
              ▼
            rerank (RAG_RERANK_TOP_K)
              │
              ▼
            numbered, token-bounded context   ──▶  [1] handbook.pdf, page 14
              │
              ▼
            citations that match those numbers
```

Scoping is decided by the application, not the model: the model chooses the
*query*, the application chooses the *scope*. A user who scoped a question to
one document cannot have that widened by a model deciding otherwise.

Detail in [`rag.md`](rag.md).

---

## Errors

Every layer raises a typed `OmnivoresError` carrying a stable `code`, an
HTTP status, a `user_message` safe to display, and a `retryable` flag.

```
OmnivoresError
├── InferenceError ── InferenceUnavailable · ModelNotLoaded · InferenceTimeout
│                     ContextLengthExceeded
├── ToolError ─────── UnknownTool · InvalidToolArguments · ToolTimeout
│                     ToolPermissionDenied
│                     └── SandboxError ── SandboxDisabled · SandboxTimeout
├── DocumentError ─── UnsupportedFileType · FileTooLarge · DocumentParseError
├── AgentError ────── UnknownAgent · AgentTimeout
└── AuthenticationError · RateLimitExceeded · ValidationError
```

The rule this enforces: **a dead vLLM must never look like a hung UI.** The
socket stays open, the error arrives as a typed event, and the user is told
what happened.

---

## What is deliberately simple

These are choices, not omissions. Each has a seam for when it stops being
enough.

| Simplification | Why | Seam |
|---|---|---|
| In-memory messages | Persistence adds ops weight before it adds capability | `MessageService` |
| In-memory WebSocket hub | One process is enough at this scale | `MessageService.broadcast` |
| Brute-force vector scan by default | Correct to tens of thousands of chunks, zero dependencies | `VectorStore` |
| Token estimation, not a tokenizer | Avoids a tokenizer dependency; the server reports the real count | `InferenceClient.count_tokens` |
| Sequential multi-agent replies | Two agents streaming at once is unreadable | `AgentService.handle_user_message` |
| Single shared API key | No user model exists yet | `api/deps.py` |

---

## Extending it

| To add | Do this | Files touched |
|---|---|---|
| An agent | Write a YAML definition in `backend/agents.d/` | 0 |
| A tool | Subclass `Tool`, register it | 2 |
| A document format | Subclass `DocumentParser`, register it | 1 |
| An inference backend | Subclass `InferenceClient`, add a factory branch | 2 |
| A vector store | Subclass `VectorStore`, add a factory branch | 2 |
| An embedding provider | Subclass `EmbeddingProvider`, add a factory branch | 2 |

The counts are the point. If adding an agent required touching the runtime, the
abstraction would have failed.
