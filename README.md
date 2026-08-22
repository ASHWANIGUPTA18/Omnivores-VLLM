# Omnivores AI

A local-first multi-agent AI workspace. Agents are first-class chat
participants you address by `@mention`, they answer from documents you upload
with real citations, and the model runs on your own hardware through vLLM.

```
┌──────────────┬────────────────────────────────────────────┐
│ # general    │ # hr                                       │
│ # research   │                                            │
│ # hr        ●│  Ashwani  @hr how many days of leave do I  │
│ # engineering│           get?                             │
│              │                                            │
│ Documents    │  HR Assistant 🤖                           │
│              │  ├─ ✓ Searching documents  “leave”  41ms   │
│ 📄 handbook  │                                            │
│    42 pages  │  Full-time employees receive 20 days of    │
│    1,284 ch. │  paid annual leave per calendar year. [1]  │
│              │                                            │
│ 📄 policy.pdf│  Sources                                   │
│    indexing  │  1  handbook.pdf   page 14          92%    │
│    ████░░ 62%│                                            │
└──────────────┴────────────────────────────────────────────┘
```

---

## What it is

* **Agents as chat participants.** `@researcher`, `@coder`, `@hr` live in
  channels alongside people. Adding `@finance` is a YAML file, not a code
  change.
* **Document RAG with citations.** Upload a PDF, Word file, Markdown or text
  file; ask about it; get an answer that cites the filename and page.
* **Streaming everything.** Tokens, tool status, and document indexing progress
  all arrive live over one WebSocket.
* **Real tool calling.** Document search, sandboxed Python execution, web
  search - bounded by rounds, call count and timeouts.
* **A pluggable inference backend.** vLLM, OpenAI, or a deterministic mock. The
  application never imports any of them directly.

## What makes it portable

**No GPU is required to develop it.** The default configuration runs the entire
application - WebSocket, agent loop, tool dispatch, document ingestion,
retrieval, citations, the whole UI - against a deterministic mock model and a
dependency-free hashed embedding provider.

Moving to a GPU workstation is a configuration change:

```diff
- LLM_PROVIDER=mock
+ LLM_PROVIDER=vllm
+ VLLM_BASE_URL=http://localhost:8001/v1
+ MODEL_NAME=Qwen/Qwen3-8B
- EMBEDDING_PROVIDER=mock
+ EMBEDDING_PROVIDER=local
- VECTOR_STORE=memory
+ VECTOR_STORE=chroma
```

No source file changes. That property is enforced by the architecture: the only
places a provider name is resolved to a class are six factory functions.

---

## Quick start

### No GPU (any machine)

```bash
git clone https://github.com/ASHWANIGUPTA18/Omnivores-VLLM.git
cd Omnivores-VLLM

./scripts/dev.sh          # macOS / Linux
scripts\dev.bat           # Windows
```

Open <http://localhost:5173>, then try:

```
@coder write a function that reverses a linked list
```

Drop a PDF into the sidebar, wait for it to say `ready`, then:

```
@hr what does this document say about the notice period?
```

Responses come from the mock model, so the *content* is synthetic - but the
routing, streaming, retrieval, citations and tool dispatch around it are all
real.

### With Docker (still no GPU)

```bash
docker compose up
# http://localhost:3000
```

### With a GPU

```bash
# 1. Start vLLM
cp inference/.env.example inference/.env   # set MODEL_NAME, TOOL_CALL_PARSER
./inference/start-vllm.sh

# 2. Point the backend at it
cp backend/.env.example backend/.env
#    LLM_PROVIDER=vllm
#    MODEL_NAME=<the model you served>
#    EMBEDDING_PROVIDER=local
#    VECTOR_STORE=chroma
#    PYTHON_SANDBOX_MODE=docker

# 3. Verify
python scripts/health-check.py --full
```

Or the whole stack at once:

```bash
export MODEL_NAME=Qwen/Qwen3-8B
export TOOL_CALL_PARSER=hermes
docker build -t omnivores-sandbox:latest backend/sandbox-image
docker compose -f docker-compose.gpu.yml up
```

See [`inference/README.md`](inference/README.md) for GPU setup, model choice
and tool-calling configuration.

---

## Architecture

```
                    ┌──────────────────┐
                    │     React UI     │
                    │  chat · agents   │
                    │  documents       │
                    └────────┬─────────┘
                             │ WebSocket (one event protocol)
                    ┌────────▼─────────┐
                    │     FastAPI      │
                    │  message · agent │
                    │  document svc    │
                    └────────┬─────────┘
                    ┌────────▼─────────┐         ┌──────────────────┐
                    │  Agent Runtime   │◀───────▶│   Document RAG   │
                    │  context budget  │         │ parse · chunk    │
                    │  tool registry   │         │ embed · retrieve │
                    │  bounded loop    │         │ rerank · cite    │
                    └────────┬─────────┘         └──────────────────┘
                    ┌────────▼─────────┐         ┌──────────────────┐
                    │ InferenceClient  │         │  Python Sandbox  │
                    └────────┬─────────┘         │ container · caps │
              ┌──────────────┼──────────────┐    │ no network       │
              ▼              ▼              ▼    └──────────────────┘
           vLLM/GPU       OpenAI          Mock
```

Full detail in [`docs/architecture.md`](docs/architecture.md).

---

## The agent loop

```
build context (token-budgeted)
    │
    ▼
stream from the model ──── tool calls? ──no──▶ done
    │                           │
    │                          yes
    │                           ▼
    │                   validate arguments
    │                   check permissions
    │                   execute with timeout
    │                   append results
    └───────────────────────────┘
        bounded by MAX_TOOL_ROUNDS,
        MAX_TOOL_CALLS, AGENT_TIMEOUT_SECONDS
```

Running out of rounds is not a failure: the runtime makes one final call with
tools withheld so the user always gets an answer.

---

## Repository layout

```
backend/
  app/
    api/          WebSocket + REST, auth, rate limiting
    agents/       registry, runtime loop, context budgeting, mentions
    inference/    InferenceClient and its three backends
    tools/        tool contract, validation, guarded executor
    rag/          parse → chunk → embed → store → retrieve → rerank
    sandbox/      docker / subprocess / disabled execution
    core/         config, models, events, errors, metrics
    services/     messages, documents, agent dispatch
  tests/          unit + integration, all offline
inference/        vLLM server: launcher, Dockerfile, model catalog
frontend/         React UI, zero runtime dependencies beyond React
benchmarks/       inference, agent and RAG benchmarks
docs/             architecture, inference, rag, agents, deployment
scripts/          dev, test, health-check
```

---

## Configuration

Everything is environment-driven. `backend/.env.example` documents every value.
The ones that decide what the system *is*:

| Variable | Options | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `mock` `vllm` `openai` | Which model serves inference |
| `EMBEDDING_PROVIDER` | `mock` `local` `openai` | How documents are embedded |
| `VECTOR_STORE` | `memory` `chroma` | Where the index lives |
| `RERANKER` | `none` `lexical` `cross_encoder` | How results are reordered |
| `PYTHON_SANDBOX_MODE` | `disabled` `docker` `subprocess` | How code executes |
| `WEB_SEARCH_PROVIDER` | `mock` `tavily` | Which search backend |

`mock` everywhere is the default, which is why a fresh clone runs anywhere.

---

## Testing

```bash
./scripts/test.sh              # ruff + mypy + pytest
./scripts/test.sh tests/unit   # a subset
```

The whole suite runs offline against the mock providers - no GPU, no model
download, no API key - so CI is the same as your laptop.

---

## Benchmarking

```bash
python benchmarks/benchmark_inference.py --concurrency 1 4 8 16
python benchmarks/benchmark_agents.py
python benchmarks/benchmark_rag.py --documents 50
```

Every agent run also records its own metrics, visible per-message in the UI and
aggregated at `GET /health/metrics`.

---

## Security

* Model-generated Python never executes in the backend process. There is no
  `exec()` anywhere in the codebase. The default sandbox mode is `disabled`.
* Uploads are validated by extension and by actual streamed size, and stored
  under a generated id so a crafted filename cannot escape the directory.
* Markdown from models is rendered to React elements, never via
  `dangerouslySetInnerHTML`.
* Do not expose vLLM to the internet - its `--api-key` does not cover every
  endpoint.

Details and a pre-deployment checklist in [`docs/deployment.md`](docs/deployment.md).

---

## Documentation

| | |
|---|---|
| [architecture.md](docs/architecture.md) | How the pieces fit and why |
| [inference.md](docs/inference.md) | The inference abstraction and backends |
| [rag.md](docs/rag.md) | The document pipeline end to end |
| [agents.md](docs/agents.md) | Writing and tuning agents |
| [deployment.md](docs/deployment.md) | GPU setup, Docker, hardening |
| [inference/README.md](inference/README.md) | Running vLLM |

---

## Current limitations

Stated plainly, because knowing where the edges are matters more than a longer
feature list:

* **Messages are in-memory.** They survive a page reload, not a backend
  restart. Postgres is the obvious next step; the seam is `MessageService`.
* **Agents do not talk to each other.** Mentioning two agents runs both against
  the user's message, sequentially. There is no agent-to-agent handoff.
* **No OCR.** A scanned PDF with no text layer is rejected with a clear message
  rather than silently indexing nothing.
* **Single-process.** The WebSocket hub is in-memory, so running multiple
  workers needs a pub/sub layer first.
* **No user accounts.** Authentication is a single shared API key.

---

## Licence

MIT
