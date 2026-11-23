# Running Omnivores on a Linux GPU box

Written for: **HP Z8 G5, Ubuntu 24.04.4 LTS, Xeon Silver 4416+, 128 GB RAM,
RTX A5000 24 GB (Ampere), driver 580.173.02, 6 TB disk.**

Target shape: vLLM serving `Qwen/Qwen3-8B`, local MiniLM embeddings, a Chroma
index, three built-in agents, two people chatting at once.

---

## 0. What you get

Three agents ship **built into the code** (`backend/app/agents/registry.py:188`),
so a fresh install is usable immediately. `backend/agents.d/` is an override
layer, and being empty is the expected state.

| Agent | Tools | Notes |
|---|---|---|
| `@researcher` | `search_knowledge_base`, `web_search` | Web search is **mock by default** - see the warning in step 2 |
| `@coder` | `run_python`, `search_knowledge_base` | `run_python` is disabled by default; see section 7 |
| `@hr` | `search_knowledge_base` | `auto_retrieve=true` - pulls document context without needing a tool call |

Four channels are pre-seeded: `#general` (all), `#research`, `#engineering`,
`#hr`. An agent not listed for a channel silently will not answer there.

---

## 1. Host prerequisites

```bash
nvidia-smi
```

```bash
docker --version && docker compose version
```

Install the NVIDIA Container Toolkit if `docker run --gpus` does not work, then
verify GPU passthrough actually reaches a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Your CUDA 13.0 host runs CUDA 12.x containers fine through driver
backward-compatibility, so the stock vLLM image is correct here.

Disk: keep **at least 100 GB** free on the Docker data root. Qwen3-8B weights
are ~16 GB and the vLLM base image is ~10 GB.

---

## 2. Configure

```bash
cp .env.example .env
```

The defaults are already tuned for a 24 GB Ampere card. Read the warnings in
that file before changing anything. The two that will cost you an afternoon:

- **Leave `API_KEY` empty.** The frontend bakes `VITE_API_KEY` in at build time
  and this stack passes none, so any value makes the whole UI 401 and closes
  the WebSocket with 1008. The backend is already bound to `127.0.0.1`.
- **`WEB_SEARCH_PROVIDER=mock` fabricates results.** It returns entries titled
  `[SYNTHETIC RESULT n]` at `https://example.invalid/...` and `@researcher`
  will cite them as though real. Either set `tavily` with a key, or do not ask
  `@researcher` anything web-facing.

Note that `fp8` quantization will not work on this card - it needs Hopper or
Ada. `awq`, `gptq` and `bitsandbytes` do work. Unquantized bf16 fits fine.

---

## 3. Verify the tool-call parser BEFORE you demo

This is the single highest-risk item. If `TOOL_CALL_PARSER` does not match the
model, **tool calling fails silently** - no error, no log. `@coder` and
`@researcher` produce confident prose about what they would do, with no tool
strip and no citations, while `@hr` still works because it retrieves without a
tool call. Nothing in the code detects this: `ModelInfo.supports_tools` is
hardcoded `True` (`backend/app/inference/base.py:242`), and the comment in
`backend/app/inference/vllm.py:1-17` claiming it is detected is not implemented.

Parser names change between vLLM releases. Confirm yours exists:

```bash
docker compose -f docker-compose.gpu.yml build vllm
```

```bash
docker run --rm --entrypoint vllm omnivores-vllm:latest serve --help | grep -A15 tool-call-parser
```

`hermes` must appear in that list for Qwen3. If it does not, pin
`VLLM_VERSION` in `.env` to a release where it does.

---

## 4. Start

```bash
docker compose -f docker-compose.gpu.yml up -d --build
```

```bash
docker compose -f docker-compose.gpu.yml logs -f vllm
```

The first run downloads ~16 GB of weights. The backend waits on the vLLM
healthcheck, which allows a 900-second start period. Wait for the
`/v1/models` healthcheck to pass.

```bash
docker compose -f docker-compose.gpu.yml ps
```

All three services should read healthy, and `nvidia-smi` should show roughly
21 GB in use.

---

## 5. Verify

```bash
python3 scripts/health-check.py --full
```

`--full` uploads a small Markdown file, polls until it indexes, reports the
chunk count and deletes it. It exercises the real ingestion path, not just a
liveness ping.

Then open <http://localhost:3000> and run the **tool-calling check**:

```
@coder compute 47**13
```

A tool strip must appear above the answer. If it does not, go back to step 3 -
the parser is wrong, and every other result you see will be ungrounded.

Then the retrieval check: drop a text-layer PDF into the sidebar, wait for
`ready`, and ask `@hr` something answerable only from that file. You should get
an answer with a numbered citation naming the file and page.

> **Your first upload will appear to hang.** The embedding model loads lazily
> on first use (`backend/app/rag/embeddings.py:194-219`), so ~90 MB of MiniLM
> downloads during that first document. It needs outbound network. Later
> uploads are fast.

---

## 6. The two-user test

Two browser tabs, two different questions, fired together - both should stream
at once. For an objective measurement:

```bash
pip install websockets
```

```bash
python3 scripts/concurrency-test.py
```

It opens two independent sockets, gates them so both questions start at the
same instant, and reports time-to-first-token for each plus how much the two
answer streams overlapped. Expect **overlap well above 50%**; serial execution
would show near zero.

Why this works, for reference: each message is dispatched as its own task
(`backend/app/api/websocket.py:317`), the send lock is per-connection
(`websocket.py:62`), the rate limit is 60/min *per connection*, every
CPU-bound stage already runs on `asyncio.to_thread`, and vLLM runs with
`--max-num-seqs 64`.

Two things worth knowing:

- Both tabs see the same workspace. `MessageService.broadcast()` fans every
  event to every subscriber and the frontend filters by channel - these are
  not private sessions.
- Mentioning **two agents in one message** is deliberately sequential
  (`backend/app/services/agent_service.py:126-133`). The concurrency is across
  users, not across agents within a single turn.

---

## 7. Optional: enable the @coder Python sandbox

Disabled by default because it **cannot work as shipped** - the backend image
has no `docker` CLI and runs as uid 10001, so the mounted socket is
permission-denied. To enable it you must do all four:

1. Add `docker-ce-cli` to the runtime stage of `backend/Dockerfile`.
2. Add `group_add` with the host docker gid to the backend service. Get it
   with `getent group docker` and take the third colon-separated field.
3. Uncomment the `/var/run/docker.sock` mount in `docker-compose.gpu.yml`.
4. Build the sandbox image - nothing builds it for you:

```bash
docker build -t omnivores-sandbox:latest backend/sandbox-image
```

Then set `PYTHON_SANDBOX_MODE=docker` in `.env` and rebuild the backend.

**This is a privileged mount.** Anything that executes inside the backend
container can control the host Docker daemon. Without it `@coder` still
answers - the tool reports that the sandbox is unavailable and the model
replies without running code.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| UI shows a `MOCK MODEL` badge | Backend fell back to mock providers | Confirm the env reached the container: `docker compose -f docker-compose.gpu.yml exec backend env` and grep for `LLM_PROVIDER` |
| Every request 401s, WebSocket closes 1008 | `API_KEY` is set | Empty it in `.env`, then `up -d` again |
| Agents answer but never call tools | Wrong `TOOL_CALL_PARSER` | Step 3 |
| `@researcher` cites `example.invalid` | Mock web search | Set `WEB_SEARCH_PROVIDER=tavily` plus a key, or avoid web questions |
| vLLM out-of-memory on start | KV cache does not fit | Lower `MAX_MODEL_LEN` to 16384, or `GPU_MEMORY_UTILIZATION` to 0.85 |
| Upload stalls at "embedding" | MiniLM downloading | Wait. Needs outbound network |
| Old documents return odd results | Vectors from a different embedding model | `docker compose -f docker-compose.gpu.yml down -v` to wipe volumes, then re-upload. Reindex is per-document only; there is no bulk endpoint |
| Same text appears as `[1]` and `[2]` | Known chunker bug (`backend/app/rag/chunker.py:155-174`) - overlap carries whole units on dense pages | Cosmetic for a demo. Affects tables, code blocks and single-paragraph pages |
| History silently forgotten in a long chat | Known bug (`backend/app/agents/context.py:159-168`) - the omission notice is dropped exactly when it is needed | Demo in a fresh channel, under ~20 messages |
| Messages vanish after a restart | In-memory by design (`backend/app/services/message_service.py:61`) | Expected. Document *vectors* survive in Chroma but the document list does not |

---

## 9. Shutdown

```bash
docker compose -f docker-compose.gpu.yml down
```

To also wipe the index and uploads:

```bash
docker compose -f docker-compose.gpu.yml down -v
```
