# Inference

## The contract

```python
class InferenceClient(ABC):
    async def chat(messages, *, tools, temperature, max_tokens, ...) -> ChatResult
    def stream(messages, *, tools, ...) -> AsyncIterator[ChatChunk]
    async def count_tokens(messages) -> int
    async def health() -> InferenceHealth
    async def model_info() -> ModelInfo
```

Everything above this layer uses only these five methods. Swapping backends is
a change to `LLM_PROVIDER`.

The message and tool-call shapes mirror the OpenAI chat-completions schema,
because that is what vLLM exposes. That keeps the two real backends thin and
keeps the mock honest about what a real model does.

---

## Backends

| Provider | Use | Requires |
|---|---|---|
| `mock` | Development, CI, UI work | nothing |
| `vllm` | Production, local model | NVIDIA GPU |
| `openai` | Bridge, comparison baseline | API key |

`vllm` and `openai` both extend `OpenAICompatibleClient`, which holds the HTTP,
SSE parsing, retry and error-mapping logic. `vllm.py` adds only what genuinely
differs: model auto-detection, exact token counting via `/tokenize`, and
context-window reporting.

---

## The mock backend is not a stub

This is what makes the repository developable without a GPU, so it implements
the full contract rather than returning canned strings:

* Streams text word by word, as a real server does.
* Streams **tool calls in fragments** - id and name first, then arguments in
  pieces - so `StreamAccumulator` reassembly is exercised on every CI run. A
  bug in that reassembly would break vLLM and OpenAI identically, and this is
  where it gets caught.
* Reports usage, finish reasons, and health.
* Decides whether to call a tool from the conversation and the available tools,
  and answers *from the tool result* when one comes back - so RAG citations
  render with real retrieved text.
* Is a pure function of its input: two identical runs produce identical output.

### Test directives

Embed these in a user message to force specific behaviour:

```
[[mock:tool=search_knowledge_base]]           force a tool call
[[mock:tool=run_python args={"code":"1+1"}]]  force exact arguments
[[mock:text=exact reply]]                     force plain text
[[mock:error]]                                raise InferenceError
[[mock:timeout]]                              raise InferenceTimeout
[[mock:tokens=400]]                           pad the reply to ~400 tokens
```

Used throughout the test suite to drive failure paths deterministically.

---

## Streaming

```
  vLLM SSE                    ChatChunk                 StreamAccumulator
  ─────────                   ─────────                 ─────────────────
  data: {"delta":{...}}  ──▶  content="Hel"       ──▶   "Hel"
  data: {"delta":{...}}  ──▶  content="lo"        ──▶   "Hello"
  data: {"delta":{...}}  ──▶  tool_calls=[...]    ──▶   fragment buffer
  data: [DONE]                finish_reason="..."       ChatResult
```

Tool calls arrive piecewise and out of order: the `index` identifies which call
a fragment belongs to, and argument fragments must be concatenated in arrival
order. `StreamAccumulator` handles both, and orders the finished calls by index
rather than by arrival.

A fragment stream that never names a function is dropped rather than guessed
at - inventing a name would send the runtime looking for a tool that does not
exist.

---

## Errors

HTTP responses are mapped to typed errors so callers can act on them:

| Condition | Error | Retryable |
|---|---|---|
| Cannot connect | `InferenceUnavailable` | yes |
| 503, or 404 naming a model | `ModelNotLoaded` | yes |
| Timeout | `InferenceTimeout` | yes |
| 429 | `InferenceUnavailable` | yes |
| 400 mentioning context length | `ContextLengthExceeded` | no |
| 401/403 | `InferenceError` | no |

Retryable failures are retried with exponential backoff, `INFERENCE_MAX_RETRIES`
times, before surfacing.

---

## Model auto-detection

vLLM serves exactly one model per process, so `MODEL_NAME` may be left empty:

```
MODEL_NAME=            →  GET /v1/models  →  use the first entry
```

If you do set it and it does not match, health reports `degraded` with the list
of what *is* served - a much better failure than a 404 mid-conversation.

---

## Token counting

Two paths, deliberately:

* `core/tokens.py` estimates, dependency-free, and rounds **up**. Used for
  context budgeting, because the failure mode of underestimating (a rejected
  request mid-conversation) is far worse than overestimating (slightly less
  history).
* `InferenceClient.count_tokens` is authoritative. The vLLM client uses the
  server's `/tokenize` endpoint and the served model's own tokenizer, falling
  back to the estimator if the endpoint is absent.

---

## Metrics

Every call records:

```
Model: Qwen/Qwen3-8B
TTFT: 320 ms          time to first token - what makes the UI feel alive
Input: 2,104 tokens
Output: 387 tokens
Generation: 61 tok/s  measured after the first token, so prefill does not
                      depress it - this is decode speed
Tool calls: 2
Total: 7.1 s
```

Visible per-message in the UI and aggregated at `GET /health/metrics`.

---

## Adding a backend

```python
# app/inference/anthropic.py
class AnthropicClient(InferenceClient):
    provider = "anthropic"
    async def chat(self, messages, **kwargs) -> ChatResult: ...
    def stream(self, messages, **kwargs) -> AsyncIterator[ChatChunk]: ...
    async def count_tokens(self, messages) -> int: ...
    async def health(self) -> InferenceHealth: ...
    async def model_info(self) -> ModelInfo: ...
```

```python
# app/inference/client.py
if name == "anthropic":
    from app.inference.anthropic import AnthropicClient
    return AnthropicClient()
```

Add `"anthropic"` to the `LLMProvider` literal in `core/config.py`. Two files;
nothing above the layer changes.

Import inside the branch, not at module scope - selecting one backend must
never import another's dependencies.
