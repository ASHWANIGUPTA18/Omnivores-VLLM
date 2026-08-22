# Inference server

This directory runs vLLM as an OpenAI-compatible server. The Omnivores backend
talks to it over HTTP and knows nothing else about it.

```
Omnivores backend  ──HTTP /v1/chat/completions──▶  vLLM  ──▶  GPU
```

Nothing in `backend/` imports vLLM. Moving from the mock model to a real one is
three environment variables, which is the whole point of the
`InferenceClient` abstraction.

---

## Requirements

| | |
|---|---|
| GPU | NVIDIA, compute capability 7.0+ (Volta or newer) |
| VRAM | 16 GB for an 8B model at bf16; see `config/models.yaml` |
| Driver | 525+ |
| CUDA | 12.1+ |
| OS | Linux. vLLM does not support Windows natively - use WSL2 or a Linux host |
| Disk | 20-150 GB for weights, on a fast disk |

Check your GPU:

```bash
nvidia-smi
```

---

## Option A - native install

```bash
python3.11 -m venv .venv-vllm
source .venv-vllm/bin/activate

# Install the torch build matching your CUDA version FIRST.
# https://pytorch.org/get-started/locally/
pip install vllm

cp inference/.env.example inference/.env
# edit MODEL_NAME and TOOL_CALL_PARSER
./inference/start-vllm.sh
```

The launcher validates your configuration, prints the exact command it runs,
and refuses to start with a clear message rather than failing obscurely
mid-load.

## Option B - Docker

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
export MODEL_NAME=Qwen/Qwen3-8B
export TOOL_CALL_PARSER=hermes
docker compose -f inference/docker-compose.yml up
```

The first run downloads the weights. That can take a long time; the container's
healthcheck allows 15 minutes before reporting unhealthy.

---

## Connecting the backend

In `backend/.env`:

```bash
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8001/v1
MODEL_NAME=Qwen/Qwen3-8B     # must match what /v1/models reports
VLLM_API_KEY=                # only if you set one on the server
```

Verify end to end:

```bash
curl http://localhost:8001/v1/models        # the server itself
curl http://localhost:8000/health/inference # the backend's view of it
```

`MODEL_NAME` may be left empty - the client will auto-detect it from
`/v1/models`, since vLLM serves exactly one model per process.

---

## Tool calling

**Without this, the agents cannot use any tool.** No document search, no code
execution, no web search - the models will describe what they would do instead
of doing it.

vLLM needs two flags, which `start-vllm.sh` adds when `ENABLE_TOOL_CALLING=true`:

```
--enable-auto-tool-choice
--tool-call-parser <parser>
```

The parser is model-specific and its name varies between vLLM releases. Always
confirm against your own build:

```bash
vllm serve --help | grep -A10 tool-call-parser
```

`config/models.yaml` lists a starting parser per model family, marked
`verified: false` because your vLLM version is the authority, not this file.

To confirm tool calling actually works once the server is up:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "What is 17 times 23? Use the tool."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression",
        "parameters": {
          "type": "object",
          "properties": {"expression": {"type": "string"}},
          "required": ["expression"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

A working setup returns `tool_calls` in the response. If instead you get prose
describing the calculation, the parser is wrong or missing.

---

## Choosing a model

Omnivores is model-agnostic. What matters, in order:

1. **Tool calling reliability.** The agent loop is built on it.
2. **Instruction following.** The system prompts are long and specific.
3. **Context length.** RAG context plus channel history adds up quickly.
4. **VRAM.** Weights plus KV cache headroom.
5. **Throughput.** Tokens/sec at your expected concurrency.
6. **Coding ability.** Only the `@coder` agent depends on it.

Start with the largest model that leaves 20% VRAM headroom, confirm tool
calling works, then measure with `benchmarks/`.

---

## Tuning

| Symptom | Change |
|---|---|
| Out of memory at startup | Lower `MAX_MODEL_LEN` first, then `GPU_MEMORY_UTILIZATION` |
| Out of memory under load | Lower `MAX_NUM_SEQS` |
| Slow first token | Expected on a cold cache; check `MAX_NUM_SEQS` is not throttling |
| Low tokens/sec | Raise `MAX_NUM_SEQS`, or add GPUs and raise `TENSOR_PARALLEL_SIZE` |
| Model will not fit | Try `QUANTIZATION=fp8` on Hopper/Ada, or a smaller model |

`TENSOR_PARALLEL_SIZE` must divide the model's attention head count, which is
why the launcher only auto-selects powers of two.

Measure before and after:

```bash
python benchmarks/benchmark_inference.py --concurrency 1 4 8 16
```

---

## Troubleshooting

**`Connection refused` from the backend**
The server is not up, or is on another port. Check `curl http://localhost:8001/v1/models`.

**`ModelNotLoaded` / 404 on a model id**
`MODEL_NAME` in `backend/.env` does not match what the server reports. Compare
against `/v1/models`, or clear it and let the client auto-detect.

**`401` while downloading weights**
A gated repository. Accept the licence on Hugging Face and set `HF_TOKEN`.

**Crashes with more than one GPU**
Docker's default 64 MB `/dev/shm` is too small. The compose file sets
`shm_size: 16gb`; if running `docker run` by hand, add `--shm-size=16g`.

**Agents answer but never use tools**
Tool calling is not configured. See the section above.

---

## Security

Do not expose this port to the internet. vLLM's `--api-key` does not protect
every endpoint, so it is a convenience, not a boundary. Bind to localhost, or
put it behind a reverse proxy that terminates authentication. See
`docs/deployment.md`.
