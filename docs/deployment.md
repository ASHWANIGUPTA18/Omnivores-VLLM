# Deployment

## Three configurations

| | Development | Single GPU box | Split |
|---|---|---|---|
| Model | mock | vLLM, same host | vLLM, separate host |
| Embeddings | mock | local | local |
| Vector store | memory | chroma | chroma |
| Sandbox | disabled | docker | docker |
| GPU | none | 1+ | on the model host |

---

## Development

```bash
./scripts/dev.sh
```

Creates the virtualenv, installs dependencies, copies `.env`, starts both
servers. No GPU, no model download, no API keys.

---

## Single GPU machine

### 1. Prerequisites

```bash
nvidia-smi                       # driver 525+, CUDA 12.1+
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

The third command must print your GPU. If it does not, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### 2. Choose a model

Pick from `inference/config/models.yaml` based on VRAM, then confirm the
tool-call parser exists in your vLLM build:

```bash
vllm serve --help | grep -A10 tool-call-parser
```

**Do not skip this.** A wrong parser means agents never call tools, and the
failure is silent - they describe what they would do instead of doing it.

### 3. Start everything

```bash
docker build -t omnivores-sandbox:latest backend/sandbox-image

export MODEL_NAME=Qwen/Qwen3-8B
export TOOL_CALL_PARSER=hermes
export API_KEY="$(openssl rand -hex 32)"

docker compose -f docker-compose.gpu.yml up -d
docker compose -f docker-compose.gpu.yml logs -f vllm
```

The first run downloads the weights. Wait for vLLM to report it is listening.

### 4. Verify

```bash
python scripts/health-check.py --full --api-key "$API_KEY"
```

This uploads a document, waits for it to index, and confirms it becomes
searchable - the path users care about, not just a liveness ping.

---

## Split deployment

On the GPU host:

```bash
cd inference
cp .env.example .env       # set MODEL_NAME, TOOL_CALL_PARSER, VLLM_API_KEY
docker compose up -d
```

Keep port 8001 on a private network. Then on the application host:

```bash
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://10.0.0.5:8001/v1
VLLM_API_KEY=<the same key>
MODEL_NAME=Qwen/Qwen3-8B
```

---

## Security checklist

Work through this before anything is reachable beyond localhost.

### Never expose vLLM

vLLM's `--api-key` does not protect every endpoint. Treat it as a convenience,
not a boundary.

```
  internet ──▶ reverse proxy (TLS, auth) ──▶ backend ──▶ vLLM
                                                   private network only
```

- [ ] vLLM binds to a private interface or is published only to `127.0.0.1`
- [ ] No inbound firewall rule reaches port 8001

### Authenticate the backend

- [ ] `API_KEY` is set to a long random value
- [ ] `CORS_ORIGINS` lists your exact frontend origins, never `*`
- [ ] TLS terminates at the proxy

The WebSocket takes its key as a query parameter, because browsers cannot set
headers on a handshake. Query strings land in access logs, so for a public
deployment terminate authentication at the proxy instead.

### Code execution

- [ ] `PYTHON_SANDBOX_MODE=docker`, never `subprocess` in production
- [ ] The sandbox image is built
- [ ] You accept the Docker socket mount, or set the mode to `disabled`

`subprocess` mode runs model-generated code on the host with the server's own
privileges. It is a development convenience and says so at startup.

**The Docker socket mount is a real trade-off.** The backend needs it to launch
sandbox containers, and it grants control of the host's Docker daemon to
anything that can execute inside the backend container. On a machine you do not
fully control, set `PYTHON_SANDBOX_MODE=disabled` and remove the mount - every
other feature keeps working.

### Uploads

- [ ] `MAX_UPLOAD_SIZE_MB` matches the proxy's body limit
- [ ] `ALLOWED_EXTENSIONS` is only what you need
- [ ] The upload volume is on a disk with a quota

Filenames are sanitized and files stored under a generated id, so a crafted
name cannot escape the directory. Size is enforced against the *streamed* bytes,
not the declared `Content-Length`.

### Operations

- [ ] `APP_ENV=production` (this activates the startup warnings)
- [ ] `LOG_FORMAT=json` if you ship logs
- [ ] `/health` is wired to your monitor
- [ ] `backend-data` is backed up - it holds uploads and the index

---

## Sizing

| Users | GPU | Model | `MAX_NUM_SEQS` |
|---|---|---|---|
| 1-3 | 1 × 24 GB | 8B | 16 |
| 5-15 | 1 × 48 GB | 8-14B | 64 |
| 20-50 | 2 × 48 GB | 14-32B | 128 |

Concurrency is bounded by KV cache, not compute. Halving `MAX_MODEL_LEN`
roughly doubles the sequences that fit.

Measure rather than guess:

```bash
python benchmarks/benchmark_inference.py --concurrency 1 4 8 16 32
```

Watch where aggregate output tok/s stops rising - that is your ceiling.

---

## Upgrades

```bash
git pull
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml up -d
python scripts/health-check.py --full
```

Uploads and the index live in named volumes and survive rebuilds.

**Re-index after changing `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`,
`CHUNK_SIZE_TOKENS` or `CHUNK_OVERLAP_TOKENS`.** Vectors from different models
or settings are not comparable:

```bash
for id in $(curl -s localhost:8000/api/documents | python -c \
    "import sys,json; print(' '.join(d['id'] for d in json.load(sys.stdin)['documents']))"); do
  curl -X POST "localhost:8000/api/documents/$id/reindex"
done
```

---

## Troubleshooting

| Symptom | Check |
|---|---|
| UI says "inference unavailable" | `curl localhost:8001/v1/models` |
| Agents never use tools | `--tool-call-parser` matches the model |
| `ModelNotLoaded` | `MODEL_NAME` matches `/v1/models`, or clear it to auto-detect |
| OOM at startup | Lower `MAX_MODEL_LEN`, then `GPU_MEMORY_UTILIZATION` |
| OOM under load | Lower `MAX_NUM_SEQS` |
| Crashes with 2+ GPUs | `shm_size: 16gb` is set |
| Code execution refused | `PYTHON_SANDBOX_MODE`, and the sandbox image exists |
| WebSocket drops every 60s | Proxy idle timeout below the 25s heartbeat |
| Streaming arrives in bursts | `proxy_buffering off` on the `/ws` location |
| 401 downloading weights | Gated repo - set `HF_TOKEN` |

Start here for anything unclear:

```bash
python scripts/health-check.py --full
curl localhost:8000/health/full | python -m json.tool
docker compose -f docker-compose.gpu.yml logs --tail 100 backend
```

---

## What is not production-ready yet

Honest limits, so nobody is surprised:

* **Messages are in-memory.** A backend restart loses conversation history.
  Needs Postgres behind `MessageService`.
* **Single process.** The WebSocket hub is in-memory; multiple workers need
  Redis pub/sub first.
* **One shared API key.** No users, no roles, no per-user document isolation.
* **Rate limiting is per-process.** Fine for one instance; put a real limiter
  at the proxy for more.

None of these block a team-internal deployment. All of them block a
multi-tenant one.
