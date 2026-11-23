#!/usr/bin/env bash
# Preflight for the Omnivores GPU stack on a Linux box.
#
# Answers one question: will `docker compose -f docker-compose.gpu.yml up`
# actually work here? Checks only - changes nothing.
#
#   bash scripts/preflight-linux.sh

set -uo pipefail

PASS=0; WARN=0; FAIL=0
ok()   { printf '  \033[32m[ OK ]\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$*"; WARN=$((WARN+1)); }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

head_ "System"
if [ -r /etc/os-release ]; then
  . /etc/os-release
  ok "OS: ${PRETTY_NAME:-unknown}"
else
  warn "cannot read /etc/os-release"
fi
ok "Kernel: $(uname -r)   CPUs: $(nproc)   RAM: $(free -g 2>/dev/null | awk '/^Mem:/{print $2"GB"}')"

head_ "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)

  if [ -n "${GPU_NAME:-}" ]; then
    ok "GPU: ${GPU_NAME}  VRAM: ${VRAM_MB} MiB  driver: ${DRIVER}"
  else
    bad "nvidia-smi ran but reported no GPU"
  fi

  # vLLM needs driver 525+ for CUDA 12.x images.
  DRIVER_MAJOR=${DRIVER%%.*}
  if [ -n "${DRIVER_MAJOR:-}" ] && [ "${DRIVER_MAJOR:-0}" -ge 525 ] 2>/dev/null; then
    ok "driver ${DRIVER} is new enough (needs 525+)"
  else
    bad "driver ${DRIVER:-none} is too old - vLLM CUDA 12.x images need 525+"
  fi

  if [ "${VRAM_MB:-0}" -ge 22000 ] 2>/dev/null; then
    ok "VRAM fits Qwen3-8B bf16 (~16 GB weights + KV cache)"
  elif [ "${VRAM_MB:-0}" -ge 15000 ] 2>/dev/null; then
    warn "VRAM is tight for an 8B at 32k context - lower MAX_MODEL_LEN to 16384, or use an AWQ build"
  else
    bad "VRAM too small for an 8B model - pick a 3B, or a quantized build"
  fi

  if [ "${USED_MB:-0}" -gt 1500 ] 2>/dev/null; then
    warn "${USED_MB} MiB already in use on the GPU - another process may starve vLLM"
  else
    ok "GPU is essentially idle (${USED_MB:-0} MiB used)"
  fi

  case "${GPU_NAME:-}" in
    *H100*|*L40*|*L4*|*ADA*|*Ada*|*4090*) ok "fp8 quantization is available on this architecture" ;;
    *) warn "fp8 not supported here (needs Hopper/Ada) - use awq/gptq or unquantized" ;;
  esac
else
  bad "nvidia-smi not found - no NVIDIA driver installed"
fi

head_ "Docker"
if command -v docker >/dev/null 2>&1; then
  ok "docker CLI: $(docker --version 2>/dev/null)"
  if docker info >/dev/null 2>&1; then
    ok "docker daemon reachable as $(id -un)"
  else
    bad "cannot talk to the docker daemon as $(id -un) - add yourself to the 'docker' group (needs sudo) or use sudo"
  fi
  if docker compose version >/dev/null 2>&1; then
    ok "compose v2: $(docker compose version --short 2>/dev/null)"
  else
    bad "'docker compose' (v2) not available - the compose file needs v2 syntax"
  fi
else
  bad "docker not installed"
fi

head_ "NVIDIA Container Toolkit (GPU inside containers)"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    ok "containers can see the GPU"
  else
    bad "containers CANNOT see the GPU - install nvidia-container-toolkit (needs sudo), then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  fi
else
  warn "skipped - docker unavailable"
fi

head_ "Disk"
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
AVAIL_G=$(df -BG --output=avail "$DOCKER_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${AVAIL_G:-}" ]; then
  if [ "$AVAIL_G" -ge 100 ]; then
    ok "${AVAIL_G} GB free on ${DOCKER_ROOT}"
  elif [ "$AVAIL_G" -ge 50 ]; then
    warn "${AVAIL_G} GB free on ${DOCKER_ROOT} - tight (weights ~16 GB + vLLM image ~10 GB + layers)"
  else
    bad "only ${AVAIL_G} GB free on ${DOCKER_ROOT} - need ~100 GB"
  fi
else
  warn "could not measure free space on ${DOCKER_ROOT}"
fi

head_ "Network (first run downloads ~16 GB of weights)"
if curl -fsS --max-time 10 -o /dev/null https://huggingface.co 2>/dev/null; then
  ok "huggingface.co reachable"
else
  bad "cannot reach huggingface.co - a proxy or campus firewall will block the model download"
fi
if curl -fsS --max-time 10 -o /dev/null https://registry-1.docker.io/v2/ 2>/dev/null \
   || curl -fsS --max-time 10 -o /dev/null https://hub.docker.com 2>/dev/null; then
  ok "Docker Hub reachable"
else
  warn "Docker Hub may be unreachable - image pulls could fail"
fi
if [ -n "${HTTP_PROXY:-}${HTTPS_PROXY:-}${http_proxy:-}${https_proxy:-}" ]; then
  warn "a proxy is set in the environment - Docker needs its own proxy config too (/etc/systemd/system/docker.service.d/)"
fi

head_ "Ports"
for port in 3000 8000; do
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${port} "; then
    bad "port ${port} is already in use"
  else
    ok "port ${port} is free"
  fi
done

head_ "Repo"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$ROOT/docker-compose.gpu.yml" ] && ok "docker-compose.gpu.yml present" || bad "docker-compose.gpu.yml missing - wrong directory?"
if [ -f "$ROOT/.env" ]; then
  ok ".env present"
  MODEL=$(grep -E '^MODEL_NAME=' "$ROOT/.env" | cut -d= -f2-)
  PARSER=$(grep -E '^TOOL_CALL_PARSER=' "$ROOT/.env" | cut -d= -f2-)
  [ -n "${MODEL:-}" ] && ok "MODEL_NAME=${MODEL}" || bad "MODEL_NAME is empty - compose will refuse to start"
  [ -n "${PARSER:-}" ] && ok "TOOL_CALL_PARSER=${PARSER}" || bad "TOOL_CALL_PARSER is empty - compose will refuse to start"
  if grep -qE '^API_KEY=.+' "$ROOT/.env"; then
    bad "API_KEY is set - the frontend bakes its key in at BUILD time and this stack passes none, so the whole UI will 401. Leave it empty."
  else
    ok "API_KEY is empty (correct for this stack)"
  fi
else
  bad ".env missing - run: cp .env.example .env"
fi
if command -v python3 >/dev/null 2>&1; then
  ok "python3: $(python3 --version 2>&1)  (for scripts/health-check.py)"
else
  warn "python3 not found - scripts/health-check.py and the concurrency test need it"
fi

head_ "Verdict"
printf '  %d passed, %d warnings, %d failures\n\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf '  \033[31mNOT READY.\033[0m Fix the [FAIL] lines above first.\n'
  printf '  Anything needing sudo (docker group, nvidia-container-toolkit) may\n'
  printf '  require your lab admin on a managed machine.\n\n'
  exit 1
fi
printf '  \033[32mREADY to try.\033[0m Next:\n'
printf '    docker compose -f docker-compose.gpu.yml build vllm\n'
printf '    docker run --rm --entrypoint vllm omnivores-vllm:latest serve --help | grep -A15 tool-call-parser\n'
printf '  Confirm your TOOL_CALL_PARSER appears in that list - if it does not,\n'
printf '  tool calling fails silently. Then see docs/RUNBOOK-GPU.md step 4.\n\n'
exit 0
