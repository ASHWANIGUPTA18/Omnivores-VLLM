#!/usr/bin/env bash
# Omnivores AI - native launcher (no Docker).
#
#   ./run-native.sh          start vLLM, backend and frontend, in order
#   ./run-native.sh stop     stop all three
#   ./run-native.sh status   what is running
#   ./run-native.sh logs     tail all three logs
#
# This machine has no Docker daemon access, so everything runs from the conda
# env at .conda/. Safe to re-run: already-healthy services are left alone.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVBIN="$ROOT/.conda/bin"
export PATH="$ENVBIN:$PATH"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8001}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
# 0.0.0.0 lets other people on the network reach the UI. There is no auth,
# so anyone who can route here can read every uploaded document.
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '    %s*%s %s\n' "$G" "$N" "$*"; }
note() { printf '    %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n    %s%s%s\n\n' "$R" "$*" "$N"; exit 1; }

up_vllm()     { curl -fsS --max-time 3 "http://127.0.0.1:$VLLM_PORT/v1/models"  >/dev/null 2>&1; }
up_backend()  { curl -fsS --max-time 3 "http://127.0.0.1:$BACKEND_PORT/health"  >/dev/null 2>&1; }
up_frontend() { curl -fsS --max-time 3 "http://127.0.0.1:$FRONTEND_PORT/"       >/dev/null 2>&1; }

case "${1:-start}" in
  stop)
    step "Stopping"
    pkill -f "vllm serve"      2>/dev/null && ok "vLLM stopped"     || note "vLLM was not running"
    pkill -f "uvicorn app.main" 2>/dev/null && ok "backend stopped" || note "backend was not running"
    pkill -f "vite"            2>/dev/null && ok "frontend stopped" || note "frontend was not running"
    exit 0 ;;
  status)
    up_vllm     && ok "vLLM      UP   (:$VLLM_PORT)"     || note "vLLM      DOWN"
    up_backend  && ok "backend   UP   (:$BACKEND_PORT)"  || note "backend   DOWN"
    up_frontend && ok "frontend  UP   (:$FRONTEND_PORT)" || note "frontend  DOWN"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/    GPU: /'
    exit 0 ;;
  logs)
    tail -f /tmp/vllm.log /tmp/backend.log /tmp/frontend.log
    exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|stop|status|logs]"; exit 2 ;;
esac

printf '%s\n  Omnivores AI (native)%s\n' "$B" "$N"

# --- 1. vLLM ----------------------------------------------------------------
step "vLLM  ($MODEL)"
if up_vllm; then
  ok "already running"
else
  # PATH must include the env bin: vLLM shells out to `ninja` during kernel
  # warmup and dies with FileNotFoundError if it is not there.
  setsid env PATH="$ENVBIN:$PATH" "$ENVBIN/vllm" serve "$MODEL" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --served-model-name "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --reasoning-parser "$REASONING_PARSER" \
    </dev/null >/tmp/vllm.log 2>&1 &
  disown
  printf '    loading weights'
  for _ in $(seq 1 90); do up_vllm && break; printf '.'; sleep 5; done
  echo
  up_vllm || die "vLLM did not start. Check: tail -40 /tmp/vllm.log"
  ok "up on :$VLLM_PORT"
fi
grep -m1 "Maximum concurrency" /tmp/vllm.log 2>/dev/null | sed 's/.*INFO/    /' || true

# --- 2. backend -------------------------------------------------------------
step "Backend"
if up_backend; then
  ok "already running"
else
  cd "$ROOT/backend"
  setsid env PATH="$ENVBIN:$PATH" "$ENVBIN/python" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" \
    </dev/null >/tmp/backend.log 2>&1 &
  disown
  printf '    starting'
  for _ in $(seq 1 40); do up_backend && break; printf '.'; sleep 2; done
  echo
  up_backend || die "backend did not start. Check: tail -40 /tmp/backend.log"
  ok "up on :$BACKEND_PORT"
fi
curl -fsS "http://127.0.0.1:$BACKEND_PORT/health" 2>/dev/null | sed 's/^/    /'; echo

# --- 3. frontend ------------------------------------------------------------
step "Frontend"
if up_frontend; then
  ok "already running"
else
  cd "$ROOT/frontend"
  setsid env PATH="$ENVBIN:$PATH" npm run dev -- \
    --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" \
    </dev/null >/tmp/frontend.log 2>&1 &
  disown
  printf '    starting'
  for _ in $(seq 1 30); do up_frontend && break; printf '.'; sleep 2; done
  echo
  up_frontend || die "frontend did not start. Check: tail -40 /tmp/frontend.log"
  ok "up on :$FRONTEND_PORT"
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
printf '\n%s  Ready.%s\n' "$G" "$N"
printf '    local      http://localhost:%s\n' "$FRONTEND_PORT"
[ -n "$IP" ] && [ "$FRONTEND_HOST" = "0.0.0.0" ] && \
  printf '    network    http://%s:%s      (no auth - anyone who can reach it reads every document)\n' "$IP" "$FRONTEND_PORT"
printf '\n    ./run-native.sh status | logs | stop\n\n'
