#!/usr/bin/env bash
#
# Launch vLLM as an OpenAI-compatible server for Omnivores.
#
# Everything is driven by environment variables (see inference/.env.example),
# so the same script serves any model on any GPU count without editing.
#
#   ./start-vllm.sh
#   MODEL_NAME=Qwen/Qwen3-14B TENSOR_PARALLEL_SIZE=2 ./start-vllm.sh
#
# The script deliberately does two things before launching:
#
#   1. Validates the environment - a missing MODEL_NAME or an unrecognised
#      tool-call parser is caught here with a clear message, rather than 90
#      seconds into a model load.
#   2. Prints the exact command it is about to run, so a failure can be
#      reproduced and debugged by hand.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load inference/.env if present. Values already in the environment win.
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
  set +a
fi

# --- configuration ---------------------------------------------------------

MODEL_NAME="${MODEL_NAME:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME}}"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8001}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-auto}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
DTYPE="${DTYPE:-auto}"
QUANTIZATION="${QUANTIZATION:-}"

ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-true}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

VLLM_API_KEY="${VLLM_API_KEY:-}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- helpers ---------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }
info() { echo "  $*"; }

# --- preflight -------------------------------------------------------------

command -v vllm >/dev/null 2>&1 || die \
  "vllm is not installed or not on PATH.
  Install it in a virtualenv with a CUDA-matched torch:
    pip install vllm
  See https://docs.vllm.ai/en/latest/getting_started/installation.html"

[[ -n "${MODEL_NAME}" ]] || die \
  "MODEL_NAME is not set.
  Pick one from inference/config/models.yaml, then:
    export MODEL_NAME=Qwen/Qwen3-8B
  or copy inference/.env.example to inference/.env and set it there."

# Detect GPUs so tensor parallelism can default sensibly. vLLM requires the
# tensor-parallel size to divide the attention head count, so anything other
# than a power of two is a poor automatic choice.
detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

GPU_COUNT="$(detect_gpu_count)"

if [[ "${TENSOR_PARALLEL_SIZE}" == "auto" ]]; then
  case "${GPU_COUNT}" in
    0) die "No NVIDIA GPU detected (nvidia-smi found nothing).
  vLLM needs a GPU. To develop without one, leave the backend on
  LLM_PROVIDER=mock and do not run this script." ;;
    1) TENSOR_PARALLEL_SIZE=1 ;;
    2|3) TENSOR_PARALLEL_SIZE=2 ;;
    4|5|6|7) TENSOR_PARALLEL_SIZE=4 ;;
    *) TENSOR_PARALLEL_SIZE=8 ;;
  esac
  info "auto-selected tensor-parallel-size=${TENSOR_PARALLEL_SIZE} from ${GPU_COUNT} GPU(s)"
fi

if (( GPU_COUNT > 0 && TENSOR_PARALLEL_SIZE > GPU_COUNT )); then
  die "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} but only ${GPU_COUNT} GPU(s) are visible."
fi

if [[ "${ENABLE_TOOL_CALLING}" == "true" && -z "${TOOL_CALL_PARSER}" ]]; then
  die "Tool calling is enabled but TOOL_CALL_PARSER is empty.

  vLLM cannot parse tool calls without a parser matched to the model, and
  without tool calls the Omnivores agents cannot search documents or run code.

  Find the right value for your model in inference/config/models.yaml, then
  confirm it exists in your vLLM build:
      vllm serve --help | grep -A10 tool-call-parser

  To run the model without tools anyway, set ENABLE_TOOL_CALLING=false."
fi

# --- build the command -----------------------------------------------------

ARGS=(
  serve "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --dtype "${DTYPE}"
)

if [[ "${ENABLE_TOOL_CALLING}" == "true" ]]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

[[ -n "${QUANTIZATION}" ]]   && ARGS+=(--quantization "${QUANTIZATION}")
[[ -n "${CHAT_TEMPLATE}" ]]  && ARGS+=(--chat-template "${CHAT_TEMPLATE}")
[[ -n "${VLLM_API_KEY}" ]]   && ARGS+=(--api-key "${VLLM_API_KEY}")
[[ -n "${DOWNLOAD_DIR}" ]]   && ARGS+=(--download-dir "${DOWNLOAD_DIR}")
[[ "${TRUST_REMOTE_CODE}" == "true" ]] && ARGS+=(--trust-remote-code)

# EXTRA_ARGS is intentionally word-split: it is an escape hatch for flags this
# script does not model, and it comes from the operator, not from user input.
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(${EXTRA_ARGS})
fi

# --- report ----------------------------------------------------------------

cat <<BANNER

======================================================================
 Omnivores - vLLM inference server
======================================================================
  model              : ${MODEL_NAME}
  served as          : ${SERVED_MODEL_NAME}
  endpoint           : http://${HOST}:${PORT}/v1
  GPUs visible       : ${GPU_COUNT}
  tensor parallel    : ${TENSOR_PARALLEL_SIZE}
  pipeline parallel  : ${PIPELINE_PARALLEL_SIZE}
  max model length   : ${MAX_MODEL_LEN}
  gpu memory util    : ${GPU_MEMORY_UTILIZATION}
  max sequences      : ${MAX_NUM_SEQS}
  dtype              : ${DTYPE}
  quantization       : ${QUANTIZATION:-none}
  tool calling       : ${ENABLE_TOOL_CALLING} (parser: ${TOOL_CALL_PARSER:-none})
  api key            : $([[ -n "${VLLM_API_KEY}" ]] && echo "set" || echo "NONE - do not expose this port")
======================================================================

Point the backend at this server with:

  LLM_PROVIDER=vllm
  VLLM_BASE_URL=http://localhost:${PORT}/v1
  MODEL_NAME=${SERVED_MODEL_NAME}$([[ -n "${VLLM_API_KEY}" ]] && echo "
  VLLM_API_KEY=<the same key>")

The first launch downloads the weights and can take a long time. The server
is ready when it logs that it is listening, and when this returns JSON:

  curl http://localhost:${PORT}/v1/models

Command:
  vllm ${ARGS[*]}

BANNER

if [[ -z "${VLLM_API_KEY}" ]]; then
  warn "no API key set. Bind to localhost or keep this port behind a firewall -
         vLLM's --api-key does not protect every endpoint, so network isolation
         is the real control. See docs/deployment.md."
fi

exec vllm "${ARGS[@]}"
