# vLLM inference server for Omnivores.
#
# Built on the official vLLM image, which already contains a CUDA-matched
# torch. Do not "helpfully" pip install torch on top of it - that is the most
# common way to break a working vLLM container.
#
# Build:
#   docker build -t omnivores-vllm:latest inference/
#
# Run (requires the NVIDIA Container Toolkit):
#   docker run --gpus all -p 8001:8001 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -e MODEL_NAME=Qwen/Qwen3-8B \
#     -e TOOL_CALL_PARSER=hermes \
#     omnivores-vllm:latest

ARG VLLM_VERSION=latest
FROM vllm/vllm-openai:${VLLM_VERSION}

# The entrypoint of the base image goes straight to the server; we want our
# launcher, which validates configuration and prints what it is doing.
ENTRYPOINT []

WORKDIR /opt/omnivores

COPY start-vllm.sh /opt/omnivores/start-vllm.sh
COPY config/ /opt/omnivores/config/
RUN chmod +x /opt/omnivores/start-vllm.sh

ENV VLLM_HOST=0.0.0.0 \
    VLLM_PORT=8001 \
    TENSOR_PARALLEL_SIZE=auto \
    MAX_MODEL_LEN=32768 \
    GPU_MEMORY_UTILIZATION=0.90 \
    ENABLE_TOOL_CALLING=true \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8001

# Long start period: the first run downloads weights, which can take many
# minutes. Marking the container unhealthy during that would be wrong.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
  CMD curl -fsS http://localhost:${VLLM_PORT}/v1/models || exit 1

CMD ["/opt/omnivores/start-vllm.sh"]
