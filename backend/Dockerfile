# Omnivores backend.
#
# Multi-stage so the runtime image carries no build toolchain. Runs as a
# non-root user: this process handles uploaded files and model-generated tool
# calls, so it should hold as little privilege as possible.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Optional extras are separate files so a default build never pulls in torch.
ARG INSTALL_CHROMA=false
ARG INSTALL_LOCAL_EMBEDDINGS=false
COPY requirements-chroma.txt requirements-local-embeddings.txt ./

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && if [ "$INSTALL_CHROMA" = "true" ]; then pip install --no-cache-dir -r requirements-chroma.txt; fi \
 && if [ "$INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then pip install --no-cache-dir -r requirements-local-embeddings.txt; fi


FROM python:3.11-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 omnivores

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=omnivores:omnivores app/ ./app/

# Writable state. Mount volumes here so uploads and the index survive a redeploy.
RUN mkdir -p /app/data/uploads /app/data/vectorstore /app/knowledge \
 && chown -R omnivores:omnivores /app/data /app/knowledge

USER omnivores

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# No --reload in an image; that is a development-only concern.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
