"""Environment-driven configuration.

Every runtime choice in Omnivores - which model serves inference, which
embedding provider indexes documents, whether Python execution is sandboxed -
is expressed here and nowhere else. Application code reads ``settings``; it
never reads ``os.environ`` directly and never branches on a provider name
outside the six factory functions in ``inference/client.py``,
``rag/embeddings.py``, ``rag/store.py``, ``rag/reranker.py``,
``tools/web_search.py`` and ``sandbox/__init__.py``.

That is what makes the repository portable: the same source tree runs against a
deterministic mock model on a laptop and against vLLM on a GPU workstation.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["mock", "vllm", "openai"]
EmbeddingProviderName = Literal["mock", "local", "openai"]
VectorStoreName = Literal["memory", "chroma"]
RerankerName = Literal["none", "lexical", "cross_encoder"]
SandboxMode = Literal["disabled", "docker", "subprocess"]
WebSearchProviderName = Literal["mock", "tavily"]
#: How several agents mentioned in one message are run. Sequential is the
#: historical behaviour and stays the default; parallel exists so the two
#: can be compared under identical conditions.
AgentDispatchMode = Literal["sequential", "parallel"]

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("OMNIVORES_ENV_FILE", str(BACKEND_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application -------------------------------------------------------
    app_env: Literal["development", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- Inference ---------------------------------------------------------
    llm_provider: LLMProvider = "mock"

    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_api_key: str = ""
    model_name: str = ""

    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    inference_timeout_seconds: float = 120.0
    inference_connect_timeout_seconds: float = 10.0
    inference_max_retries: int = 2

    # --- Generation --------------------------------------------------------
    temperature: float = 0.3
    top_p: float = 0.95
    max_output_tokens: int = 2048

    # --- Agent runtime -----------------------------------------------------
    max_tool_rounds: int = 5
    max_tool_calls: int = 10
    agent_timeout_seconds: float = 120.0
    tool_timeout_seconds: float = 30.0
    max_context_tokens: int = 16384
    context_rag_token_budget: int = 4000
    context_recent_messages: int = 20
    agent_dispatch: AgentDispatchMode = "sequential"

    # --- RAG ---------------------------------------------------------------
    embedding_provider: EmbeddingProviderName = "mock"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 32

    vector_store: VectorStoreName = "memory"
    vector_store_path: Path = BACKEND_ROOT / "data" / "vectorstore"
    chroma_collection: str = "omnivores"

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    rag_top_k: int = 20
    rag_rerank_top_k: int = 5
    rag_min_score: float = 0.0

    reranker: RerankerName = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Documents ---------------------------------------------------------
    max_upload_size_mb: int = 50
    upload_dir: Path = BACKEND_ROOT / "data" / "uploads"
    allowed_extensions: str = ".pdf,.txt,.md,.docx"

    # --- Sandbox -----------------------------------------------------------
    python_sandbox_mode: SandboxMode = "disabled"
    python_sandbox_image: str = "omnivores-sandbox:latest"
    python_timeout_seconds: float = 15.0
    python_memory_limit_mb: int = 512
    python_cpu_limit: float = 1.0
    python_max_output_chars: int = 10_000

    # --- Web search --------------------------------------------------------
    web_search_provider: WebSearchProviderName = "mock"
    tavily_api_key: str = ""
    web_search_max_results: int = 5

    # --- Security ----------------------------------------------------------
    api_key: str = ""
    rate_limit_messages_per_minute: int = 60
    max_websocket_connections: int = 100
    max_message_length: int = 8000

    # --- Derived helpers ---------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extension_set(self) -> set[str]:
        out: set[str] = set()
        for raw in self.allowed_extensions.split(","):
            ext = raw.strip().lower()
            if not ext:
                continue
            out.add(ext if ext.startswith(".") else "." + ext)
        return out

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def resolved_model_name(self) -> str:
        """The model id reported to clients and sent on inference requests."""
        if self.llm_provider == "openai":
            return self.openai_model
        if self.llm_provider == "vllm":
            return self.model_name or "(auto-detected from /v1/models)"
        return "mock-omnivores-1"

    # --- Validation --------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("invalid LOG_LEVEL: " + v)
        return level

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")
        return v

    @field_validator("max_tool_rounds", "max_tool_calls", "max_output_tokens")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @model_validator(mode="after")
    def _check_provider_credentials(self) -> Settings:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        if self.web_search_provider == "tavily" and not self.tavily_api_key:
            raise ValueError("WEB_SEARCH_PROVIDER=tavily requires TAVILY_API_KEY")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        if self.context_rag_token_budget >= self.max_context_tokens:
            raise ValueError("CONTEXT_RAG_TOKEN_BUDGET must be < MAX_CONTEXT_TOKENS")
        if self.rag_rerank_top_k > self.rag_top_k:
            raise ValueError("RAG_RERANK_TOP_K must be <= RAG_TOP_K")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be < CHUNK_SIZE_TOKENS")
        return self

    def production_warnings(self) -> list[str]:
        """Configuration that is fine in dev but dangerous in production.

        Surfaced at startup rather than raised, so a misconfigured production
        deploy is loud without being unbootable.
        """
        warnings: list[str] = []
        if not self.is_production:
            return warnings
        if self.llm_provider == "mock":
            warnings.append("LLM_PROVIDER=mock in production - responses are fake")
        if self.python_sandbox_mode == "subprocess":
            warnings.append(
                "PYTHON_SANDBOX_MODE=subprocess executes model-generated code on "
                "the host. Use 'docker' in production."
            )
        if not self.api_key:
            warnings.append("API_KEY is empty - the API is unauthenticated")
        if "*" in self.cors_origin_list:
            warnings.append("CORS_ORIGINS contains '*'")
        return warnings

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
