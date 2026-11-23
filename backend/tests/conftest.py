"""Shared test fixtures.

The environment is pinned to the fully offline configuration *before* any
application module is imported, because ``app.core.config.settings`` is built
at import time. Every test therefore runs against the mock model, hashed
embeddings, the in-memory store and a disabled sandbox - no GPU, no network, no
API keys, and deterministic output.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# --- environment: must precede any `app` import ----------------------------
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="omnivores-tests-"))

os.environ.update(
    {
        # Point the settings loader at a file that does not exist, so a
        # developer's own backend/.env cannot change test behaviour.
        "OMNIVORES_ENV_FILE": str(_TEST_ROOT / "nonexistent.env"),
        "APP_ENV": "development",
        "LOG_LEVEL": "WARNING",
        "LLM_PROVIDER": "mock",
        "EMBEDDING_PROVIDER": "mock",
        "VECTOR_STORE": "memory",
        "RERANKER": "lexical",
        "WEB_SEARCH_PROVIDER": "mock",
        "PYTHON_SANDBOX_MODE": "disabled",
        "VECTOR_STORE_PATH": str(_TEST_ROOT / "vectorstore"),
        "UPLOAD_DIR": str(_TEST_ROOT / "uploads"),
        "MAX_CONTEXT_TOKENS": "8192",
        "MAX_OUTPUT_TOKENS": "1024",
        "API_KEY": "",
    }
)

import pytest  # noqa: E402

from app.agents.registry import AgentRegistry, get_agent_registry, set_agent_registry  # noqa: E402
from app.inference.client import set_inference_client  # noqa: E402
from app.inference.mock import MockInferenceClient  # noqa: E402
from app.rag.embeddings import (  # noqa: E402
    HashingEmbeddingProvider,
    set_embedding_provider,
)
from app.rag.retriever import Retriever  # noqa: E402
from app.rag.store import MemoryVectorStore, set_vector_store  # noqa: E402
from app.services.agent_service import AgentService, set_agent_service  # noqa: E402
from app.services.document_service import DocumentService, set_document_service  # noqa: E402
from app.services.message_service import MessageService, set_message_service  # noqa: E402
from app.tools.registry import build_default_registry, set_tool_registry  # noqa: E402


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_client() -> MockInferenceClient:
    """A mock model with streaming delays removed so tests run fast."""
    return MockInferenceClient(stream_delay=0.0)


@pytest.fixture
def embeddings() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(dimensions=128)


@pytest.fixture
def vector_store(tmp_path: Path) -> MemoryVectorStore:
    return MemoryVectorStore(path=tmp_path / "store", persist=False)


@pytest.fixture
def retriever(embeddings, vector_store) -> Retriever:
    return Retriever(embedding_provider=embeddings, vector_store=vector_store)


@pytest.fixture(autouse=True)
def isolated_globals(tmp_path, mock_client, embeddings, vector_store):
    """Give every test its own services, then restore the globals.

    Autouse because leaking a populated vector store or a channel full of
    messages between tests produces failures that look like real bugs.
    """
    set_inference_client(mock_client)
    set_embedding_provider(embeddings)
    set_vector_store(vector_store)
    set_tool_registry(build_default_registry())
    set_agent_registry(AgentRegistry())
    set_message_service(MessageService())
    set_document_service(
        DocumentService(store=vector_store, upload_dir=tmp_path / "uploads")
    )
    set_agent_service(AgentService())

    yield

    set_inference_client(None)
    set_embedding_provider(None)
    set_vector_store(None)
    set_tool_registry(None)
    set_agent_registry(None)
    set_message_service(None)
    set_document_service(None)
    set_agent_service(None)


@pytest.fixture
def agent_registry():
    return get_agent_registry()


# --- document fixtures -----------------------------------------------------


@pytest.fixture
def policy_text() -> str:
    return (
        "# Leave Policy\n\n"
        "All full-time employees are entitled to 20 days of paid annual leave "
        "per calendar year. Leave requests must be submitted at least two weeks "
        "in advance through the HR portal.\n\n"
        "# Notice Period\n\n"
        "The standard notice period for resignation is 30 days for employees "
        "below manager level and 60 days for managers and above.\n\n"
        "# Remote Work\n\n"
        "Employees may work remotely up to three days per week with written "
        "manager approval. Fully remote arrangements require director sign-off.\n"
    )


@pytest.fixture
def policy_file(tmp_path: Path, policy_text: str) -> Path:
    path = tmp_path / "employee_handbook.md"
    path.write_text(policy_text, encoding="utf-8")
    return path
