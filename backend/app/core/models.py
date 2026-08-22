"""Domain models shared across the API, services and agent runtime.

These are the shapes that cross the WebSocket boundary, so they are Pydantic
models rather than plain dataclasses: the frontend contract is defined here.

Inference-layer types (ChatMessage, ToolCall, ChatChunk) deliberately live in
``app/inference/base.py`` instead. Those describe how we talk to a model; these
describe what the product stores and shows.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


# --- Authors ---------------------------------------------------------------


class AuthorType(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class Author(BaseModel):
    """Agents are first-class chat participants, not a response mode."""

    id: str
    type: AuthorType
    display_name: str
    avatar: str | None = None

    @classmethod
    def system(cls) -> "Author":
        return cls(id="system", type=AuthorType.SYSTEM, display_name="Omnivores")


# --- Messages --------------------------------------------------------------


class MessageStatus(str, Enum):
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolInvocation(BaseModel):
    """A tool call as it is surfaced in the UI timeline."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["running", "completed", "failed"] = "running"
    result_preview: str | None = None
    error: str | None = None
    duration_ms: int | None = None


class Citation(BaseModel):
    """A pointer back into a source document, rendered under agent answers."""

    document_id: str
    filename: str
    chunk_id: str
    page: int | None = None
    score: float | None = None
    snippet: str = ""

    @property
    def label(self) -> str:
        if self.page is not None:
            return f"{self.filename} - page {self.page}"
        return self.filename


class Message(BaseModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    channel_id: str
    author: Author
    content: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    status: MessageStatus = MessageStatus.COMPLETED

    #: Agent ids mentioned with @ in this message.
    mentions: list[str] = Field(default_factory=list)
    #: Set on agent replies: the user message that triggered them.
    reply_to: str | None = None

    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    #: Document ids the user explicitly scoped the question to.
    document_ids: list[str] = Field(default_factory=list)

    error: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None


class MessageCreate(BaseModel):
    """Inbound payload from the composer."""

    channel_id: str
    content: str
    document_ids: list[str] = Field(default_factory=list)
    client_message_id: str | None = None


# --- Channels --------------------------------------------------------------


class Channel(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    #: Agent ids that live in this channel. Empty means all registered agents.
    agent_ids: list[str] = Field(default_factory=list)


# --- Documents -------------------------------------------------------------


class DocumentStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class Document(BaseModel):
    id: str = Field(default_factory=lambda: new_id("doc"))
    filename: str
    #: Lowercased extension including the dot, e.g. ".pdf".
    extension: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0

    channel_id: str | None = None
    uploaded_by: str = "user"

    status: DocumentStatus = DocumentStatus.UPLOADING
    #: 0.0 - 1.0, advanced as the ingestion pipeline moves through stages.
    progress: float = 0.0
    stage: str = "uploading"

    page_count: int | None = None
    chunk_count: int = 0
    character_count: int = 0

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    indexed_at: datetime | None = None

    error: str | None = None
    #: Absolute path on disk. Never sent to the client.
    storage_path: str | None = Field(default=None, exclude=True)

    def public(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"storage_path"})


class DocumentChunk(BaseModel):
    """One indexed unit of text plus the metadata that makes citation possible."""

    id: str = Field(default_factory=lambda: new_id("chunk"))
    document_id: str
    filename: str
    text: str
    chunk_index: int
    page: int | None = None
    channel_id: str | None = None
    token_estimate: int = 0
    created_at: datetime = Field(default_factory=utcnow)

    def metadata(self) -> dict[str, Any]:
        """Flat metadata for the vector store.

        Values are restricted to str/int/float/bool because Chroma rejects
        anything else, and the memory store mirrors that constraint so the two
        backends stay swappable.
        """
        meta: dict[str, Any] = {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "token_estimate": self.token_estimate,
            "created_at": self.created_at.isoformat(),
        }
        if self.page is not None:
            meta["page"] = self.page
        if self.channel_id is not None:
            meta["channel_id"] = self.channel_id
        return meta


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, with its scores attached."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    chunk_index: int
    page: int | None = None
    channel_id: str | None = None
    #: Vector similarity, 0..1 where higher is closer.
    score: float = 0.0
    #: Set by the reranker when one is active.
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score

    def to_citation(self) -> Citation:
        snippet = self.text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        return Citation(
            document_id=self.document_id,
            filename=self.filename,
            chunk_id=self.chunk_id,
            page=self.page,
            score=round(self.final_score, 4),
            snippet=snippet,
        )


# --- Health ----------------------------------------------------------------


class ComponentHealth(BaseModel):
    status: Literal["healthy", "degraded", "unavailable"] = "healthy"
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthReport(BaseModel):
    status: Literal["healthy", "degraded", "unavailable"] = "healthy"
    version: str = "1.0.0"
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=utcnow)
