"""Document search tool.

The bridge between the RAG pipeline and the agent loop. It returns the numbered
excerpt block the retriever builds, so the model sees ``[1]``, ``[2]`` markers
it can cite, and it attaches structured citations to the result for the UI.

Scoping precedence is decided here, not by the model: if the user scoped the
question to specific documents, the model cannot widen it by passing a
different filter. The model chooses the *query*; the application chooses the
*scope*.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.rag.retriever import Retriever, default_retriever
from app.tools.base import Tool, ToolContext, ToolResult

logger = get_logger(__name__)


class KnowledgeBaseTool(Tool):
    name = "search_knowledge_base"
    description = (
        "Search the documents the user has uploaded. Use this whenever the "
        "question could be answered by a document - policies, handbooks, "
        "reports, contracts, specifications - or when the user refers to "
        "'the document', 'the file', or 'what was uploaded'. Returns numbered "
        "excerpts; cite them in your answer as [1], [2]. If the excerpts do not "
        "contain the answer, say so instead of guessing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to look for, in natural language. Prefer the user's own "
                    "wording and key terms over a rephrasing - exact terms match "
                    "better."
                ),
                "minLength": 2,
                "maxLength": 500,
            },
            "top_k": {
                "type": "integer",
                "description": "How many excerpts to return. Default 5.",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> Retriever:
        return self._retriever or default_retriever

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        query = arguments["query"]
        rerank_top_k = arguments.get("top_k") or settings.rag_rerank_top_k

        try:
            result = await self.retriever.retrieve(
                query,
                document_ids=context.document_ids or None,
                channel_id=context.channel_id,
                rerank_top_k=rerank_top_k,
            )
        except RetrievalError as exc:
            logger.error("knowledge base search failed: %s", exc)
            return ToolResult.failure(
                f"Document search failed: {exc.user_message}",
                query=query,
            )

        if result.is_empty:
            scope = (
                f"the {len(context.document_ids)} selected document(s)"
                if context.document_ids
                else "the uploaded documents"
            )
            return ToolResult(
                content=(
                    f"No relevant passages found in {scope} for {query!r}. "
                    "Do not invent an answer - tell the user nothing matched."
                ),
                ok=True,
                summary="No matching passages",
                metadata={"query": query, **result.metrics()},
            )

        sources = ", ".join(
            sorted({chunk.filename for chunk in result.chunks})
        )
        return ToolResult(
            content=result.context,
            citations=result.citations,
            summary=f"{len(result.chunks)} excerpt(s) from {sources}",
            metadata={"query": query, **result.metrics()},
        )

    async def health(self) -> tuple[str, str | None]:
        try:
            total = await self.retriever.store.count()
            return "healthy", f"{total} chunks indexed"
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return "unavailable", f"{type(exc).__name__}: {exc}"
