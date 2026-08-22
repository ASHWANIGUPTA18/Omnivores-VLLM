"""Web search tool and providers.

The provider is behind an interface for the same reason inference is: search
APIs are interchangeable and none of them should be baked into the agent layer.

``mock`` is deterministic and offline. It does not pretend to have searched the
web - it returns clearly-labelled synthetic results, so a developer reading a
transcript can never mistake them for real ones.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext, ToolResult

logger = get_logger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0


class WebSearchProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        ...

    async def health(self) -> tuple[str, str | None]:
        return "healthy", None

    async def close(self) -> None:
        return None


class MockSearchProvider(WebSearchProvider):
    """Deterministic offline results, labelled as synthetic."""

    name = "mock"

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        results: list[SearchResult] = []
        for index in range(min(max_results, 3)):
            token = digest[index * 8 : index * 8 + 8]
            results.append(
                SearchResult(
                    title=f"[SYNTHETIC RESULT {index + 1}] {query[:60]}",
                    url=f"https://example.invalid/{token}",
                    snippet=(
                        f"This is a placeholder search result for {query!r} produced "
                        "by the mock provider. It contains no real information. Set "
                        "WEB_SEARCH_PROVIDER=tavily and TAVILY_API_KEY to search the "
                        "actual web."
                    ),
                    score=1.0 - index * 0.1,
                )
            )
        return results


class TavilySearchProvider(WebSearchProvider):
    """Tavily search, spoken over plain HTTP."""

    name = "tavily"
    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else settings.tavily_api_key
        if not self.api_key:
            raise ConfigurationError(
                "WEB_SEARCH_PROVIDER=tavily requires TAVILY_API_KEY"
            )
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        client = await self._get_client()
        response = await client.post(
            self.ENDPOINT,
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
        )
        response.raise_for_status()
        payload = response.json()

        return [
            SearchResult(
                title=str(item.get("title") or item.get("url") or "untitled"),
                url=str(item.get("url") or ""),
                snippet=str(item.get("content") or "")[:1000],
                score=float(item.get("score") or 0.0),
            )
            for item in payload.get("results", [])
        ]

    async def health(self) -> tuple[str, str | None]:
        return ("healthy", "tavily") if self.api_key else ("unavailable", "no API key")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def create_search_provider(name: str | None = None) -> WebSearchProvider:
    provider = (name or settings.web_search_provider).lower()
    if provider == "mock":
        return MockSearchProvider()
    if provider == "tavily":
        return TavilySearchProvider()
    raise ConfigurationError(
        f"unknown WEB_SEARCH_PROVIDER '{provider}'",
        details={"supported": ["mock", "tavily"]},
    )


_provider: WebSearchProvider | None = None


def get_search_provider() -> WebSearchProvider:
    global _provider
    if _provider is None:
        _provider = create_search_provider()
    return _provider


def set_search_provider(provider: WebSearchProvider | None) -> None:
    global _provider
    _provider = provider


async def close_search_provider() -> None:
    global _provider
    if _provider is not None:
        await _provider.close()
        _provider = None


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the public web for current information. Use it for events, "
        "releases, prices, or anything that changed after your training data. "
        "Do not use it for questions about the user's own uploaded documents - "
        "use search_knowledge_base for those. Always attribute claims to the "
        "result you took them from."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Keep it short and keyword-led.",
                "minLength": 2,
                "maxLength": 400,
            },
            "max_results": {
                "type": "integer",
                "description": "How many results to return. Default 5.",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, provider: WebSearchProvider | None = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> WebSearchProvider:
        return self._provider or get_search_provider()

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        query = arguments["query"]
        max_results = arguments.get("max_results") or settings.web_search_max_results

        try:
            results = await self.provider.search(query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 - a failed search is not a failed turn
            logger.error("web search failed: %s", exc)
            return ToolResult.failure(
                f"Web search failed ({type(exc).__name__}). Answer from what you "
                "already know, and say the search did not run.",
                query=query,
            )

        if not results:
            return ToolResult(
                content=f"No web results for {query!r}.",
                summary="No results",
                metadata={"query": query, "provider": self.provider.name},
            )

        lines = [f"Web results for {query!r}:", ""]
        for index, item in enumerate(results, start=1):
            lines.append(f"[{index}] {item.title}")
            lines.append(f"    {item.url}")
            lines.append(f"    {item.snippet.strip()[:500]}")
            lines.append("")

        return ToolResult(
            content="\n".join(lines).strip(),
            summary=f"{len(results)} result(s) via {self.provider.name}",
            metadata={
                "query": query,
                "provider": self.provider.name,
                "urls": [r.url for r in results],
            },
        )

    async def health(self) -> tuple[str, str | None]:
        return await self.provider.health()
