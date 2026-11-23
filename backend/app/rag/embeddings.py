"""Embedding providers.

The ``mock`` provider is not a placeholder returning random numbers. It is a
hashed bag-of-words model - the classic "hashing trick" - which produces real
lexical similarity. That distinction matters: with random vectors the retrieval
pipeline could not be developed or tested at all, whereas with hashing you get
a working (if unsophisticated) search that behaves sensibly on real documents,
with no model download and no torch.

Vectors are always L2-normalized, so cosine similarity reduces to a dot
product and every store can assume unit vectors.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.core.config import settings
from app.core.exceptions import ConfigurationError, EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)

Vector = list[float]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Words that carry no retrieval signal. Dropping them keeps the hashed
#: representation from being dominated by grammar.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how i in is it its of on or
    that the their there these they this to was were what when where which who
    will with would you your can could do does did not no if then than but
    about into over under also more most such only own same so too very s t""".split()
)


def l2_normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Assumes unit vectors but does not require them."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return tokens


class EmbeddingProvider(ABC):
    """Turns text into vectors."""

    name: str = "base"

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed indexed passages."""

    async def embed_query(self, text: str) -> Vector:
        """Embed a search query.

        Separate from ``embed_documents`` because asymmetric models (and the
        hashing provider's query expansion) treat the two differently.
        """
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def health(self) -> tuple[str, str | None]:
        """Return ``(status, detail)`` without raising."""
        try:
            vector = await self.embed_query("health check")
            if len(vector) != self.dimensions:
                return "degraded", (
                    f"expected {self.dimensions} dimensions, got {len(vector)}"
                )
            return "healthy", None
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return "unavailable", f"{type(exc).__name__}: {exc}"

    async def close(self) -> None:
        return None


class HashingEmbeddingProvider(EmbeddingProvider):
    """Dependency-free lexical embeddings via the hashing trick.

    Each token is hashed to a bucket and a sign, then accumulated with
    sublinear term-frequency weighting. Character trigrams are hashed alongside
    whole words so that morphological variants ("policy"/"policies") and typos
    still land near each other.
    """

    name = "mock"

    def __init__(self, dimensions: int | None = None) -> None:
        self._dimensions = dimensions or settings.embedding_dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @staticmethod
    def _bucket(token: str, dimensions: int) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimensions
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def _embed_one(self, text: str) -> Vector:
        vector = [0.0] * self._dimensions
        tokens = tokenize(text)
        if not tokens:
            return vector

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            weight = 1.0 + math.log(count)
            index, sign = self._bucket(token, self._dimensions)
            vector[index] += sign * weight

            # Character trigrams, down-weighted: they add fuzzy matching
            # without swamping the whole-word signal.
            if len(token) > 3:
                padded = f"#{token}#"
                for i in range(len(padded) - 2):
                    gram_index, gram_sign = self._bucket(
                        padded[i : i + 3], self._dimensions
                    )
                    vector[gram_index] += gram_sign * weight * 0.25

        return l2_normalize(vector)

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers, loaded on demand.

    The model is imported and loaded lazily inside a thread so that importing
    this module never pulls in torch, and so that a multi-second model load
    does not block the event loop at startup.
    """

    name = "local"

    def __init__(
        self, model_name: str | None = None, batch_size: int | None = None
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None
        self._dimensions: int | None = None
        self._load_lock = asyncio.Lock()

    @property
    def dimensions(self) -> int:
        # Before the model loads we can only report what config declares; the
        # real value replaces it after the first embed call.
        return self._dimensions or settings.embedding_dimensions

    async def _ensure_model(self):
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model

            def _load():
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise EmbeddingError(
                        "EMBEDDING_PROVIDER=local requires sentence-transformers. "
                        "Install it with `pip install -r requirements-local-embeddings.txt`.",
                        user_message="Local embeddings are not installed on this server.",
                    ) from exc
                return SentenceTransformer(self.model_name)

            logger.info("loading embedding model %s", self.model_name)
            self._model = await asyncio.to_thread(_load)
            self._dimensions = int(self._model.get_sentence_embedding_dimension())
            logger.info(
                "embedding model ready",
                extra={"model": self.model_name, "dimensions": self._dimensions},
            )
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        model = await self._ensure_model()

        def _encode() -> list[Vector]:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [[float(x) for x in row] for row in vectors]

        try:
            return await asyncio.to_thread(_encode)
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"local embedding failed: {exc}") from exc


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Hosted embedding API, spoken over plain HTTP."""

    name = "openai"

    #: Dimensions for the models we expect to see configured.
    _KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model_name = model_name or "text-embedding-3-small"
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        if not self.api_key:
            raise ConfigurationError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        self._client = None
        self._dimensions = self._KNOWN_DIMENSIONS.get(self.model_name)

    @property
    def dimensions(self) -> int:
        return self._dimensions or settings.embedding_dimensions

    async def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
            )
        return self._client

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        client = await self._get_client()
        out: list[Vector] = []
        batch_size = settings.embedding_batch_size

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                response = await client.post(
                    "/embeddings", json={"model": self.model_name, "input": batch}
                )
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"embedding request failed: {exc}") from exc

            data = sorted(response.json().get("data", []), key=lambda d: d.get("index", 0))
            for item in data:
                vector = [float(x) for x in item["embedding"]]
                self._dimensions = len(vector)
                out.append(l2_normalize(vector))
        return out

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def create_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    provider = (name or settings.embedding_provider).lower()
    if provider == "mock":
        return HashingEmbeddingProvider()
    if provider == "local":
        return LocalEmbeddingProvider()
    if provider == "openai":
        return OpenAIEmbeddingProvider()
    raise ConfigurationError(
        f"unknown EMBEDDING_PROVIDER '{provider}'",
        details={"supported": ["mock", "local", "openai"]},
    )


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = create_embedding_provider()
        logger.info(
            "embedding provider ready",
            extra={"provider": _provider.name, "dimensions": _provider.dimensions},
        )
    return _provider


def set_embedding_provider(provider: EmbeddingProvider | None) -> None:
    global _provider
    _provider = provider


async def close_embedding_provider() -> None:
    global _provider
    if _provider is not None:
        await _provider.close()
        _provider = None
