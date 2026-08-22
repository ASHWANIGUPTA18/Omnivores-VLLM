"""Chunking.

Two properties matter more than cleverness here:

1. **Chunks never span pages.** A chunk that straddles page 13 and 14 cannot be
   cited honestly, so page boundaries are hard splits.
2. **Splits prefer semantic boundaries.** We try paragraph breaks first, then
   sentence ends, then whitespace, and only split mid-word as a last resort.

Overlap is carried between adjacent chunks so that a fact sitting on a chunk
boundary is still retrievable from one side or the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings
from app.core.models import DocumentChunk
from app.core.tokens import estimate_tokens
from app.rag.parser import ParsedDocument

#: Split candidates, strongest boundary first.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkerConfig:
    chunk_size_tokens: int = 512
    overlap_tokens: int = 64
    #: Chunks smaller than this are merged into their neighbour rather than
    #: indexed alone - a 5-token chunk is noise in a similarity search.
    min_chunk_tokens: int = 24

    @classmethod
    def from_settings(cls) -> "ChunkerConfig":
        return cls(
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )


class Chunker:
    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig.from_settings()
        if self.config.overlap_tokens >= self.config.chunk_size_tokens:
            raise ValueError("overlap must be smaller than chunk size")

    # --- public ------------------------------------------------------------

    def chunk_document(
        self,
        parsed: ParsedDocument,
        *,
        document_id: str,
        filename: str,
        channel_id: str | None = None,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        index = 0
        for page in parsed.pages:
            for text in self.split_text(page.text):
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        filename=filename,
                        text=text,
                        chunk_index=index,
                        page=page.page,
                        channel_id=channel_id,
                        token_estimate=estimate_tokens(text),
                    )
                )
                index += 1
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split one page into overlapping, size-bounded pieces."""
        text = text.strip()
        if not text:
            return []
        if estimate_tokens(text) <= self.config.chunk_size_tokens:
            return [text]

        units = self._atomic_units(text)
        chunks = self._pack(units)
        return self._merge_runts(chunks)

    # --- internals ---------------------------------------------------------

    def _atomic_units(self, text: str) -> list[str]:
        """Break text into the smallest pieces we are willing to keep whole."""
        units: list[str] = []
        for block in self._recursive_split(text, 0):
            if block.strip():
                units.append(block)
        return units

    def _recursive_split(self, text: str, separator_index: int) -> list[str]:
        """Split until every piece fits, walking down the separator list."""
        if estimate_tokens(text) <= self.config.chunk_size_tokens:
            return [text]

        if separator_index >= len(_SEPARATORS):
            return self._hard_split(text)

        separator = _SEPARATORS[separator_index]
        parts = text.split(separator)
        if len(parts) == 1:
            return self._recursive_split(text, separator_index + 1)

        pieces: list[str] = []
        for position, part in enumerate(parts):
            # Put the separator back so sentences keep their punctuation.
            restored = part + separator if position < len(parts) - 1 else part
            if not restored.strip():
                continue
            if estimate_tokens(restored) > self.config.chunk_size_tokens:
                pieces.extend(self._recursive_split(restored, separator_index + 1))
            else:
                pieces.append(restored)
        return pieces

    def _hard_split(self, text: str) -> list[str]:
        """Last resort for text with no usable boundary - a long base64 blob,
        a minified script, a language without spaces."""
        approx_chars = max(1, self.config.chunk_size_tokens * 4)
        return [text[i : i + approx_chars] for i in range(0, len(text), approx_chars)]

    def _pack(self, units: list[str]) -> list[str]:
        """Greedily fill chunks, then seed the next one with overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = estimate_tokens(unit)

            if current and current_tokens + unit_tokens > self.config.chunk_size_tokens:
                chunks.append("".join(current).strip())
                current, current_tokens = self._overlap_tail(current)

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            tail = "".join(current).strip()
            if tail:
                chunks.append(tail)
        return chunks

    def _overlap_tail(self, units: list[str]) -> tuple[list[str], int]:
        """Take whole trailing units worth up to ``overlap_tokens``.

        Whole units, not a character slice, so the overlap never starts
        mid-sentence.
        """
        if self.config.overlap_tokens <= 0:
            return [], 0

        tail: list[str] = []
        tokens = 0
        for unit in reversed(units):
            unit_tokens = estimate_tokens(unit)
            if tokens + unit_tokens > self.config.overlap_tokens and tail:
                break
            tail.insert(0, unit)
            tokens += unit_tokens
            if tokens >= self.config.overlap_tokens:
                break
        return tail, tokens

    def _merge_runts(self, chunks: list[str]) -> list[str]:
        """Fold undersized trailing chunks back into the previous one."""
        if len(chunks) < 2:
            return chunks

        merged: list[str] = []
        for chunk in chunks:
            if (
                merged
                and estimate_tokens(chunk) < self.config.min_chunk_tokens
                and estimate_tokens(merged[-1]) + estimate_tokens(chunk)
                <= self.config.chunk_size_tokens * 1.2
            ):
                merged[-1] = (merged[-1].rstrip() + " " + chunk.lstrip()).strip()
            else:
                merged.append(chunk)
        return merged


def split_sentences(text: str) -> list[str]:
    """Sentence split used by the lexical reranker for snippet selection."""
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]


default_chunker = Chunker()
