"""Document parsing.

Parsers return text split by page, not one blob. Page numbers are the whole
reason citations can say "handbook.pdf, page 14" instead of "handbook.pdf",
so the page boundary is preserved from parse all the way to the rendered
citation.

Adding a format means writing a ``DocumentParser`` subclass and registering it.
Nothing else changes.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from app.core.exceptions import DocumentParseError, UnsupportedFileType
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    #: 1-based. ``None`` for formats without pagination (txt, md).
    page: int | None
    text: str


@dataclass
class ParsedDocument:
    pages: list[ParsedPage] = field(default_factory=list)
    page_count: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def character_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse the whitespace damage typical of PDF extraction.

    Deliberately keeps paragraph breaks: the chunker splits on them, so
    flattening everything to single spaces would produce worse chunks.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("﻿", "")
    # Words split across a line break by hyphenation.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DocumentParser(ABC):
    """Extracts text and page structure from one family of file types."""

    #: Lowercased extensions this parser handles, including the dot.
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Read ``path``. Raises ``DocumentParseError`` on unreadable input."""

    def supports(self, extension: str) -> bool:
        return extension.lower() in self.extensions


class TextParser(DocumentParser):
    extensions = (".txt",)

    def parse(self, path: Path) -> ParsedDocument:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DocumentParseError(f"cannot read {path.name}: {exc}") from exc
        text = normalize_whitespace(raw)
        return ParsedDocument(pages=[ParsedPage(page=None, text=text)], page_count=None)


class MarkdownParser(DocumentParser):
    """Markdown, split on top-level headings.

    Sections make better retrieval units than an arbitrary character window, so
    we hand the chunker pre-split sections rather than one blob. Section index
    is carried in the page field, which the UI renders as "section N".
    """

    extensions = (".md", ".markdown")

    _HEADING_RE = re.compile(r"^#{1,2} .+$", re.MULTILINE)

    def parse(self, path: Path) -> ParsedDocument:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DocumentParseError(f"cannot read {path.name}: {exc}") from exc

        text = normalize_whitespace(raw)
        boundaries = [m.start() for m in self._HEADING_RE.finditer(text)]

        if not boundaries:
            return ParsedDocument(
                pages=[ParsedPage(page=None, text=text)], page_count=None
            )

        if boundaries[0] > 0:
            boundaries.insert(0, 0)
        boundaries.append(len(text))

        pages: list[ParsedPage] = []
        for index in range(len(boundaries) - 1):
            section = text[boundaries[index] : boundaries[index + 1]].strip()
            if section:
                pages.append(ParsedPage(page=len(pages) + 1, text=section))
        return ParsedDocument(pages=pages, page_count=len(pages))


class PDFParser(DocumentParser):
    extensions = (".pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise DocumentParseError(
                "PDF support requires pypdf. Install it with "
                "`pip install -r requirements.txt`.",
                user_message="PDF support is not installed on this server.",
            ) from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise DocumentParseError(f"cannot open PDF {path.name}: {exc}") from exc

        if getattr(reader, "is_encrypted", False):
            # An empty user password is common and decrypts silently.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise DocumentParseError(
                    f"{path.name} is password protected",
                    user_message="That PDF is password protected.",
                ) from exc

        pages: list[ParsedPage] = []
        failed_pages = 0
        for index, page in enumerate(reader.pages, start=1):
            try:
                extracted = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - one bad page must not fail all
                logger.warning("page %d of %s failed to extract: %s", index, path.name, exc)
                failed_pages += 1
                continue
            text = normalize_whitespace(extracted)
            if text:
                pages.append(ParsedPage(page=index, text=text))

        total_pages = len(reader.pages)
        if not pages:
            raise DocumentParseError(
                f"no extractable text in {path.name} ({total_pages} pages)",
                user_message=(
                    "No text could be extracted from that PDF. It is most likely "
                    "a scan - OCR is not supported yet."
                ),
                details={"page_count": total_pages, "failed_pages": failed_pages},
            )

        metadata: dict[str, str] = {}
        try:
            info = reader.metadata or {}
            for key in ("/Title", "/Author", "/Subject"):
                value = info.get(key)
                if value:
                    metadata[key.lstrip("/").lower()] = str(value)
        except Exception:  # noqa: BLE001 - metadata is decorative
            pass

        return ParsedDocument(pages=pages, page_count=total_pages, metadata=metadata)


class DocxParser(DocumentParser):
    """Word documents.

    python-docx exposes no page boundaries - pagination is a rendering concern
    the file does not record - so we group paragraphs into synthetic sections
    and cite by section instead of inventing page numbers.
    """

    extensions = (".docx",)

    #: Roughly a page of prose.
    _CHARS_PER_SECTION = 3000

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import docx
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise DocumentParseError(
                "DOCX support requires python-docx.",
                user_message="Word document support is not installed on this server.",
            ) from exc

        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise DocumentParseError(f"cannot open {path.name}: {exc}") from exc

        blocks: list[str] = []
        for paragraph in document.paragraphs:
            text = (paragraph.text or "").strip()
            if text:
                blocks.append(text)

        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))

        if not blocks:
            raise DocumentParseError(
                f"no text found in {path.name}",
                user_message="That Word document appears to be empty.",
            )

        pages: list[ParsedPage] = []
        buffer: list[str] = []
        size = 0
        for block in blocks:
            buffer.append(block)
            size += len(block)
            if size >= self._CHARS_PER_SECTION:
                pages.append(
                    ParsedPage(page=len(pages) + 1, text=normalize_whitespace("\n".join(buffer)))
                )
                buffer, size = [], 0
        if buffer:
            pages.append(
                ParsedPage(page=len(pages) + 1, text=normalize_whitespace("\n".join(buffer)))
            )

        return ParsedDocument(pages=pages, page_count=len(pages))


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._parsers: list[DocumentParser] = parsers or [
            PDFParser(),
            DocxParser(),
            MarkdownParser(),
            TextParser(),
        ]

    def register(self, parser: DocumentParser) -> None:
        self._parsers.insert(0, parser)

    @property
    def supported_extensions(self) -> set[str]:
        return {ext for parser in self._parsers for ext in parser.extensions}

    def for_extension(self, extension: str) -> DocumentParser:
        extension = extension.lower()
        for parser in self._parsers:
            if parser.supports(extension):
                return parser
        raise UnsupportedFileType(
            f"no parser for '{extension}'",
            user_message=(
                f"'{extension}' files are not supported. Supported types: "
                + ", ".join(sorted(self.supported_extensions))
            ),
        )

    def parse(self, path: Path) -> ParsedDocument:
        return self.for_extension(path.suffix).parse(path)


parser_registry = ParserRegistry()
