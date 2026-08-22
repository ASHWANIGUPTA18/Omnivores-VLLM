# Document RAG

## The pipeline

```
 UPLOAD ──▶ VALIDATE ──▶ STORE ──▶ PARSE ──▶ CHUNK ──▶ EMBED ──▶ INDEX ──▶ READY
                │                     │                                      │
                └── reject early      └── FAILED, visibly, with a reason      │
                                                                             ▼
 QUERY ──▶ SCOPE ──▶ EMBED ──▶ SEARCH (top 20) ──▶ RERANK (top 5) ──▶ CONTEXT + CITATIONS
```

Each stage advances `status`, `stage` and `progress` on the document and emits
a WebSocket event, so the UI shows a progress bar that reflects reality rather
than a spinner that lies.

---

## Document lifecycle

```
UPLOADING  ──▶  PROCESSING  ──▶  INDEXING  ──▶  READY
                     │               │
                     └───────────────┴────────▶  FAILED
```

A failed document **stays in the list** with the reason attached. Silently
dropping it, or leaving it stuck at 40%, are both worse: the user needs to know
their file will not be searched, and why.

---

## Parsing

| Format | Parser | Citation unit |
|---|---|---|
| `.pdf` | pypdf | page number |
| `.docx` | python-docx | synthetic section |
| `.md` | heading splitter | section number |
| `.txt` | plain | whole file |

Two decisions worth stating:

**Word documents have no pages.** Pagination is a rendering concern the file
does not record, so we group paragraphs into sections and cite by section
rather than inventing page numbers that would not match what the user sees.

**Markdown is split on headings before chunking.** Sections are better
retrieval units than an arbitrary character window.

A scanned PDF with no text layer is rejected with a clear message. There is no
OCR, and pretending otherwise by indexing nothing would be worse than failing.

### Adding a format

```python
class CSVParser(DocumentParser):
    extensions = (".csv",)
    def parse(self, path: Path) -> ParsedDocument: ...

parser_registry.register(CSVParser())
```

Then add `.csv` to `ALLOWED_EXTENSIONS`.

---

## Chunking

Two invariants:

**Chunks never span pages.** A chunk straddling pages 13 and 14 cannot be cited
honestly, so page boundaries are hard splits.

**Splits prefer semantic boundaries.** Paragraph breaks, then sentence ends,
then punctuation, then whitespace, and only mid-word as a last resort for text
with no boundary at all - a minified script, a base64 blob.

```
CHUNK_SIZE_TOKENS=512     target size
CHUNK_OVERLAP_TOKENS=64   carried between neighbours
```

Overlap is taken as **whole trailing units**, not a character slice, so it
never begins mid-sentence. Undersized trailing chunks are merged back into
their neighbour - a 5-token chunk is noise in a similarity search.

---

## Embeddings

| Provider | What it is | Needs |
|---|---|---|
| `mock` | Hashed bag-of-words | nothing |
| `local` | sentence-transformers | torch |
| `openai` | Hosted API | API key |

**The mock provider is not random vectors.** It is the hashing trick: tokens
are hashed into buckets with signs and accumulated with sublinear term
frequency, plus down-weighted character trigrams for fuzzy matching. The result
is genuine lexical similarity.

That distinction is what makes the RAG pipeline developable and testable
offline. With random vectors the tests could only check plumbing; with hashing
they assert that "annual leave entitlement" retrieves the leave policy and not
the build pipeline - and they do.

All vectors are L2-normalized, so every store can assume unit vectors.

> **Changing `EMBEDDING_PROVIDER` requires re-indexing.** Vectors from different
> models are not comparable. The memory store logs and skips mismatched
> dimensions rather than returning nonsense; use
> `POST /api/documents/{id}/reindex`.

---

## Vector stores

| Store | Index | Good for |
|---|---|---|
| `memory` | brute-force scan | up to ~50k chunks, zero dependencies |
| `chroma` | HNSW | larger corpora, the GPU machine |

The memory store persists to JSON with a write-then-rename, so an interrupted
save cannot truncate the index. A corrupt file raises rather than starting
empty - an empty index that looks healthy produces confidently wrong "nothing
found" answers.

Both stores return scores on the **same 0..1 scale**, so `RAG_MIN_SCORE` means
the same thing whichever backend is configured.

---

## Scoping

The single most important thing for answer quality. Retrieving from every
document ever uploaded, for every question, produces confident nonsense.

```
1. explicit document_ids   ← "Ask this document"
2. otherwise: the channel
3. otherwise: everything
```

**The application decides scope; the model decides the query.** A user who
scoped a question to one document cannot have that silently widened.

Every chunk carries `document_id`, `filename`, `chunk_index`, `page`,
`channel_id`, `token_estimate` and `created_at`. Metadata values are restricted
to scalars because Chroma rejects anything else, and the memory store mirrors
that so the two stay swappable.

---

## Reranking

Vector search is recall-oriented: good at pulling 20 plausible passages out of
thousands, mediocre at ordering those 20. Reranking fixes the ordering, which
is what actually reaches the context window.

| Reranker | Cost | Notes |
|---|---|---|
| `none` | zero | baseline for benchmarks |
| `lexical` | microseconds | BM25 over candidates, fused with the vector score |
| `cross_encoder` | ~50ms | strongest; needs sentence-transformers |

`lexical` is the default because it rescues the case vector search handles
worst: an exact term the user typed verbatim - `Section 4.2`, a product code -
that the embedding blurred away. BM25 statistics are computed over the
candidate set rather than the corpus: a deliberate approximation that needs no
global index and discriminates well within 20 candidates.

---

## Citations

The context block is numbered, and the citation list uses the same numbers:

```
Excerpts from the user's documents. Cite them by their bracketed number...

[1] employee_handbook.pdf, page 14
Full-time employees receive 20 days of paid annual leave...

[2] employee_handbook.pdf, page 15
Leave requests must be submitted two weeks in advance...
```

If the context is truncated to fit the budget, the citation list is truncated
to match - so `[2]` in an answer always points at a source the user can see.
The last chunk to partially fit is truncated rather than dropped, because a
partial best-match passage is more useful than none.

The UI renders `[1]` as a clickable chip that scrolls to the source and expands
its snippet.

---

## Tuning

| Setting | Default | Raise it when | Lower it when |
|---|---|---|---|
| `CHUNK_SIZE_TOKENS` | 512 | Answers lack context | Retrieval is imprecise |
| `CHUNK_OVERLAP_TOKENS` | 64 | Facts fall across boundaries | The index is bloated |
| `RAG_TOP_K` | 20 | Recall is poor | Reranking is slow |
| `RAG_RERANK_TOP_K` | 5 | Answers need more sources | The context is crowded |
| `RAG_MIN_SCORE` | 0.0 | Irrelevant passages appear | Valid answers are missed |
| `CONTEXT_RAG_TOKEN_BUDGET` | 4000 | Excerpts are cut short | History is being squeezed |

Measure changes rather than guessing:

```bash
python benchmarks/benchmark_rag.py --documents 50 --queries 100
```

---

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| "No text could be extracted" | Scanned PDF | OCR it first; unsupported here |
| Answers cite the wrong document | Scope too wide | Upload into a channel, or use "Ask" |
| Nothing found after changing providers | Stale vectors | Re-index |
| Exact terms are missed | Reranker off | `RERANKER=lexical` |
| Retrieval is slow at scale | Brute-force scan | `VECTOR_STORE=chroma` |
| Excerpts cut mid-sentence | Budget too small | Raise `CONTEXT_RAG_TOKEN_BUDGET` |
