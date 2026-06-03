# Indexer

The indexer reads all notes from your Obsidian vault, chunks them into manageable segments, extracts entities and summaries, embeds each chunk as a semantic vector, and stores the results in ChromaDB for search.

## Running

### One-shot index

```bash
python -m obsidian_ai.indexer
```

Or via CLI:

```bash
python cli.py index
```

### Skip LLM-dependent steps

```bash
python -m obsidian_ai.indexer --skip-entities   # Skip entity extraction
python -m obsidian_ai.indexer --skip-summaries   # Skip summary generation
python -m obsidian_ai.indexer --skip-entities --skip-summaries  # Skip both
```

These flags make indexing faster when you only need semantic search.

### Watch mode

```bash
python -m obsidian_ai.indexer --watch
```

Or via CLI:

```bash
python cli.py watch
```

The watcher monitors the vault directory using `watchdog` and automatically:
- **Creates/Updates:** Re-embeds modified notes and upserts to ChromaDB
- **Deletes:** Removes deleted notes from ChromaDB
- **Renames:** Deletes old entry, re-indexes under new path

Uses 2-second debounce to avoid re-indexing rapid successive saves.

## Pipeline

```
Obsidian vault
    │
    ├─ list_all_notes() ──────── get all .md paths (exclude patterns filtered)
    │
    ├─ get_note(path) ────────── fetch note content
    │
    ├─ _sanitize(content) ────── replace broken Unicode characters
    │
    ├─ word_count < 20? ──────── skip short notes
    │
    ├─ delete_by_path(path) ──── clear old chunks from ChromaDB
    │
    ├─ _extract_entities() ───── extract entities via LLM (cached, unless --skip-entities)
    │
    ├─ _generate_summary() ───── generate summary via LLM (cached, unless --skip-summaries)
    │
    ├─ chunk_text_heading_aware() ── heading-aware chunking
    │
    ├─ embed(chunk) ──────────── Ollama embedding per chunk
    │
    └─ upsert(path, i, vec, metadata) ── store in ChromaDB with full metadata
```

## Entity Extraction

During indexing, each note's sanitized content is sent to the LLM (Qwen3:8b) for entity extraction:

1. LLM receives a prompt requesting JSON entity output
2. Returns entities like `[{"name": "Python", "type": "Technology", "confidence": 0.95}]`
3. Entities are validated against known entity types (Person, Project, Hardware, Technology, Location, Concept, Event)
4. Stored in `entity_store` (in-memory index + `data/entities.json`)
5. Written as `entities_str` metadata on every chunk (e.g. `,Technology:Python,Project:Obsidian,`)

**Caching:** Results are cached per content hash in memory (`_entity_cache`) and on disk (`data/entity_cache.json`). Re-indexing identical content skips LLM calls. The cache persists across indexer runs using content hashes.

**Retries:** LLM calls retry 3× with exponential backoff (1s, 2s, 4s) on timeout before giving up. On failure, the note is still indexed (without entities).

## Summary Generation

During indexing, each note's content is sent to the LLM for a 2-3 sentence summary:

1. LLM receives a prompt requesting a concise summary
2. Summary is stored in chunk-0's ChromaDB metadata (to avoid duplication across chunks)
3. Passed through in `retrieve()` results for pipeline context
4. Used by `query()`, `summarize_topic()`, `tag_notes()` to provide compact note context

**Caching:** Identical to entity extraction — cached per content hash in memory (`_summary_cache`) and on disk (`data/summary_cache.json`).

**Retries:** Same 3× exponential backoff as entity extraction.

## Chunking

Large notes are split into overlapping chunks to stay within Ollama's embedding limit.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CHUNK_SIZE` | 500 words | Max words per chunk (~650 tokens) |
| `CHUNK_OVERLAP` | 100 words | Preserves context at chunk boundaries |

### Heading-Aware Chunking

The indexer uses heading-aware chunking by default:

1. Notes are split by markdown headings (`#`, `##`, etc.) via `_split_by_headings()`
2. Small sections are kept whole (under chunk size)
3. Large sections are split into word-based chunks with overlap
4. Each chunk retains its heading context as metadata

**Why heading-aware?** Without it, a section split across two chunks would lose its heading context, making it harder to find via semantic search.

**Example:** A 1200-word note with 3 sections:
```
Section 1: "## Introduction" (200 words)
Section 2: "## Methods" (600 words)     → split into 2 chunks
Section 3: "## Conclusion" (400 words)  → kept whole
```

### Flat Chunking

The original word-based chunking is also available as `chunk_text()`:

```
Chunk 0: words 0–499
Chunk 1: words 400–899   (100-word overlap)
Chunk 2: words 800–1199
```

## Content Sanitization

Before embedding, content is sanitized:

```python
def _sanitize(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")
```

This replaces invalid UTF-8 bytes with `?`, preventing Ollama errors from corrupted characters.

## Excluded Content

The indexer skips files/folders matching these patterns:

| Pattern | What it is |
|---------|------------|
| `_gsdata_` | GoodSync backup folder |
| `.gsbak` | GoodSync backup files |
| `.git` | Git repository |
| `__pycache__` | Python cache |
| `node_modules` | Node.js dependencies |
| `.excalidraw.md` | Excalidraw diagrams (JSON, not text) |

To exclude additional patterns, add them to `EXCLUDE_PATTERNS` in `config.py`.

## Short Note Filtering

Notes under 20 words (`SKIP_MIN_TOKENS`) are skipped — they don't contain enough signal for meaningful embeddings.

## Re-Indexing

Running the indexer again is safe. For each note, it:
1. Computes a content hash and compares to the stored hash
2. If unchanged (and not forced), skips the note (incremental)
3. If changed or new, deletes old chunks from ChromaDB
4. Re-chunks, re-extracts entities, re-generates summary, re-embeds

## Concurrency

- Indexing uses 2-6 parallel workers (auto-detected via `_embed_workers_for_current_machine()`)
- Entity extraction and summary generation are serialized to 1 concurrent call (via `_llm_chat_lock` semaphore) to prevent Ollama timeout pileup
- Embedding calls are not rate-limited (they're fast and cheap)

## YAML Frontmatter Utilities

### `add_tags_to_note(path, tags)`

Adds tags to a note's YAML frontmatter. Used by the `tag_notes` pipeline.

```python
from obsidian_ai.indexer import add_tags_to_note
add_tags_to_note("Notes/topic.md", ["python", "ml"])
```

**Behavior:**
- Parses existing YAML frontmatter (or creates new)
- Appends new tags (no duplicates)
- Converts string `tags` field to list if needed
- Writes updated content back to Obsidian

## Error Logging

Errors are logged to two places:

1. **Console output** — printed to stdout during the run
2. **`logs/indexer.log`** — timestamped log with rotating file handler

Example `logs/indexer.log` entry:
```
2026-05-31 14:23:01 [ERROR] obsidian_ai.indexer — FAILED: path/to/note.md — 150 words — 500 Server Error...
```

---

## CLI Wrapper

A command-line interface is available via `cli.py` in the project root:

```bash
# Run full index
python cli.py index

# Start file watcher
python cli.py watch

# Semantic search
python cli.py search "machine learning" -n 5

# Auto-tag notes
python cli.py tag-notes "python" -k 3

# Show index stats
python cli.py stats

# Re-run indexer
python cli.py sync
```
