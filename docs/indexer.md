# Indexer

The indexer reads all notes from your Obsidian vault, chunks them into manageable segments, embeds each chunk as a semantic vector, and stores the results in ChromaDB for search.

## Running

### One-shot index

```bash
python -m obsidian_ai.indexer
```

Or via CLI:

```bash
python cli.py index
```

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
    ├─ list_notes() ──────────── get all .md paths (exclude patterns filtered)
    │
    ├─ get_note(path) ────────── fetch note content
    │
    ├─ _sanitize(content) ────── replace broken Unicode characters
    │
    ├─ word_count < 20? ──────── skip short notes
    │
    ├─ delete_by_path(path) ──── clear old chunks from ChromaDB
    │
    ├─ chunk_text(content) ───── split into 500-word chunks (100-word overlap)
    │
    ├─ embed(chunk) ──────────── Ollama embedding per chunk
    │
    └─ upsert(path, i, vec) ──── store in ChromaDB
```

## Chunking

Large notes are split into overlapping chunks to stay within Ollama's 8192 token embedding limit.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CHUNK_SIZE` | 500 words | Max words per chunk (~650 tokens) |
| `CHUNK_OVERLAP` | 100 words | Preserves context at chunk boundaries |

**Why overlap?** Without overlap, a sentence split across two chunks would lose its meaning. The 100-word overlap ensures each chunk has surrounding context.

**Example:** A 1200-word note becomes 3 chunks:
```
Chunk 0: words 0–499
Chunk 1: words 400–899
Chunk 2: words 800–1199
```

Chunks are logged with their count when indexing:
```
  indexed: Courses/notes/Long Note.md (3 chunks)
```

## Content Sanitization

Before embedding, content is sanitized to handle broken Unicode:

```python
def _sanitize(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")
```

This replaces invalid UTF-8 bytes with `?`, preventing Ollama 500 errors from corrupted characters (e.g., `Ferrero roch\ufffd`).

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
1. Deletes all existing chunks from ChromaDB (`delete_by_path`)
2. Re-chunks and re-embeds the current content

This means updated notes get fresh embeddings without duplicates.

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

**Example:**

Before:
```markdown
# My Note

Some content here.
```

After `add_tags("Notes/topic.md", ["python", "ml"])`:
```markdown
---
tags:
- python
- ml
---

# My Note

Some content here.
```

## Error Logging

Errors are logged to two places:

1. **Console output** — printed to stdout during the run
2. **`logs/indexer.log`** — timestamped log with rotating file handler

Example `logs/indexer.log` entry:
```
2026-05-31 14:23:01 [ERROR] obsidian_ai.indexer — FAILED: path/to/note.md — 150 words — 500 Server Error...
```

Log level: `INFO` for console, `INFO` for file. Format: `YYYY-MM-DD HH:MM:SS [LEVEL] logger — message`.

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
python cli.py tag "python" -n 3

# Show index stats
python cli.py stats
```

| Command | Description |
|---------|-------------|
| `index` | Run full vault index |
| `watch` | Start file watcher daemon |
| `search <query>` | Semantic search with `-n` for result count |
| `tag <query>` | Auto-tag notes with `-n` for note count |
| `stats` | Show total notes in index |
