# Architecture

## Overview

The system reads notes from Obsidian, embeds them as semantic vectors, and stores them in a local vector database for search. An MCP server exposes these capabilities to AI agents. All processing runs locally — no data leaves the machine.

```
Agent (Goose / Claude / Cursor)
        │
        ▼
   mcp_server.py ──── FastMCP (stdio)
   ├── search_notes(query)
   ├── read_note(path)
   ├── write_note(path, content)
   ├── list_notes()
   ├── add_tags(path, tags)
   ├── create_backlink(a, b)
   ├── sync_index()
   ├── ask_vault(question)      ← LLM-powered
   └── tag_notes(query)         ← LLM-powered
        │
        ├──────────────────┐
        ▼                  ▼
obsidian_client.py    chroma_store.py
        │                  │
        ▼                  ▼
  Obsidian REST API    ChromaDB
  (port 27123)         ./chroma_db/
        │
        ▼
   llm_client.py
   ├── embed(text) → Ollama → nomic-embed-text
   └── chat(messages) → Ollama → qwen3:8b
        │
        ▼
   pipelines.py
   ├── query(question) → search + LLM answer
   └── tag_notes(query) → search + LLM tag suggestions
```

## Components

### config.py
Loads environment variables from `.env` via `python-dotenv`. Exports all settings as module-level constants. Defines `EXCLUDE_PATTERNS` for vault traversal filtering.

### obsidian_client.py
Thin HTTP wrapper around the Obsidian Local REST API. Provides:
- `list_notes()` — recursively walks the vault, returns all `.md` paths
- `get_note(path)` — fetches note content as text
- `put_note(path, content)` — creates or overwrites a note

Excludes files/folders matching `EXCLUDE_PATTERNS` during traversal.

### llm_client.py
Wrapper around Ollama's embedding and chat APIs. Provides:
- `embed(text)` — returns a 768-dimensional float vector
- `chat(messages, model, think)` — sends chat messages to Ollama, returns response text
- `truncate_to_budget(text, max_words)` — truncates text to fit LLM context
- `_request_with_retry()` — all requests use 3 retries with exponential backoff for timeouts and transient HTTP errors

All HTTP requests use retry logic with exponential backoff (2s → 4s → 8s) for `ReadTimeout`, `ConnectionError`, and HTTP 429/502/503 responses.

### chroma_store.py
Local ChromaDB persistence layer. Provides:
- `upsert(path, chunk_idx, embedding, metadata)` — stores a chunk's embedding
- `delete_by_path(path)` — removes all chunks for a note (used before re-indexing)
- `query(embedding, n)` — semantic search, returns top-k results with distances

Document IDs use the format `{path}::chunk_{N}` so multiple chunks per note can coexist.

### indexer.py
One-shot indexing script and file watcher. Pipeline:
1. Fetch all note paths from Obsidian
2. For each note: get content, skip if under 20 words
3. Delete old chunks from ChromaDB (for re-indexing)
4. Split content into 500-word chunks (100-word overlap)
5. Embed each chunk, store in ChromaDB with metadata

Also provides:
- `add_tags_to_note()` for YAML frontmatter manipulation
- `_index_note()` / `_delete_note()` for incremental updates
- `watch()` — file watcher daemon using `watchdog`

The watcher monitors the vault directory and automatically re-indexes on changes with 2-second debounce.

### pipelines.py
LLM-powered pipelines for querying and auto-tagging. Provides:
- `query(question)` — search notes → fetch full content → LLM answers using vault context
- `tag_notes(query)` — search notes → LLM suggests tags → auto-applies via `add_tags`

Both pipelines use `truncate_to_budget()` to stay within context limits.

### mcp_server.py
FastMCP server exposing 9 vault tools via stdio transport. Wraps `obsidian_client`, `chroma_store`, `llm_client`, `indexer`, and `pipelines` for agent access. Logs all tool calls to `mcp_calls.log`.

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   Agent (Goose / Claude / Cursor)            │
└──────────────────────────────┬───────────────────────────────┘
                               │ MCP (stdio)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                      mcp_server.py                           │
│  search_notes │ read_note │ write_note │ ask_vault │ ...     │
└────────┬─────────────────────────┬───────────────────────────┘
         │                         │
         ▼                         ▼
┌──────────────┐          ┌──────────────┐
│  obsidian_   │          │  chroma_     │
│  client.py   │          │  store.py    │
└──────┬───────┘          └──────┬───────┘
       │                         │
       ▼                         ▼
┌──────────────┐          ┌──────────────┐
│  Obsidian    │          │  ChromaDB    │
│  REST API    │          │  (local)     │
└──────┬───────┘          └──────────────┘
       │
       ▼
┌──────────────┐          ┌──────────────┐
│  llm_client  │          │  pipelines   │
│  └─ embed()  │          │  └─ query()  │
│  └─ chat()   │          │  └─ tag()    │
└──────┬───────┘          └──────────────┘
       │
       ▼
┌──────────────┐
│  Ollama      │
│  ├─ nomic-embed-text (embeddings)
│  └─ qwen3:8b (chat/LLM)
└──────────────┘
```

## Chunking Strategy

Notes are split into overlapping chunks to stay within Ollama's 8192 token embedding limit:

- **Chunk size:** 500 words (~650 tokens)
- **Overlap:** 100 words (~130 tokens)
- **Why overlap:** Preserves context at chunk boundaries so semantic meaning isn't lost at split points

Example: A 1200-word note becomes 3 chunks:
```
Chunk 0: words 0-499
Chunk 1: words 400-899    (100-word overlap with chunk 0)
Chunk 2: words 800-1199   (100-word overlap with chunk 1)
```

## LLM Integration

Two modes of LLM usage:

### Query Mode (`ask_vault`)
1. Embed the user's question
2. Search ChromaDB for top-k relevant notes
3. Fetch full note content (truncated to 3000 words each)
4. Stuff into LLM prompt with system instructions
5. Return LLM-generated answer

### Action Mode (`tag_notes`)
1. Embed the search query
2. Search ChromaDB for matching notes
3. Fetch full note content
4. Send to LLM with instructions to suggest tags as JSON
5. Parse JSON response, apply tags via `add_tags`

## Excluded Content

The following patterns are skipped during vault traversal:

| Pattern | Reason |
|---------|--------|
| `_gsdata_` | GoodSync backup folder |
| `.gsbak` | GoodSync backup files |
| `.git` | Git repository data |
| `__pycache__` | Python bytecode cache |
| `node_modules` | Node.js dependencies |
| `.excalidraw.md` | Excalidraw diagram files (JSON, not readable text) |
