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
    ├── list_all_notes()
    ├── list_folder(folder_path)         ← non-recursive
    ├── list_folder_deep(folder_path)    ← recursive
    ├── add_tags(path, tags)
    ├── create_backlink(a, b)
    ├── read_note_by_title(title, folder_path)
    ├── get_index_stats()
    ├── sync_index()
    ├── ask_vault(question)              ← LLM-powered
    ├── tag_notes(query)                 ← LLM-powered
    ├── search_entities(query)           ← entity-aware
    ├── get_note_entities(path)
    └── get_entity_types()
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
- `list_all_notes()` — recursively walks the vault, returns all `.md` paths
- `list_folder(folder_path)` — lists `.md` files and subdirectories in a folder (non-recursive)
- `list_folder_deep(folder_path)` — recursively walks a folder, returns all `.md` paths
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

### entity_store.py
Persistent inverted index mapping entity names to note paths. Entities are extracted per-note during indexing via LLM (Qwen3:8b) and stored as:

- **`data/entities.json`** — entity index (JSON), rebuilt from ChromaDB metadata on `sync_index`
- **`entities_str` metadata** — per-chunk string like `,Technology:Python,Project:Obsidian,` for ChromaDB `$contains` queries
- **Graph nodes** — entity nodes (`__entity:{type}:{name}`) in `GraphStore` for cross-referencing

Provides: `add`, `search`, `search_by_type`, `get_note_entities`, `rebuild`, `stats`, `entity_types`.

### indexer.py
One-shot indexing script and file watcher. Pipeline:
1. Fetch all note paths from Obsidian
2. For each note: get content, skip if under 20 words
3. Delete old chunks from ChromaDB (for re-indexing)
4. Split content into 500-word chunks (100-word overlap)
5. Extract entities via LLM (cached per path per run)
6. Embed each chunk, store in ChromaDB with metadata (including `entities_str`)

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

### graph_store.py
In-memory wiki-link graph built during indexing. Uses a dict-based adjacency list (`_adj: dict[str, set[str]]`), populated by extracting `[[wikilinks]]` from note content. Supports BFS traversal, broken link detection, community detection (label propagation), and graph export (DOT/JSON). Excludes `__entity:*` nodes from stats, orphan detection, and exports.

### mcp_server.py
FastMCP server exposing vault tools via stdio transport. Wraps `obsidian_client`, `chroma_store`, `llm_client`, `indexer`, `pipelines`, `entity_store`, and `graph_store` for agent access. Logs all tool calls to `mcp_calls.log`.

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   Agent (Goose / Claude / Cursor)            │
└──────────────────────────────┬───────────────────────────────┘
                               │ MCP (stdio)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      mcp_server.py                               │
│  search_notes │ read_note │ write_note │ ask_vault │ ...         │
│  search_entities │ get_note_entities │ get_entity_types           │
└────────┬─────────────────────────┬───────────────────────────────┘
         │                         │
         ▼                         ▼
┌──────────────┐          ┌──────────────────┐
│  obsidian_   │          │  chroma_store    │
│  client.py   │          │  + entity_store  │
└──────┬───────┘          │  + graph_store   │
       │                  └────────┬─────────┘
       ▼                           │
┌──────────────┐                   │
│  Obsidian    │                   ▼
│  REST API    │          ┌──────────────────┐
└──────┬───────┘          │  ChromaDB        │
       │                  │  entities.json   │
       ▼                  │  graph (memory)  │
┌──────────────┐          └──────────────────┘
│  llm_client  │          ┌──────────────┐
│  └─ embed()  │          │  pipelines   │
│  └─ chat()   │          │  └─ query()  │
└──────┬───────┘          │  └─ tag()    │
       │                  │  └─ extract_ │
       │                  │     entities()│
       ▼                  └──────────────┘
┌──────────────┐
│  Ollama      │
│  ├─ nomic-embed-text (embeddings)
│  └─ qwen3:8b (chat/LLM + entity extraction)
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

Three modes of LLM usage:

### Entity Extraction (`extract_entities`)
1. During indexing, each note's sanitized content is sent to Qwen3:8b with an extraction prompt
2. LLM returns JSON entities: `[{"name": "...", "type": "...", "confidence": 0.0-1.0}]`
3. Entities are stored in `entity_store` and written as `entities_str` metadata on every chunk
4. Subsequent `sync_index` rebuilds the entity store from ChromaDB metadata (no LLM calls)

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
