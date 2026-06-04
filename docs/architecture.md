# Architecture

## Overview

The system reads notes from Obsidian, embeds them as semantic vectors, and stores them in a local vector database for search. An MCP server exposes these capabilities to AI agents. All processing runs locally — no data leaves the machine.

```
Agent (Goose / Claude / Cursor / opencode)
        │
        ▼
    mcp_server.py ──── FastMCP (stdio) ──── 67 tools
    ├── Search & Retrieval (search_notes, batch_search, retrieve_notes, ...)
    ├── Read & Write (read_note, write_note, list_all_notes, ...)
    ├── Tag Management (add_tags, remove_tags, set_tags, tag_notes, ...)
    ├── Entity System (search_entities, get_note_entities, ...)
    ├── Graph (create_backlink, related_notes, communities, ...)
    ├── LLM-Powered (ask_vault, summarize_topic, ask_agent, ...)
    ├── Clustering (get_clusters)
    ├── Index Management (sync_index, switch_embedding_model, ...)
    └── Todo Management (add_todo, get_todos, complete_todo, ...)
        │
        ├──────────────────────────────────┬──────────────────────┐
        ▼                                  ▼                      ▼
obsidian_client.py                  chroma_store.py         pipelines.py
        │                            + entity_store           ├── query()
        ▼                            + graph_store            ├── tag_notes()
  Obsidian REST API                                          ├── summarize_topic()
  (port 27123)                    ChromaDB (./chroma_db/)     ├── extract_entities()
        │                             │                      ├── expand_query()
        ▼                             ▼                      └── route_query()
   llm_client.py               data/*.json (caches,
   ├── embed(text) → Ollama →  content hashes, entities,        graph_store.py
   │   nomic-embed-text         summaries, title maps)          ├── BFS traversal
   └── chat(messages) → Ollama                                 ├── community detection
       qwen3:8b                                                 ├── orphan/broken links
                                                                └── DOT/JSON export
```

## Components

### config.py
Loads environment variables from `.env` via `python-dotenv`. Exports all settings as module-level constants including `data_dir`, `chunk_size`, `chunk_overlap`, and `EXCLUDE_PATTERNS`.

### obsidian_client.py
Thin HTTP wrapper around the Obsidian Local REST API. Provides list, read, write, and directory operations. Cache files (`note_paths.json`, `title_to_path.json`) stored in `config.data_dir`.

### llm_client.py
Wrapper around Ollama's embedding and chat APIs. Provides:
- `embed(text)` — returns a vector (768-dim for `nomic-embed-text`)
- `chat(messages, model, think)` — chat completion with retry
- `truncate_to_budget(text, max_words)` — truncation for context limits
- `clear_embed_cache()` / `embed_cache_info()` — cache management
- `switch_embed_model(model_name)` — runtime model switching
- `_request_with_retry()` — 3 retries with exponential backoff (2s → 4s → 8s)

Embeddings are cached in memory by text hash. Chat requests are limited to 1 concurrent call via `_llm_chat_lock` (semaphore) to prevent Ollama timeout pileup during parallel indexing.

### chroma_store.py
Local ChromaDB persistence layer. Provides:
- `upsert`, `delete_by_path`, `get_by_path`, `get_by_title`, `query` — core CRUD
- `get_all_documents`, `get_all_embeddings`, `get_metadata_by_ids` — bulk access
- `find_duplicate_notes` — semantic dedup via embedding similarity
- `search_by_tags` — tag-based lookup via client-side filtering (workaround for ChromaDB `$contains` bug)
- `get_index_stats` — chunk and note counts
- `reset_collection` — wipe for model switching

Document IDs use the format `{path}::chunk_{N}`.

### entity_store.py
Persistent inverted index mapping entity names to note paths. Entities are extracted per-note during indexing via LLM (Qwen3:8b) and stored as:
- **`data/entities.json`** — entity index (JSON), rebuilt from ChromaDB metadata on `sync_index`
- **`entities_str` metadata** — per-chunk string for client-side entity filtering
- **Graph nodes** — entity nodes in `GraphStore` for cross-referencing

Provides: `add`, `search`, `search_by_type`, `get_note_entities`, `rebuild`, `stats`, `entity_types`.

### graph_store.py
In-memory wiki-link graph built during indexing. Dict-based adjacency list (`_adj: dict[str, set[str]]`). Built by extracting `[[wikilinks]]` from note content. Provides:
- BFS traversal (`multi_hop_traversal`)
- Backlinks / outgoing links
- Broken link detection
- Orphan note detection
- Community detection (label propagation)
- Graph export (DOT / JSON)
- Entity node support (`__entity:{type}:{name}`), excluded from stats/export

### indexer.py
Indexing pipeline with incremental updates and file watcher. Three-phase batch embedding for efficiency:
1. **Fetch** all note paths from Obsidian
2. **Phase 1 — Prepare:** For each note, get content, skip if under 20 words, delete old chunks, split into chunks (heading-aware, 500-word with 100-word overlap), extract entities via LLM (cached), generate summary via LLM (cached), check delta hashes (skip unchanged chunks).
3. **Phase 2 — Batch Embed:** All prepared chunks are embedded in a single batch call to Ollama (faster than per-note embedding).
4. **Phase 3 — Finalize:** Store embedded chunks in ChromaDB with full metadata.

Supports `SKIP_ENTITIES` and `SKIP_SUMMARIES` flags to skip LLM-dependent steps. Uses `_llm_chat_lock` semaphore to limit concurrent LLM calls during parallel indexing.

### pipelines.py
LLM-powered pipelines:
- `query()` — search → fetch content → LLM answers using vault context (supports `auto_rewrite`)
- `tag_notes()` — search → LLM suggests tags → auto-applies via `add_tags`
- `summarize_topic()` — multi-note LLM summary (supports `auto_rewrite`)
- `retrieve()` — multi-strategy retrieval pipeline (supports `auto_weights`, `auto_rewrite`)
- `extract_entities()` — entity extraction (cached, with retry)
- `expand_query()` — LLM query expansion for broader search (TTL-cached)
- `_rewrite_query()` — rewrites queries using known vault terminology (entities, titles)
- `_get_vault_terminology()` — collects entity names and note titles for query context
- `route_query()` — agentic tool routing (`AGENT_SYSTEM` prompt with 6 tools)
- `detect_intent()` — auto-detects query intent (entity, keyword, or graph-focused)

All pipelines use `truncate_to_budget()` to stay within context limits. Summary from chunk-0 metadata is injected into prompts when available.

### wiki_links.py
Wiki-link parsing and normalization. Extracts `[[wikilinks]]` from markdown text, handles display text (`|`), section anchors (`#`), and deduplication.

### frontmatter.py
YAML frontmatter parsing and manipulation. Used by tag operations.

### clustering.py
Semantic clustering module for grouping notes by meaning. Connected-components clustering based on embedding cosine similarity. Provides:
- `get_clusters()` — return clusters with auto-generated labels
- TTL-cached results persisted to `data/clusters.json`
- Configurable similarity threshold
- Dashboard integration for visual exploration

Uses `chroma_store.get_all_embeddings()` to fetch embedding vectors. Has zero additional Python dependencies (pure NumPy/math).

### mcp_server.py
FastMCP server exposing 67 vault tools via stdio transport. All path parameters are auto-normalized (absolute or vault-relative). Includes a `health_check` tool for backend service status. Wraps all other modules for agent access. Logs all tool calls to `mcp_calls.log`.

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   Agent (MCP client)                         │
└──────────────────────────────┬───────────────────────────────┘
                               │ MCP (stdio)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      mcp_server.py (67 tools)                    │
└────────┬─────────────────────────┬───────────────────────────────┘
         │                         │
         ▼                         ▼
┌──────────────┐          ┌──────────────────┐
│  obsidian_   │          │  chroma_store    │
│  client.py   │          │  + entity_store  │
│  + wiki_links│          │  + graph_store   │
└──────┬───────┘          └────────┬─────────┘
       │                           │
       ▼                           ▼
┌──────────────┐          ┌──────────────────┐
│  Obsidian    │          │  ChromaDB        │
│  REST API    │          │  entities.json   │
└──────┬───────┘          │  graph (memory)  │
       │                  └──────────────────┘
       ▼                  ┌──────────────┐
┌──────────────┐          │  pipelines   │
│  llm_client  │          │  ├─ query()  │
│  ├─ embed()  │          │  ├─ tag()    │
│  └─ chat()   │          │  ├─ extract  │
└──────┬───────┘          │  │  entities()│
       │                  │  ├─ expand   │
       ▼                  │  │  query()  │
┌──────────────┐          │  └─ route    │
│  Ollama      │          │     query()  │
│  ├─ nomic-embed-text    └──────────────┘
│  └─ qwen3:8b
└──────────────┘
```

## Chunking Strategy

Notes are split into overlapping chunks to stay within Ollama's embedding limit:

- **Chunk size:** 500 words (~650 tokens)
- **Overlap:** 100 words (~130 tokens)
- **Heading-aware:** Notes are split by markdown headings first, then large sections are chunked. Each chunk retains its heading context.

## LLM Integration

### Entity Extraction
1. During indexing, each note's sanitized content is sent to Qwen3:8b with an extraction prompt
2. LLM returns JSON entities: `[{"name": "...", "type": "...", "confidence": 0.0-1.0}]`
3. Entities are stored in `entity_store` and written as `entities_str` metadata on every chunk
4. Results are cached per content hash (in-memory + `entity_cache.json`)

### Summary Generation
1. During indexing, each note's sanitized content is sent to Qwen3:8b with a summarization prompt
2. Summary (2-3 sentences) is stored in chunk-0's ChromaDB metadata
3. Passed through `retrieve()` results and used in `query()`, `summarize_topic()`, `tag_notes()` context
4. Results are cached per content hash (in-memory + `summary_cache.json`)

### Agentic Routing (`route_query`)
1. LLM receives the `AGENT_SYSTEM` prompt describing 6 available tools (search_notes, read_note, get_backlinks, get_linked_notes, related_notes, multi_hop_traversal)
2. LLM decides which tool to call and with what parameters
3. The tool is called and its result returned
4. Used by the `ask_agent` MCP tool

### Embedding Model Switching
1. MCP tool `switch_embedding_model` verifies the model exists in Ollama
2. Clears embedding cache, resets ChromaDB collection, updates config
3. Triggers full re-index with the new model

## Concurrency

- Parallel indexing uses 2-6 workers via `ThreadPoolExecutor`
- At most 1 `llm_client.chat()` call runs at any time (entity extraction and summary generation are serialized via `_llm_chat_lock`)
- Embedding calls (`embed()`) are not rate-limited (they're fast)
- Entity and summary caches are thread-safe (per-cache locks)

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
