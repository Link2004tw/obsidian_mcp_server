# Changelog

## v0.3.1 (2026-06-07)

### New Actions
- **`notes(action=read_by_subject)`** — search for notes by subject/topic using LLM expansion + entity/graph/summary augmentation pipeline
- **`entities(action=link_note)`** — associate an existing note with an entity (creates entity if needed)

### Improvements
- **Structural Subjects folder hidden from search** — `read_by_title`, `search_by_tags`, `entities search` exclude `Subjects/` results; model tool descriptions no longer expose the internal path
- **`read_by_title` now returns path** — single-match responses include the note path header
- **`add_note_to_subject` auto-generates title** — when title omitted, extracts first meaningful line from content (no more "untitled" notes)
- **Root-level file logging** — all tool dispatch, search, and error logs captured in `logs/mcp_calls.log`

### Fixes
- Subjects folder hub notes no longer clutter search results

---

## v0.3.0 (2026-06-07)

### Tool Consolidation — 50+ tools → 9 dispatch tools
- **Massive reduction** — ~50 specialized MCP tools consolidated into 9 action-dispatch tools so small models don't get confused by too many choices
- **`ask(query)`** — universal discovery tool; uses LLM intent detection to route queries to 20 internal capabilities (search, Q&A, entities, graph, summaries, tags, stats, and more)
- **`notes(action, ...)`** — replaces `read_note`, `write_note`, `list_all_notes`, `list_folder`, `read_note_by_title`, `add_note_to_subject`, `search_by_tags`
- **`tags(action, ...)`** — replaces `add_tags`, `remove_tags`, `set_tags`, `batch_tag_notes`, `tag_notes`
- **`links(action, ...)`** — replaces `create_backlink`, `get_backlinks`, `get_linked_notes`, `get_broken_links`
- **`graph(action, ...)`** — replaces `get_communities`, `get_note_community`, `get_orphan_notes`, `get_shortest_path`, `get_graph_stats`, `related_notes`, `multi_hop_traversal`, `export_graph`
- **`entities(action, ...)`** — replaces `search_entities`, `get_note_entities`, `list_entities`, `get_entity_aliases`, `entity_timeline`, `related_entities`, `add_entity`, `merge_entities`, `change_entity_type`, `get_entity_types`, `get_ranking_weights`, `set_ranking_weights`, `import_entities`
- **`todo(action, ...)`** — replaces `get_todos`, `add_todo`, `complete_todo`, `update_todo`, `delete_todo`, `get_todo_stats`, `suggest_task_priority`, `suggest_due_date`, `suggest_task_splitting`, `get_overdue_summary`, `link_todo_to_notes`, `ask_vault_about_todo`/`ask_vault_about_todos`
- **`admin(action, ...)`** — replaces `health_check`, `sync_index`, `get_index_stats`, `switch_embedding_model`, `sync_todos`
- **`tools()`** — tool introspection, returns all tool names, descriptions, and parameter schemas as JSON
- **`_tool_base.py`** — new `build_tool()` decorator with standardized logging, exception handling, and `TOOL_MODULES` list (avoids circular imports)
- **Old `search.py`, `misc.py`, `todos.py`** deleted; `notes.py`/`graph.py` overwritten with consolidated versions
- **Tests** — 41 passing tests for consolidated API (`test_mcp_server.py` rewritten)
- **Backend** — `mcp_server.py` re-exports all 9 tools; `pipelines.py` updated; `log_error()` signature clarified

### Documentation
- `docs/mcp_server.md` — completely rewritten for 9-tool API
- `docs/api.md` — updated mcp_server.py section for consolidated tools
- `README.md` — tool tables replaced with 9-tool summary
- `CHANGELOG.md` — this entry

---

## v0.2.0 (2026-06-06)

### Multi-Provider LLM Support
- **New provider abstraction layer** — `providers/` package with `BaseLLMProvider` interface
- **OllamaProvider** — existing local Ollama support, refactored into a class
- **OpenAIProvider** — new provider supporting OpenAI + compatible APIs (Groq, Together, vLLM)
- **Provider registry** — auto-select via `LLM_PROVIDER` and `EMBED_PROVIDER` env vars (can differ)
- **Config** — new env vars: `LLM_PROVIDER`, `EMBED_PROVIDER`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBED_MODEL`
- **Backward compatible** — default provider remains Ollama, zero caller changes

### Cross-Vault Entity Resolution
- **`entity_resolver.py`** — new module for importing entities from other vaults
- **`import_entities` MCP tool** — import + merge entities with 3 dedup strategies (exact name, alias overlap, fuzzy similarity)
- Configurable matching thresholds via `dedup_config` parameter

### Documentation
- README, setup docs, and architecture docs updated for multi-provider support
- `.env.example` updated with all new variables

---

## v0.1.0 (2026-06-06)

Initial release of Obsidian AI — a local, privacy-first knowledge management layer for Obsidian vaults.

### Core Architecture
- **MCP Server** (`mcp_server.py`) — FastMCP-based server exposing 70+ tools via stdio transport
- **Modular tool organization** — tools split across `tools/search.py`, `tools/notes.py`, `tools/graph.py`, `tools/todos.py`, `tools/misc.py`
- **Path normalization** — all path parameters accept absolute or vault-relative paths interchangeably
- **Structured logging** — per-tool call logging to `mcp_calls.log`, indexer logging to `indexer.log`

### Indexing Pipeline
- **Three-phase indexing** — chunk + embed (Phase 1), entity extraction (Phase 2a), summary generation (Phase 2b)
- **Heading-aware chunking** — splits on Markdown headings (`#`, `##`, etc.) with 500-word chunks and 100-word overlap
- **Heading context prefix** — each chunk retains its parent heading path for structural context
- **Content hash tracking** — incremental indexing via file content hashes; only re-embeds changed notes
- **File watcher** (`--watch`) — monitors vault directory via `watchdog`; auto-indexes on create, modify, delete, rename
- **Mtime-based skip** — skips re-embedding for notes with unchanged modification time
- **Batch embedding** — supports `/api/embed` batch endpoint for efficient multi-chunk embedding
- **Parallel indexing** — `ThreadPoolExecutor` with configurable worker count for concurrent note processing

### Search & Retrieval (8 tools)
- `search_notes` — semantic search with metadata filters (`tags`, `folder`, `date_after/before`, `exclude_tags`), optional auto-rewrite, TTL-cached query expansion, diversity penalty, graph-aware proximity boost
- `batch_search` — run multiple semantic searches in a single call
- `composite_search` — high-recall composite search combining summary embeddings, entity expansion, and community-aware graph traversal
- `retrieve_notes` — multi-strategy retrieval (semantic + entity + graph + keyword) with full note content
- `find_duplicate_notes` — near-duplicate detection via embedding cosine similarity
- `search_by_tags` — find notes by YAML frontmatter tags (AND logic)
- `search_entities` — find notes mentioning a specific named entity
- `get_subject` — LLM-powered subject expansion + hybrid search for broad topic exploration

### Reading & Writing (7 tools)
- `read_note`, `write_note` — full CRUD against Obsidian REST API
- `list_all_notes`, `list_folder`, `list_folder_deep` — vault navigation
- `read_note_by_title` — look up notes by filename with optional folder scoping
- `add_note_to_subject` — create a note under a subject folder with auto-generated hub note and mutual backlinks

### Tag Management (5 tools)
- `add_tags`, `remove_tags`, `set_tags` — YAML frontmatter tag CRUD
- `batch_tag_notes` — apply tags to multiple notes in one call
- `tag_notes` — auto-suggest and apply tags via LLM (semantic search → LLM suggests tags → applies)

### Wiki-Link Graph (12 tools)
- `create_backlink` — bidirectional `[[wiki-link]]` creation
- `get_backlinks` / `get_linked_notes` — inbound/outbound link queries
- `get_broken_links` — find unresolved wiki-link targets
- `get_orphan_notes` — find notes with no wiki-links (disconnected)
- `get_graph_stats` — graph metrics: nodes, edges, avg degree, hubs, isolated notes
- `get_communities` / `get_note_community` — label-propagation community detection
- `multi_hop_traversal` — BFS traversal from seed note up to N hops
- `related_notes` — combined semantic + graph proximity ranking
- `export_graph` — graph export in DOT/JSON format
- `get_shortest_path` — shortest path between two notes via BFS

### LLM-Powered Tools (3 tools)
- `ask_vault` — natural language question answering using vault context (retrieve → LLM synthesize)
- `ask_agent` — agentic tool routing: LLM decides which search/graph tools to call based on query intent
- `summarize_topic` — multi-note consolidated summary via LLM

### Entity System (13 tools)
- `search_entities`, `get_note_entities`, `get_entity_types`, `get_entity_aliases`, `list_entities`
- `add_entity` — manually register entities with aliases and relationships
- `merge_entities` — deduplicate entity records in the index
- `import_entities` — cross-vault entity resolution with configurable dedup strategies
- `entity_timeline` — chronological timeline of entity mentions across notes
- `related_entities` — entity-relationship graph traversal (works_on, uses, part_of, etc.)
- `get_ranking_weights` / `set_ranking_weights` — view and adjust ranking signal weights at runtime

### Clustering (1 tool)
- `get_clusters` — semantic clustering via connected-components on embedding similarity graph; configurable threshold; TTL-cached to `data/clusters.json`

### Health & Index Management (3 tools)
- `health_check` — backend service status (Ollama/OpenAI, ChromaDB, Obsidian API)
- `sync_index` — on-demand full re-index pipeline
- `get_index_stats` — index diagnostics (chunks, notes, embedding model, cache stats)
- `switch_embedding_model` — runtime model switching with auto re-index

### Todo Management (20 tools)
- Core CRUD: `get_todos`, `add_todo`, `complete_todo`, `update_todo`, `delete_todo`, `sync_todos`
- Smart queries: `get_todo_stats`, `get_todos_by_priority`, `get_overdue_summary`
- LLM-powered: `add_todo_from_natural_language`, `suggest_task_priority`, `suggest_due_date`, `suggest_task_splitting`, `estimate_completion_date`
- Vault integration: `get_todos_for_note`, `get_notes_for_todo`, `link_todo_to_notes`, `ask_vault_about_todo`, `ask_vault_about_todos`
- Lifecycle: `ensure_todo_file`

### CLI
- `obsidian-ai` entry point via `cli.py` — commands: `index`, `watch`, `sync`, `stats`, `search`, `read`, `write`, `list-all`, `list-folder`, `list-folder-deep`, `read-by-title`, `search-by-tags`, `add-tags`, `create-backlink`, `ask`, `tag-notes`, `dashboard`, `eval`

### Dashboard
- `obsidian-ai dashboard` — generates standalone interactive HTML dashboard with graph metrics, entity stats, todo summaries, and index health
- Live server mode (`--serve`, port 8765) for real-time exploration

### Infrastructure
- **Multi-signal ranking** — `Ranker` class blending semantic (0.40), entity (0.30), graph (0.20), and keyword (0.10) signals with thread-safe weights
- **Hybrid search** (BM25 + semantic) — keyword fallback configurable via `keyword_weight`
- **LLM query expansion** — TTL-cached synonym expansion for broader recall
- **Query auto-rewrite** — uses known vault terminology (entity names, note titles) to rewrite queries
- **Intent detection** — auto-detects entity/keyword/graph-focused queries and adjusts ranking weights dynamically
- **Embedding cache** — shared LRU cache (in-memory + persistent to disk via `embed_cache.json`)
- **Entity / summary caches** — per-content-hash caching with disk persistence, thread-safe
- **Config validation** — startup checks for Ollama connectivity, model existence, vault dir writability, API key presence
- **GPU safety** — configurable temperature/VRAM limits prevent GPU TDR crashes during indexing
- **Disk temperature monitoring** — optional SSD temp checks during long index runs

### Known Limitations
- **Delta indexing is not yet optimized** — when a note changes, all its chunks are re-embedded rather than only the changed ones. This is the primary optimization opportunity for v0.2.0.
- **Embedding is sequential per-note** — embeddings are computed one chunk at a time; batch embedding is supported but not yet parallelized across notes during indexing.
- **Entity extraction is LLM-intensive** — the pipeline calls the LLM for every chunk during entity extraction, which is the dominant time cost for large vaults.
- **Watcher event coalescing** — rapid file saves can generate overlapping index operations; a queue/coalescing strategy is planned.
- **ChromaDB stats** — `get_index_stats` fetches all metadata, which is slow for very large vaults (>10k chunks). A dedicated note-level collection is planned.
