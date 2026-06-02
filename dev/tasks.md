# Tasks

All tasks are organized by phase. Difficulty: `Low` / `Medium` / `High`. Priority: `P1` (must) / `P2` (should) / `P3` (nice to have).

---

## Phase 1 — Vault API + Embeddings Foundation

| #    | Task                                   | Description                                                                                                      | Difficulty | Priority | Status |
| ---- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------- | -------- | ------ |
| 1.1  | Install Obsidian Local REST API plugin | Enable the community plugin, grab the API key and confirm it responds on port 27123                              | Low        | P1       | ✅     |
| 1.2  | Write `obsidian_client.py`             | Thin wrapper around the REST API: `list_notes()`, `get_note(path)`, `put_note(path, content)`                    | Low        | P1       | ✅     |
| 1.3  | Pull `nomic-embed-text` model          | Run `ollama pull nomic-embed-text` and verify it responds via `POST /api/embeddings`                             | Low        | P1       | ✅     |
| 1.4  | Write `llm_client.py` — embed function | Function that takes a string and returns a 768-dim vector via Ollama                                             | Low        | P1       | ✅     |
| 1.5  | Write `chroma_store.py`                | Initialize ChromaDB client and collection; implement `upsert(id, embedding, metadata)` and `query(embedding, n)` | Medium     | P1       | ✅     |
| 1.6  | Write `indexer.py` — one-shot mode     | Read all notes via Obsidian client, embed each, upsert to ChromaDB with `path`, `title`, `mtime`                 | Medium     | P1       | ✅     |
| 1.7  | Write `config.py`                      | Centralize vault path, Obsidian API key/port, Ollama base URL, model names, ChromaDB path                        | Low        | P1       | ✅     |
| 1.8  | Test semantic search manually          | Run a Python snippet: embed a query string, call `chroma_store.query()`, print top-5 results                     | Low        | P1       | ✅     |
| 1.9  | Handle empty and very short notes      | Skip notes under ~20 tokens to avoid noisy embeddings                                                            | Low        | P2       | ✅     |
| 1.10 | Store note word count in metadata      | Useful later for context-stuffing decisions in the LLM layer                                                     | Low        | P3       | ✅     |

---

## Phase 2 — MCP Server

| #    | Task                                             | Description                                                                                | Difficulty | Priority | Status |
| ---- | ------------------------------------------------ | ------------------------------------------------------------------------------------------ | ---------- | -------- | ------ |
| 2.1  | Install and configure `fastmcp`                  | `pip install fastmcp`, scaffold `mcp_server.py`, verify server starts                      | Low        | P1       | ✅     |
| 2.2  | Implement `search_notes(query)` tool             | Embed the query, search ChromaDB, return top-k note paths + snippets                       | Medium     | P1       | ✅     |
| 2.3  | Implement `read_note(path)` tool                 | Fetch full note content from Obsidian REST API by path                                     | Low        | P1       | ✅     |
| 2.4  | Implement `write_note(path, content)` tool       | Create or overwrite a note via Obsidian REST API                                           | Low        | P1       | ✅     |
| 2.5  | Implement `list_all_notes()` tool                | Return a flat list of all note paths in the vault                                          | Low        | P1       | ✅     |
| 2.6  | Implement `list_folder(folder_path)` tool        | Return a list of note paths within a specific folder (non-recursive)                       | Low        | P1       | ✅     |
| 2.7  | Implement `add_tags(path, tags)` tool            | Parse YAML frontmatter, merge new tags, write note back; create frontmatter if absent      | High       | P1       | ✅     |
| 2.8  | Implement `create_backlink(path_a, path_b)` tool | Append `[[note_b]]` to note A and `[[note_a]]` to note B if not already present            | Medium     | P1       | ✅     |
| 2.9  | Implement `sync_index()` tool                    | Re-run the full indexer pipeline on demand; returns count of notes indexed                 | Medium     | P2       | ✅     |
| 2.10 | Test all tools via MCP inspector                 | Call each tool directly and verify correct input/output before any agent is involved       | Low        | P1       | ✅     |
| 2.11 | Add error handling to all tools                  | Return structured error messages instead of raw exceptions for missing notes, API failures | Medium     | P2       | ✅     |
| 2.12 | Log all tool calls to a file                     | Append timestamped tool name + args to `mcp_calls.log` for debugging                       | Low        | P3       | ✅     |
| 2.13 | Implement `list_folder_deep(folder_path)` tool   | Return all note paths within a folder (recursive traversal)                                | Low        | P2       | ✅     |

---

## Phase 3 — LLM + Agent Integration

| #    | Task                                  | Description                                                                                      | Difficulty | Priority | Status |
| ---- | ------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------- | -------- | ------ |
| 3.1  | Write `llm_client.py` — chat function | Function that sends a messages array to Qwen3:8b via Ollama and returns the text response        | Low        | P1       | ✅     |
| 3.2  | Implement query mode pipeline         | Embed query → `search_notes` → fetch top-3 full contents → stuff into LLM prompt → return answer | Medium     | P1       | ✅     |
| 3.3  | Implement action mode pipeline        | `search_notes` → for each result, LLM suggests tags → call `add_tags` per note                   | High       | P1       | ✅     |
| 3.4  | Configure Goose to use the MCP server | Add `mcp_server.py` to Goose's MCP config, verify Goose can list and call tools                  | Low        | P1       | ✅     |
| 3.5  | Test end-to-end query flow in Goose   | "Find notes related to X" → confirms correct notes are returned with context                     | Medium     | P1       | ✅     |
| 3.6  | Test end-to-end action flow in Goose  | "Tag all notes about X" → confirms tags appear in Obsidian frontmatter                           | Medium     | P1       | ✅     |
| 3.7  | Test backlink creation flow           | "Link all notes about X to each other" → confirms `[[links]]` appear in correct notes            | Medium     | P2       | ✅     |
| 3.8  | Tune LLM prompt for tagging           | Iterate on the system prompt until suggested tags are concise and consistent in format           | Medium     | P2       | ✅     |
| 3.9  | Add context length guard              | Truncate note content before stuffing into LLM context if total tokens exceed a safe limit       | Medium     | P2       | ✅     |
| 3.10 | Test with Qwen3 thinking mode off     | Add `/no_think` prefix to prompts where speed matters more than reasoning depth                  | Low        | P3       | ✅     |

---

## Phase 4 — Incremental Indexing + Polish

| #    | Task                                        | Description                                                                                                                        | Difficulty | Priority | Status |
| ---- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | ------ |
| 4.1  | Add `watchdog` file watcher to `indexer.py` | Monitor vault directory; on file save, re-embed only the changed note and upsert to ChromaDB                                       | Medium     | P1       | ✅     |
| 4.10 | Fix watcher to use configurable VAULT_PATH  | Replace hardcoded project-root path with `config.vault_path` from .env; strip absolute path prefix for API-friendly relative paths | Medium     | P1       | ✅     |
| 4.11 | Remove debug print from obsidian_client.py  | Delete leftover `print(_headers())` from `_list_dir` that was leaking the API token to stdout                                      | Low        | P1       | ✅     |
| 4.12 | Lift inline import in mcp_server.py         | Move `from .frontmatter import parse` from inside `add_tags()` to the top-level imports                                            | Low        | P3       | ✅     |
| 4.2  | Handle note deletion                        | On file delete event, remove the corresponding ChromaDB entry by ID                                                                | Medium     | P1       | ✅     |
| 4.3  | Handle note rename                          | Delete old ChromaDB entry, re-index under new path                                                                                 | Medium     | P1       | ✅     |
| 4.4  | Add `--watch` flag to `indexer.py`          | `python indexer.py --watch` starts the watcher; without flag does a one-shot full index                                            | Low        | P1       | ✅     |
| 4.5  | Skip re-embedding if `mtime` unchanged      | Compare stored `mtime` in ChromaDB metadata before re-embedding to avoid unnecessary work                                          | Low        | P2       | ✅     |
| 4.6  | Write `README.md`                           | Setup instructions, usage examples, config reference, Goose integration guide                                                      | Low        | P2       | ✅     |
| 4.7  | Write a basic CLI wrapper                   | `cli.py` with commands: `index`, `watch`, `search <query>`, `tag <query>` using `argparse`                                         | Medium     | P3       | ✅     |
| 4.8  | Add index stats command                     | Print total notes indexed, ChromaDB collection size, last index time                                                               | Low        | P3       | ✅     |
| 4.9  | Test full cold-start setup                  | Clone repo on a fresh env, follow README, confirm system works end-to-end                                                          | Low        | P2       | ✅     |

---

## Phase 5 — Search Improvements

| #    | Task                                       | Description                                                                                                                                                                                                                                                                           | Difficulty | Priority | Status |
| ---- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | ------ |
| 5.1  | Hybrid search (BM25 + semantic)            | Implement BM25/TF-IDF keyword fallback when semantic results are low-confidence; blend scores with configurable `keyword_weight` and `min_similarity`                                                                                                                                 | High       | P2       | ✅     |
| 5.2  | Implement `search_by_tags(tags)` function  | New function + MCP tool that queries ChromaDB using `where` filter on metadata `tags` field to find notes with matching tags. Requires storing tags in ChromaDB metadata during indexing. Return paths + snippets.                                                                    | Medium     | P1       | ✅     |
| 5.3  | Implement `read_note_by_title(title)` tool | New MCP tool that looks up a note by its title (basename without extension) in ChromaDB metadata, returns the full note content. Supports optional `folder_path` to disambiguate duplicates.                                                                                          | Low        | P1       | ✅     |
| 5.4  | Metadata filtering & faceting              | Add optional filters to `search_notes`: `tags`, `folder`, `exclude_tags`, `date_after`, `date_before` for scoped searches                                                                                                                                                             | Medium     | P2       | ✅     |
| 5.5  | Passage-level search returns               | Return matching chunk snippet with context window instead of full note path; include `snippet`, `matched_chunk_idx`, `similarity_score`                                                                                                                                               | Medium     | P2       | ✅     |
| 5.6  | Query expansion (LLM)                      | Use Qwen to expand queries with synonyms before embedding (e.g., "PID tuning" → "proportional integral derivative control, feedback loop")                                                                                                                                            | Medium     | P3       | ✅     |
| 5.7  | Relevance threshold & diversity            | Add `min_similarity` filter; implement `diversity_penalty` to penalize results too similar to already-selected ones                                                                                                                                                                   | Low        | P3       | ✅     |
| 5.8  | Search caching                             | LRU cache (`max_size=100`) for query embeddings + results; common queries become ~1ms after first run                                                                                                                                                                                 | Low        | P3       | ✅     |
| 5.9  | Implement `get_related_notes(path)`        | New MCP tool that takes a note path, retrieves its embedding from ChromaDB, then performs semantic search to find the most similar notes. Returns top-k related note paths + snippets + similarity scores. Exclude the source note itself.                                            | Medium     | P1       | ✅     |
| 5.10 | Implement `get_subject(subject)`           | New MCP tool that takes a subject/topic string, uses LLM to expand the subject with related terms, then performs hybrid search (semantic + BM25) across the vault to find all notes related to that subject. Returns grouped results by relevance with optional tag/folder filtering. | High       | P2       | ✅     |

---

## Phase 6 — Todo Management System

| #   | Task                        | Description                                                                                                                                                                        | Difficulty | Priority | Status |
| --- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | ------ |
| 6.1 | Todo file structure         | Design `todos.md` format: YAML frontmatter (last_synced, counts), Markdown headers for projects, `[ ]`/`[x]` checkboxes, inline metadata in parentheses `(due:, priority:, tags:)` | Medium     | P2       | ⬜     |
| 6.2 | Auto-create todo file       | `ensure_todos_file_exists()` on first index/MCP startup or when file is deleted; create default templated `todos.md`                                                               | Low        | P2       | ⬜     |
| 6.3 | Core todo MCP tools         | Implement `get_todos`, `add_todo`, `complete_todo`, `update_todo`, `delete_todo`, `sync_todos` — full CRUD against `todos.md`                                                      | High       | P1       | ⬜     |
| 6.4 | Smart todo queries          | `get_overdue_todos`, `get_blocked_todos`, `get_todos_by_project`, `search_todos` (semantic), `get_todo_stats`                                                                      | Medium     | P2       | ⬜     |
| 6.5 | LLM-powered todo features   | Natural language todo creation (`add_todo_from_natural_language`), `suggest_task_priority`, `suggest_due_date`, `suggest_task_splitting`, `suggest_task_dependencies`              | High       | P3       | ⬜     |
| 6.6 | Todo reporting & metrics    | `get_todos_by_priority`, `get_burndown_chart(project, days)`, `get_overdue_summary`, `estimate_completion_date`                                                                    | Medium     | P3       | ⬜     |
| 6.7 | Todo integration with vault | `get_todos_for_note`, `get_notes_for_todo`, `link_todo_to_notes`, `ask_vault_about_todo`, `ask_vault_about_todos`                                                                  | Medium     | P3       | ⬜     |

---

## Phase 7 — Advanced Features

| #   | Task                                 | Description                                                                                                                                       | Difficulty | Priority | Status |
| --- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | ------ |
| 7.1 | Incremental indexing via file hashes | Track file content hashes in JSON; only re-embed changed notes (delta updates) instead of re-indexing entire vault                                | Medium     | P2       | ⬜     |
| 7.2 | Entity extraction                    | Extract named entities (people, projects, hardware, dates, concepts) using LLM during indexing; store in ChromaDB metadata for `search_by_entity` | High       | P3       | ⬜     |
| 7.3 | Note summaries                       | Pre-generate 1-2 sentence summaries during indexing; `ask_vault` uses summary first, loads full content only if needed                            | High       | P3       | ⬜     |
| 7.4 | Embedding model switching            | Make embedding model configurable at runtime; implement `switch_embedding_model(new_model)` that re-indexes with new model                        | Medium     | P3       | ⬜     |
| 7.5 | Performance metrics                  | `get_index_stats()` exposing total_notes, total_chunks, index_size_mb, avg_query_latency_ms, cache_hit_rate, last_sync, embedding_model           | Medium     | P3       | ✅     |
| 7.6 | Batch operations                     | `batch_search(queries)` returning `dict[query, results]`; `batch_tag_notes(note_paths, tags)` for bulk tagging                                    | Low        | P3       | ⬜     |
| 7.7 | Semantic deduplication               | `find_duplicate_notes(threshold)` using embedding similarity to detect near-duplicate note pairs                                                  | Low        | P3       | ⬜     |

---

## Phase 8 — Performance optimization follow-ups

| #   | Task                                     | Description                                                                                                                                                    | Difficulty | Priority | Status |
| --- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | ------ |
| 8.1 | Apply indexing throughput optimizations  | Implement faster indexing by (a) parallelizing/batching embeddings during indexing and/or (b) reusing chunk embeddings when identical text occurs across notes | Medium     | P2       | ✅     |
| 8.2 | Optimize incremental indexing skip check | Remove per-note Chroma reads in `_should_skip_by_mtime()` by preloading note→stored_mtime map once per index run (or redesign note-level metadata storage)     | Low        | P2       | ✅     |
| 8.3 | Reduce watcher thrash                    | Replace per-event `threading.Thread` spawning with a queue/coalescing strategy so many file save events do not cause redundant indexing/deleting work          | Medium     | P2       | ✅     |
| 8.4 | Cache query expansion                    | Add caching for LLM-generated expanded query phrases to reduce repeated expansion cost in search flows                                                         | Low        | P3       | ⬜     |

---

## Phase 9 — Bug Fixes

| #   | Task                                          | Description                                                                                                                                                     | Difficulty | Priority | Status |
| --- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | ------ |
| 9.1 | Fix duplicate `_expand_query` in mcp_server.py | Remove the first bare `def _expand_query` (line 29) and dead code between the two definitions; keep only the `lru_cache`-decorated version                      | Low        | P1       | N/A (stale) |
| 9.2 | Fix `ondef` typo in mcp_server.py             | `ondef _matches_where(...)` should be `def _matches_where(...)` — syntax error preventing import                                                                | Low        | P1       | N/A (stale) |
| 9.3 | Fix `local_embedding_cache` thread-safety      | `indexer.py` line 121 — dict written from multiple `ThreadPoolExecutor` workers without a lock; add a `threading.Lock` or use a thread-safe cache               | Low        | P2       | ✅     |
| 9.4 | Fix BM25 `ensure_index` race condition         | `keyword_search.py` lines 59-64 — count check and rebuild happen inside lock but `search()` acquires lock twice, allowing staleness between acquisitions         | Medium     | P2       | ✅     |

---

## Phase 10 — Security Hardening

| #   | Task                                  | Description                                                                                                                                                  | Difficulty | Priority | Status |
| --- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | -------- | ------ |
| 10.1 | Add path traversal protection        | Validate all user-supplied paths in `obsidian_client.py` and `mcp_server.py`: reject `..` sequences, enforce vault-relative paths, canonicalize before use    | Medium     | P1       | ✅     |
| 10.2 | Sanitize LLM prompt inputs           | `pipelines.tag_notes` interpolates note content directly into prompts — escape or bracket user content to mitigate prompt injection                          | Medium     | P2       | ✅     |
| 10.3 | Mask API key in logs/config repr     | Ensure `obsidian_api_key` is not exposed if `config` is repr'd or logged; add redaction to logger                                                           | Low        | P2       | ✅     |
| 10.4 | Validate `write_note` paths          | Add sandboxing or allowlist check so MCP clients cannot write to arbitrary paths the Obsidian API allows                                                    | Medium     | P2       | ✅     |

---

## Phase 11 — Tech Debt / Refactoring

| #   | Task                                          | Description                                                                                                                                                     | Difficulty | Priority | Status |
| --- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | ------ |
| 11.1 | Extract `_hybrid_search()` blending function  | `search_notes` and `get_subject` in `mcp_server.py` have near-identical semantic+BM25 merge logic — extract to a shared function                                | Medium     | P2       | ✅     |
| 11.2 | Extract `truncate_snippet()` utility          | `raw_snippet[:400]` + `"..."` pattern repeated 5+ times in `mcp_server.py` — make a shared utility with configurable max length                                | Low        | P2       | ✅     |
| 11.3 | Normalize snippet sizes to shared constant    | Inconsistent: 400 chars in `search_notes`, 300 in `get_subject`/`search_by_tags`/`get_related_notes` — define `SNIPPET_MAX_CHARS` constant                     | Low        | P3       | ✅     |
| 11.4 | Lazy-import heavy deps in `__init__.py`       | Currently loads ChromaDB and all deps at import time even if only `config` or `frontmatter` is needed                                                           | Low        | P3       | ✅     |
| 11.5 | Make `chroma_store` configurable at init      | Module-level `PersistentClient` created at import time — accept a path parameter so it can be reconfigured and tested without monkey-patching                   | Medium     | P3       | ⬜     |

---

## Phase 12 — Test Coverage

| #   | Task                                  | Description                                                                                                                                                  | Difficulty | Priority | Status |
| --- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | -------- | ------ |
| 12.1 | Tests for `keyword_search.py`        | Test BM25 index build, search, ensure_index, and edge cases (empty index, single doc)                                                                        | Medium     | P2       | ⬜     |
| 12.2 | Tests for `indexer.py`               | Test `chunk_text`, `_sanitize`, `_word_count`, `_extract_tags`, `_should_skip_by_mtime`, and the mtime map builder                                            | Medium     | P2       | ⬜     |
| 12.3 | Tests for `chroma_store.py`          | Test upsert, query, delete_by_path, get_by_path, get_all_documents, count — with a temp ChromaDB instance                                                   | Medium     | P2       | ⬜     |
| 12.4 | Tests for `llm_client.py`            | Test embed and chat functions with mocked HTTP responses (no live Ollama needed)                                                                             | Medium     | P2       | ⬜     |
| 12.5 | Tests for `mcp_server.py` tools      | Test search_notes, read_note, write_note, add_tags, create_backlink, get_related_notes, get_subject — with mocked deps                                        | High       | P2       | ⬜     |
| 12.6 | Tests for `config.py`                | Test env var loading, defaults, missing vars                                                                                                                 | Low        | P3       | ⬜     |
| 12.7 | Tests for `logger.py`                | Test log file creation, error formatting                                                                                                                     | Low        | P3       | ⬜     |

---

## Phase 13 — New Functionality

| #   | Task                                  | Description                                                                                                                                                  | Difficulty | Priority | Status |
| --- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | -------- | ------ |
| 13.1 | Implement `remove_tags` MCP tool     | Remove specific tags from a note's YAML frontmatter (inverse of `add_tags`)                                                                                  | Low        | P2       | ✅     |
| 13.2 | Implement `set_tags` MCP tool        | Replace all tags on a note with a given list (overwrite mode vs add mode)                                                                                    | Low        | P3       | ✅     |

---

## Summary

| Phase                                         | Total Tasks | P1     | P2     | P3     | Done   | Remaining |
| --------------------------------------------- | ----------- | ------ | ------ | ------ | ------ | --------- |
| Phase 1 — Foundation                          | 10          | 8      | 1      | 1      | 10     | 0         |
| Phase 2 — MCP Server                          | 13          | 8      | 4      | 1      | 13     | 0         |
| Phase 3 — LLM + Agent                         | 10          | 6      | 3      | 1      | 10     | 0         |
| Phase 4 — Polish                              | 12          | 6      | 3      | 3      | 12     | 0         |
| Phase 5 — Search Improvements                 | 10          | 3      | 4      | 3      | 10     | 0         |
| Phase 6 — Todo Management                     | 7           | 1      | 3      | 3      | 0      | 7         |
| Phase 7 — Advanced Features                   | 7           | 0      | 1      | 6      | 1      | 6         |
| Phase 8 — Performance optimization follow-ups | 4           | 0      | 3      | 1      | 3      | 1         |
| Phase 9 — Bug Fixes                           | 4           | 2      | 2      | 0      | 4      | 0         |
| Phase 10 — Security Hardening                 | 4           | 1      | 3      | 0      | 4      | 0         |
| Phase 11 — Tech Debt / Refactoring            | 5           | 0      | 2      | 3      | 4      | 1         |
| Phase 12 — Test Coverage                      | 7           | 0      | 5      | 2      | 0      | 7         |
| Phase 13 — New Functionality                  | 2           | 0      | 1      | 1      | 2      | 0         |
| **Total**                                     | **99**      | **35** | **35** | **29** | **71** | **28**    |
