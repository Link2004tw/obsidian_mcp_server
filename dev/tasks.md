# Tasks

All tasks are organized by phase. Difficulty: `Low` / `Medium` / `High`. Priority: `P1` (must) / `P2` (should) / `P3` (nice to have).

---

## Phase 1 — Vault API + Embeddings Foundation

| # | Task | Description | Difficulty | Priority | Status |
|---|---|---|---|---|---|
| 1.1 | Install Obsidian Local REST API plugin | Enable the community plugin, grab the API key and confirm it responds on port 27123 | Low | P1 | ✅ |
| 1.2 | Write `obsidian_client.py` | Thin wrapper around the REST API: `list_notes()`, `get_note(path)`, `put_note(path, content)` | Low | P1 | ✅ |
| 1.3 | Pull `nomic-embed-text` model | Run `ollama pull nomic-embed-text` and verify it responds via `POST /api/embeddings` | Low | P1 | ✅ |
| 1.4 | Write `llm_client.py` — embed function | Function that takes a string and returns a 768-dim vector via Ollama | Low | P1 | ✅ |
| 1.5 | Write `chroma_store.py` | Initialize ChromaDB client and collection; implement `upsert(id, embedding, metadata)` and `query(embedding, n)` | Medium | P1 | ✅ |
| 1.6 | Write `indexer.py` — one-shot mode | Read all notes via Obsidian client, embed each, upsert to ChromaDB with `path`, `title`, `mtime` | Medium | P1 | ✅ |
| 1.7 | Write `config.py` | Centralize vault path, Obsidian API key/port, Ollama base URL, model names, ChromaDB path | Low | P1 | ✅ |
| 1.8 | Test semantic search manually | Run a Python snippet: embed a query string, call `chroma_store.query()`, print top-5 results | Low | P1 | ✅ |
| 1.9 | Handle empty and very short notes | Skip notes under ~20 tokens to avoid noisy embeddings | Low | P2 | ✅ |
| 1.10 | Store note word count in metadata | Useful later for context-stuffing decisions in the LLM layer | Low | P3 | ✅ |

---

## Phase 2 — MCP Server

| # | Task | Description | Difficulty | Priority | Status |
|---|---|---|---|---|---|
| 2.1 | Install and configure `fastmcp` | `pip install fastmcp`, scaffold `mcp_server.py`, verify server starts | Low | P1 | ✅ |
| 2.2 | Implement `search_notes(query)` tool | Embed the query, search ChromaDB, return top-k note paths + snippets | Medium | P1 | ✅ |
| 2.3 | Implement `read_note(path)` tool | Fetch full note content from Obsidian REST API by path | Low | P1 | ✅ |
| 2.4 | Implement `write_note(path, content)` tool | Create or overwrite a note via Obsidian REST API | Low | P1 | ✅ |
| 2.5 | Implement `list_notes()` tool | Return a flat list of all note paths in the vault | Low | P1 | ✅ |
| 2.6 | Implement `add_tags(path, tags)` tool | Parse YAML frontmatter, merge new tags, write note back; create frontmatter if absent | High | P1 | ✅ |
| 2.7 | Implement `create_backlink(path_a, path_b)` tool | Append `[[note_b]]` to note A and `[[note_a]]` to note B if not already present | Medium | P1 | ✅ |
| 2.8 | Implement `sync_index()` tool | Re-run the full indexer pipeline on demand; returns count of notes indexed | Medium | P2 | ✅ |
| 2.9 | Test all tools via MCP inspector | Call each tool directly and verify correct input/output before any agent is involved | Low | P1 | ✅ |
| 2.10 | Add error handling to all tools | Return structured error messages instead of raw exceptions for missing notes, API failures | Medium | P2 | ✅ |
| 2.11 | Log all tool calls to a file | Append timestamped tool name + args to `mcp_calls.log` for debugging | Low | P3 | ✅ |

---

## Phase 3 — LLM + Agent Integration

| # | Task | Description | Difficulty | Priority | Status |
|---|---|---|---|---|---|
| 3.1 | Write `llm_client.py` — chat function | Function that sends a messages array to Qwen3:8b via Ollama and returns the text response | Low | P1 | ✅ |
| 3.2 | Implement query mode pipeline | Embed query → `search_notes` → fetch top-3 full contents → stuff into LLM prompt → return answer | Medium | P1 | ✅ |
| 3.3 | Implement action mode pipeline | `search_notes` → for each result, LLM suggests tags → call `add_tags` per note | High | P1 | ✅ |
| 3.4 | Configure Goose to use the MCP server | Add `mcp_server.py` to Goose's MCP config, verify Goose can list and call tools | Low | P1 | ✅ |
| 3.5 | Test end-to-end query flow in Goose | "Find notes related to X" → confirms correct notes are returned with context | Medium | P1 | ✅ |
| 3.6 | Test end-to-end action flow in Goose | "Tag all notes about X" → confirms tags appear in Obsidian frontmatter | Medium | P1 | ✅ |
| 3.7 | Test backlink creation flow | "Link all notes about X to each other" → confirms `[[links]]` appear in correct notes | Medium | P2 | ✅ |
| 3.8 | Tune LLM prompt for tagging | Iterate on the system prompt until suggested tags are concise and consistent in format | Medium | P2 | ✅ |
| 3.9 | Add context length guard | Truncate note content before stuffing into LLM context if total tokens exceed a safe limit | Medium | P2 | ✅ |
| 3.10 | Test with Qwen3 thinking mode off | Add `/no_think` prefix to prompts where speed matters more than reasoning depth | Low | P3 | ✅ |

---

## Phase 4 — Incremental Indexing + Polish

| # | Task | Description | Difficulty | Priority | Status |
|---|---|---|---|---|---|
| 4.1 | Add `watchdog` file watcher to `indexer.py` | Monitor vault directory; on file save, re-embed only the changed note and upsert to ChromaDB | Medium | P1 | ✅ |
| 4.2 | Handle note deletion | On file delete event, remove the corresponding ChromaDB entry by ID | Medium | P1 | ✅ |
| 4.3 | Handle note rename | Delete old ChromaDB entry, re-index under new path | Medium | P1 | ✅ |
| 4.4 | Add `--watch` flag to `indexer.py` | `python indexer.py --watch` starts the watcher; without flag does a one-shot full index | Low | P1 | ✅ |
| 4.5 | Skip re-embedding if `mtime` unchanged | Compare stored `mtime` in ChromaDB metadata before re-embedding to avoid unnecessary work | Low | P2 | ✅ |
| 4.6 | Write `README.md` | Setup instructions, usage examples, config reference, Goose integration guide | Low | P2 | ✅ |
| 4.7 | Write a basic CLI wrapper | `cli.py` with commands: `index`, `watch`, `search <query>`, `tag <query>` using `argparse` | Medium | P3 | ✅ |
| 4.8 | Add index stats command | Print total notes indexed, ChromaDB collection size, last index time | Low | P3 | ✅ |
| 4.9 | Test full cold-start setup | Clone repo on a fresh env, follow README, confirm system works end-to-end | Low | P2 | ✅ |

---

## Phase 5 — Search Improvements

| # | Task | Description | Difficulty | Priority | Status |
|---|---|---|---|---|---|
| 5.1 | Hybrid search (BM25 + semantic) | Implement BM25/TF-IDF keyword fallback when semantic results are low-confidence; blend scores with configurable `keyword_weight` and `min_similarity` | High | P2 | ⬜ |
| 5.2 | Metadata filtering & faceting | Add optional filters to `search_notes`: `tags`, `folder`, `exclude_tags`, `date_after`, `date_before` for scoped searches | Medium | P2 | ⬜ |
| 5.3 | Passage-level search returns | Return matching chunk snippet with context window instead of full note path; include `snippet`, `matched_chunk_idx`, `similarity_score` | Medium | P2 | ⬜ |
| 5.4 | Query expansion (LLM) | Use Qwen to expand queries with synonyms before embedding (e.g., "PID tuning" → "proportional integral derivative control, feedback loop") | Medium | P3 | ⬜ |
| 5.5 | Relevance threshold & diversity | Add `min_similarity` filter; implement `diversity_penalty` to penalize results too similar to already-selected ones | Low | P3 | ⬜ |
| 5.6 | Search caching | LRU cache (`max_size=100`) for query embeddings + results; common queries become ~1ms after first run | Low | P3 | ⬜ |

## Phase 6 — Todo Management System

| # | Task | Description | Difficulty | Priority | Status |
|---|---|---|---|---|---|
| 6.1 | Todo file structure | Design `todos.md` format: YAML frontmatter (last_synced, counts), Markdown headers for projects, `[ ]`/`[x]` checkboxes, inline metadata in parentheses `(due:, priority:, tags:)` | Medium | P2 | ⬜ |
| 6.2 | Auto-create todo file | `ensure_todos_file_exists()` on first index/MCP startup or when file is deleted; create default templated `todos.md` | Low | P2 | ⬜ |
| 6.3 | Core todo MCP tools | Implement `get_todos`, `add_todo`, `complete_todo`, `update_todo`, `delete_todo`, `sync_todos` — full CRUD against `todos.md` | High | P1 | ⬜ |
| 6.4 | Smart todo queries | `get_overdue_todos`, `get_blocked_todos`, `get_todos_by_project`, `search_todos` (semantic), `get_todo_stats` | Medium | P2 | ⬜ |
| 6.5 | LLM-powered todo features | Natural language todo creation (`add_todo_from_natural_language`), `suggest_task_priority`, `suggest_due_date`, `suggest_task_splitting`, `suggest_task_dependencies` | High | P3 | ⬜ |
| 6.6 | Todo reporting & metrics | `get_todos_by_priority`, `get_burndown_chart(project, days)`, `get_overdue_summary`, `estimate_completion_date` | Medium | P3 | ⬜ |
| 6.7 | Todo integration with vault | `get_todos_for_note`, `get_notes_for_todo`, `link_todo_to_notes`, `ask_vault_about_todo`, `ask_vault_about_todos` | Medium | P3 | ⬜ |

## Phase 7 — Advanced Features

| # | Task | Description | Difficulty | Priority | Status |
|---|---|---|---|---|---|
| 7.1 | Incremental indexing via file hashes | Track file content hashes in JSON; only re-embed changed notes (delta updates) instead of re-indexing entire vault | Medium | P2 | ⬜ |
| 7.2 | Entity extraction | Extract named entities (people, projects, hardware, dates, concepts) using LLM during indexing; store in ChromaDB metadata for `search_by_entity` | High | P3 | ⬜ |
| 7.3 | Note summaries | Pre-generate 1-2 sentence summaries during indexing; `ask_vault` uses summary first, loads full content only if needed | High | P3 | ⬜ |
| 7.4 | Embedding model switching | Make embedding model configurable at runtime; implement `switch_embedding_model(new_model)` that re-indexes with new model | Medium | P3 | ⬜ |
| 7.5 | Performance metrics | `get_index_stats()` exposing total_notes, total_chunks, index_size_mb, avg_query_latency_ms, cache_hit_rate, last_sync, embedding_model | Medium | P3 | ⬜ |
| 7.6 | Batch operations | `batch_search(queries)` returning `dict[query, results]`; `batch_tag_notes(note_paths, tags)` for bulk tagging | Low | P3 | ⬜ |
| 7.7 | Semantic deduplication | `find_duplicate_notes(threshold)` using embedding similarity to detect near-duplicate note pairs | Low | P3 | ⬜ |

---

## Summary

| Phase | Total Tasks | P1 | P2 | P3 | Done | Remaining |
|---|---|---|---|---|---|---|
| Phase 1 — Foundation | 10 | 8 | 1 | 1 | 10 | 0 |
| Phase 2 — MCP Server | 11 | 7 | 3 | 1 | 11 | 0 |
| Phase 3 — LLM + Agent | 10 | 6 | 3 | 1 | 10 | 0 |
| Phase 4 — Polish | 9 | 4 | 3 | 2 | 9 | 0 |
| Phase 5 — Search Improvements | 6 | 0 | 3 | 3 | 0 | 6 |
| Phase 6 — Todo Management | 7 | 1 | 3 | 3 | 0 | 7 |
| Phase 7 — Advanced Features | 7 | 0 | 1 | 6 | 0 | 7 |
| **Total** | **60** | **26** | **17** | **17** | **40** | **20** |
