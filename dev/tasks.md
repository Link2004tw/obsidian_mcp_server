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

## Summary

| Phase | Total Tasks | P1 | P2 | P3 | Done | Remaining |
|---|---|---|---|---|---|---|
| Phase 1 — Foundation | 10 | 8 | 1 | 1 | 10 | 0 |
| Phase 2 — MCP Server | 11 | 7 | 3 | 1 | 11 | 0 |
| Phase 3 — LLM + Agent | 10 | 6 | 3 | 1 | 10 | 0 |
| Phase 4 — Polish | 9 | 4 | 3 | 2 | 9 | 0 |
| **Total** | **40** | **25** | **10** | **5** | **40** | **0** |
