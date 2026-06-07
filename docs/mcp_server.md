# MCP Server

The MCP server exposes Obsidian vault operations as 9 consolidated callable tools for any MCP-compatible agent (Goose, Claude, Cursor, etc.). Built with [FastMCP](https://github.com/jlowin/fastmcp).

Each tool (except `ask` and `tools`) takes an **`action`** parameter that selects the specific operation — reducing ~50 specialized tools down to 9 dispatch tools while keeping all functionality.

## Running

```bash
python -m obsidian_ai.mcp_server
```

The server runs on **stdio** transport — no HTTP port needed. Connect to it via an MCP client or agent.

## Path Normalization

All tools that accept a `path` parameter automatically normalize it: if the LLM passes an absolute path or one prefixed with the vault directory, the vault prefix is stripped to produce a vault-relative path. You can pass either format interchangeably.

---

## Tools (9 total)

Each tool (except `ask` and `tools`) accepts an `action` parameter to select the specific operation. All `path` parameters auto-normalize (absolute or vault-relative accepted).

---

### 1. `ask(query: str) -> str`

**Universal discovery tool.** Routes any vault query to the right internal capability via LLM intent detection.

Handles: semantic search, Q&A, entity lookup, relationship discovery, wiki-link traversal, topic summaries, tag suggestions, index stats, orphan detection, community discovery, shortest path, graph export, and more.

| Param | Type | Description |
|-------|------|-------------|
| `query` | `str` | Natural language request about your vault |

---

### 2. `notes(action, path, content, folder, title, tags, n, subject, sync, reindex_matches) -> str`

**Create, read, list, and organize notes.**

| Action | Description | Key Params |
|--------|-------------|------------|
| `read` | Fetch full note content | `path` |
| `write` | Create or overwrite a note | `path`, `content`, `sync` |
| `list` | List all note paths | — |
| `list_folder` | List entries in a folder (non-recursive) | `folder` |
| `search_by_tags` | Find notes with all given YAML tags (AND) | `tags`, `n` |
| `read_by_title` | Look up note by filename | `title`, `folder` |
| `add_note_to_subject` | Create note under Subjects/ with auto hub + backlinks | `subject`, `title`, `content`, `tags` |

---

### 3. `tags(action, path, tags, note_paths, query, top_k, sync) -> str`

**Manage YAML frontmatter tags on notes.**

| Action | Description | Key Params |
|--------|-------------|------------|
| `add` | Add tags without affecting existing | `path`, `tags`, `sync` |
| `remove` | Remove specific tags | `path`, `tags`, `sync` |
| `set` | Replace all tags on a note | `path`, `tags`, `sync` |
| `batch_add` | Add same tags to multiple notes | `note_paths`, `tags`, `sync` |
| `auto_suggest` | LLM suggests + applies tags for a query | `query`, `top_k`, `sync` |

---

### 4. `links(action, path, path_a, path_b, sync) -> str`

**Create and explore [[wiki-link]] connections.**

| Action | Description | Key Params |
|--------|-------------|------------|
| `create` | Bidirectional wiki-link between two notes | `path_a`, `path_b`, `sync` |
| `backlinks` | Notes linking TO a given note | `path` |
| `outgoing` | Notes a given note links TO | `path` |
| `broken` | Find wiki-links to non-existent notes | — |

---

### 5. `graph(action, path, start, end, max_depth, k, top_k, graph_weight, format) -> str`

**Explore vault wiki-link graph structure — communities, paths, orphans, and more.**

| Action | Description | Key Params |
|--------|-------------|------------|
| `communities` | Detect densely connected note groups | — |
| `community_of` | Identify a note's community + neighbors | `path`, `top_k` |
| `orphans` | Find notes with no wiki-links | — |
| `path` | Shortest wiki-link chain between two notes | `start`, `end` |
| `stats` | Graph summary statistics | — |
| `related` | Semantically similar + graph-connected notes | `path`, `k`, `graph_weight` |
| `traverse` | BFS outward from a seed note | `path`, `max_depth` |
| `export` | Export graph as JSON or DOT | `format` |

---

### 6. `entities(action, name, entity_name, entity_type, path, n, aliases, relations, reindex_matches, primary, secondary, new_type, date_from, date_to, relation_type, depth, use_graph, data, dedup_config, semantic, entity, graph, keyword) -> str`

**Search, create, and manage named entities (people, projects, concepts, etc.).**

| Action | Description | Key Params |
|--------|-------------|------------|
| `search` | Find notes mentioning an entity | `entity_name`, `entity_type`, `n`, `use_graph` |
| `note_entities` | List entities in a specific note | `path` |
| `list` | All entities sorted by mention count | `entity_type`, `n` |
| `aliases` | Canonical name, type, aliases for an entity | `name` |
| `timeline` | Chronological event timeline for an entity | `name`, `date_from`, `date_to` |
| `related` | Entities connected via relationship graph | `name`, `relation_type`, `depth` |
| `add` | Manually create an entity with aliases + relations | `name`, `entity_type`, `aliases`, `relations` |
| `merge` | Merge two entity records | `primary`, `secondary` |
| `change_type` | Correct entity classification type | `name`, `new_type` |
| `types` | List all available entity type labels | — |
| `weights_get` | Show current ranking weights | — |
| `weights_set` | Adjust ranking weights | `semantic`, `entity`, `graph`, `keyword` |
| `import` | Import entities + relations from another vault | `data`, `dedup_config` |

---

### 7. `todo(action, project, task, due, priority, tags, status, todo_id, note_paths, query, overdue, blocked, search, sync) -> str`

**Manage tasks, suggestions, and todo analysis.**

| Action | Description | Key Params |
|--------|-------------|------------|
| `list` | List todos with filters | `project`, `status`, `priority`, `overdue`, `blocked`, `search` |
| `add` | Create a new todo | `project`, `task`, `due`, `priority`, `tags` |
| `complete` | Mark a todo as completed | `todo_id` |
| `update` | Update one or more fields | `todo_id`, `task`, `due`, `priority`, `tags`, `project`, `status` |
| `delete` | Permanently delete a todo | `todo_id` |
| `stats` | Aggregated todo statistics | — |
| `suggest_priority` | LLM suggests priority for a task | `task` |
| `suggest_date` | LLM suggests due date for a task | `task` |
| `suggest_split` | LLM splits a large task into subtasks | `task` |
| `overdue_summary` | LLM summary of all overdue todos | — |
| `link` | Link a todo to notes via wiki-links | `todo_id`, `note_paths` |
| `ask` | Ask LLM about a specific todo or all todos | `todo_id`, `query` |

---

### 8. `admin(action, folder, subject, model_name, sync) -> str`

**System administration — health checks, re-indexing, model configuration.**

| Action | Description | Key Params |
|--------|-------------|------------|
| `health` | Verify LLM, ChromaDB, vault accessibility | — |
| `reindex` | Re-run full index (folder or entity-specific) | `folder`, `subject` |
| `stats` | Diagnostic index statistics | — |
| `switch_model` | Switch embedding model at runtime | `model_name` |
| `sync_todos` | Recalculate todo frontmatter stats | `sync` |

---

### 9. `tools() -> str`

**Tool discovery.** Lists all available MCP tools with their descriptions and parameter schemas. Use this at the start of a session if your client cannot see all tools at once.

**Parameters:** None

**Returns:** JSON array of tool objects, each with `name`, `description`, and `parameters`.

---

## Error Handling

All tools catch exceptions and log them to `logs/mcp_calls.log` with the full traceback. String-returning tools return an error message string on failure.

---

## Logging

Every tool call is logged to `logs/mcp_calls.log`:

```
2026-05-31 14:23:01 [INFO] obsidian_ai.mcp_server — ask — query=python n=5
2026-05-31 14:23:02 [INFO] obsidian_ai.mcp_server — notes — action=read path=Notes/topic.md
```

---

## Dependencies

- `fastmcp` — MCP server framework
- `pyyaml` — YAML frontmatter parsing
- `requests` — HTTP client for Obsidian API and Ollama
- All `obsidian_ai.*` modules (config, logger, frontmatter, obsidian_client, llm_client, chroma_store, indexer, pipelines, entity_store, graph_store, wiki_links, clustering, mcp_server)
