# MCP Server

The MCP server exposes Obsidian vault operations as callable tools for any MCP-compatible agent (Goose, Claude, Cursor, etc.). Built with [FastMCP](https://github.com/jlowin/fastmcp).

## Running

```bash
python -m obsidian_ai.mcp_server
```

The server runs on **stdio** transport — no HTTP port needed. Connect to it via an MCP client or agent.

## Path Normalization

All tools that accept a `path` parameter automatically normalize it: if the LLM passes an absolute path or one prefixed with the vault directory, the vault prefix is stripped to produce a vault-relative path. You can pass either format interchangeably.

---

## Tools (57 total)

### Health

#### `health_check()`

Check the health status of all backend services (Ollama, ChromaDB, Obsidian API). Useful before starting a long workflow to confirm everything is available.

**Parameters:** None

**Returns:** `str` — formatted status of each service (Ollama embed model, chat model, API reachability, index state).

---

### Search & Retrieval

#### `search_notes(query, n=5, tags=None, exclude_tags=None, folder=None, date_after=None, date_before=None, expand_query=False, keyword_weight=0.0, min_similarity=None, diversity_penalty=0.0, use_graph=False, graph_depth=1, graph_weight=0.2, use_entities=False, entity_types=None, group_by_note=False)`

Semantic search across all indexed notes with rich metadata filtering.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | *(required)* | Natural language search query |
| `n` | `int` | `5` | Number of results to return |
| `tags` | `list[str]` | `None` | Filter: notes must have ALL these YAML tags |
| `exclude_tags` | `list[str]` | `None` | Filter: exclude notes with ANY of these tags |
| `folder` | `str` | `None` | Filter: only notes under this vault-relative folder |
| `date_after` | `str` | `None` | Filter: ISO date string (e.g. `2024-06-01`) |
| `date_before` | `str` | `None` | Filter: ISO date string (e.g. `2024-12-31`) |
| `expand_query` | `bool` | `False` | Expand query with LLM-generated alternative phrasings |
| `keyword_weight` | `float` | `0.0` | BM25 keyword blend (0.0 = pure semantic, 1.0 = pure keyword) |
| `min_similarity` | `float` | `None` | Minimum similarity threshold (0–1) |
| `diversity_penalty` | `float` | `0.0` | Penalize notes already represented (0.0–1.0) |
| `use_graph` | `bool` | `False` | Expand results via wiki-link graph traversal |
| `graph_depth` | `int` | `1` | Max hops for graph traversal |
| `graph_weight` | `float` | `0.2` | Weight for graph proximity boost |
| `use_entities` | `bool` | `False` | Also search entity index for matching entities |
| `entity_types` | `list[str]` | `None` | Filter entity types when `use_entities=True` |
| `group_by_note` | `bool` | `False` | Group results by note path |

**Returns:** `list[dict]`

```json
[
  {"path": "Notes/topic.md", "title": "topic", "matched_chunk_idx": 0, "similarity_score": 0.89, "snippet": "..."},
  {"path": "Notes/related.md", "title": "related", "matched_chunk_idx": 1, "similarity_score": 0.72, "snippet": "..."}
]
```

---

#### `batch_search(queries, n=5, tags=None, exclude_tags=None, folder=None, keyword_weight=0.0, min_similarity=None)`

Run multiple searches in one call. Each query is searched independently.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `queries` | `list[str]` | *(required)* | List of search queries |
| `n` | `int` | `5` | Results per query |
| `tags` | `list[str]` | `None` | Filter by tags (all queries) |
| `exclude_tags` | `list[str]` | `None` | Exclude by tags (all queries) |
| `folder` | `str` | `None` | Filter by folder (all queries) |
| `keyword_weight` | `float` | `0.0` | BM25 keyword blend (all queries) |
| `min_similarity` | `float` | `None` | Minimum similarity (all queries) |

**Returns:** `dict[str, list[dict]]` — `{query: results}`

---

#### `retrieve_notes(query, top_k=5, use_graph=False, graph_depth=1, graph_weight=0.2, use_entities=False, entity_types=None, keyword_weight=0.0, min_similarity=None, expand_query=False)`

Multi-strategy retrieval pipeline combining semantic search, entity lookup, and wiki-link graph traversal into a single unified result set.

**Parameters:** Similar to `search_notes`, without `diversity_penalty`, `tags`/`exclude_tags`/`folder`/`date_*` filters.

**Returns:** `list[dict]` — note-level results with a `matched_by` field indicating which strategy found each result.

---

#### `find_duplicate_notes(threshold=0.9, n=20)`

Find near-duplicate notes via embedding similarity. Compares first-chunk embeddings using cosine distance.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `threshold` | `float` | `0.9` | Similarity threshold (0.0–1.0) |
| `n` | `int` | `20` | Max pairs to return |

**Returns:** `list[dict]` — `[{path_a, path_b, similarity}, ...]`

---

#### `search_by_tags(tags, n=10)`

Find notes that have ALL of the given YAML frontmatter tags.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `tags` | `list[str]` | *(required)* | Tags to match (all must be present) |
| `n` | `int` | `10` | Max results |

**Returns:** `list[dict]` — `[{path, title, tags, snippet}, ...]`

---

#### `get_subject(subject, top_k=10, keyword_weight=0.3, group_by_note=False)`

Get notes related to a free-form subject. Uses LLM to expand the subject with related terms, then performs hybrid search (semantic + BM25).

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `subject` | `str` | *(required)* | Free-form subject or topic |
| `top_k` | `int` | `10` | Max results |
| `keyword_weight` | `float` | `0.3` | BM25 keyword blend |
| `group_by_note` | `bool` | `False` | Group passages by note |

---

### Read & Write

#### `read_note(path)`

Fetch the full content of a note.

| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative path (e.g., `Notes/topic.md`) |

**Returns:** `str` — note content as Markdown.

---

#### `write_note(path, content)`

Create or overwrite a note.

| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative path |
| `content` | `str` | Full note content (Markdown) |

**Returns:** `str` — confirmation message.

---

#### `list_all_notes()`

List all note paths in the vault. **Parameters:** None

**Returns:** `list[str]` — all `.md` file paths (excludes backup folders, `.excalidraw.md`, etc.).

---

#### `list_folder(folder_path)`

List entries directly inside a specific folder (non-recursive — no subdirectory traversal). Returns both `.md` files and subdirectory names (with trailing `/`).

| Param | Type | Description |
|-------|------|-------------|
| `folder_path` | `str` | Vault-relative folder path |

**Returns:** `list[str]`

---

#### `list_folder_deep(folder_path)`

List all note paths within a specific folder (recursive — traverses all subdirectories).

| Param | Type | Description |
|-------|------|-------------|
| `folder_path` | `str` | Vault-relative folder path |

**Returns:** `list[str]`

---

#### `read_note_by_title(title, folder_path="")`

Look up a note by its title (filename without `.md` extension). Optionally scope to a folder to disambiguate duplicate titles.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | `str` | *(required)* | Note title without `.md` extension |
| `folder_path` | `str` | `""` | Vault-relative folder to narrow search |

**Returns:** `str`

**Duplicate title handling:**
- Single match → returns note content directly
- Multiple matches, no folder → all matches with `─── path ───` headers
- Multiple matches with folder → narrows to notes inside that folder
- No match → `Error: No note found with title: ...`

---

### Tag Management

#### `add_tags(path, tags)`

Add tags to a note's YAML frontmatter. Creates frontmatter if absent.

| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative path |
| `tags` | `list[str]` | Tags to add |

**Returns:** `str` — confirmation with updated tag list.

---

#### `remove_tags(path, tags)`

Remove specific tags from a note's YAML frontmatter.

| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative path |
| `tags` | `list[str]` | Tags to remove |

**Returns:** `str` — confirmation message.

---

#### `set_tags(path, tags)`

Replace all tags on a note with the given list.

| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative path |
| `tags` | `list[str]` | New tag list |

**Returns:** `str` — confirmation message.

---

#### `batch_tag_notes(note_paths, tags)`

Add tags to multiple notes at once.

| Param | Type | Description |
|-------|------|-------------|
| `note_paths` | `list[str]` | Paths to notes to tag |
| `tags` | `list[str]` | Tags to add to each note |

**Returns:** `dict[str, str]` — `{path: result_message}`

---

### Graph / Wiki-Links

#### `create_backlink(path_a, path_b)`

Create mutual `[[backlinks]]` between two notes.

| Param | Type | Description |
|-------|------|-------------|
| `path_a` | `str` | First note path |
| `path_b` | `str` | Second note path |

**Returns:** `str` — confirmation message.

---

#### `get_backlinks(path)`

Return all notes linking TO the given note (incoming wiki-link edges).

| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative note path |

**Returns:** `list[dict]` — `[{path, title}, ...]`

---

#### `get_linked_notes(path)`

Return all notes the given note links TO (outgoing wiki-link edges).

| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative note path |

**Returns:** `list[dict]` — `[{path, title}, ...]`

---

#### `get_broken_links()`

Find wiki-links across all notes that don't resolve to any existing note. **Parameters:** None

**Returns:** `list[dict]` — `[{source_path, link_target}, ...]`

---

#### `get_graph_stats()`

Return graph statistics. **Parameters:** None

**Returns:** `dict` — `{nodes, edges, avg_degree, isolated_count, isolated (list), hubs (top 5)}`

---

#### `get_communities()`

Detect communities in the wiki-link graph using label propagation. **Parameters:** None

**Returns:** `dict[str, list[str]]` — `{community_id: [note_paths]}`

---

#### `multi_hop_traversal(path, max_depth=2)`

Perform BFS graph traversal from a seed note up to N hops.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | *(required)* | Seed note path |
| `max_depth` | `int` | `2` | Max hops |

**Returns:** `list[dict]` — `[{path, title, depth, trace}, ...]`

---

#### `related_notes(path, k=10, graph_weight=0.3)`

Find notes related to a given note using both semantic similarity and graph proximity. Combines embedding-based search with wiki-link graph traversal.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | *(required)* | Source note path |
| `k` | `int` | `10` | Max results |
| `graph_weight` | `float` | `0.3` | Graph proximity weight (0.0 = pure semantic, 1.0 = pure graph) |

**Returns:** `list[dict]`

---

#### `export_graph(format="json")`

Export the wiki-link graph for external visualization.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | `str` | `"json"` | `"dot"` for Graphviz DOT, `"json"` for JSON |

**Returns:** `str` — DOT or JSON string.

---

#### `get_orphan_notes()`

Find notes with no incoming or outgoing wiki-links (orphans). **Parameters:** None

**Returns:** `list[str]` — note paths.

---

### LLM-Powered

#### `ask_vault(question, top_k=3, use_graph=False, graph_depth=1, use_entities=False, entity_types=None, keyword_weight=0.0, expand_query=False)`

Ask a natural language question about your vault. Searches relevant notes and uses the LLM to generate an answer.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | `str` | *(required)* | Natural language question |
| `top_k` | `int` | `3` | Notes to retrieve for context |
| `use_graph` | `bool` | `False` | Expand via wiki-link graph |
| `graph_depth` | `int` | `1` | Max graph hops |
| `use_entities` | `bool` | `False` | Expand via entity lookup |
| `entity_types` | `list[str]` | `None` | Filter entity types |
| `keyword_weight` | `float` | `0.0` | BM25 keyword blend |
| `expand_query` | `bool` | `False` | LLM query expansion |

**Returns:** `str` — LLM-generated answer based on vault content.

---

#### `ask_agent(query)`

Route a query to the best tool automatically using an LLM agent. The agent decides whether to search, read notes, traverse the graph, etc.

| Param | Type | Description |
|-------|------|-------------|
| `query` | `str` | Free-form query or request |

**Returns:** `str` — tool result or answer.

---

#### `summarize_topic(topic, top_k=5, use_graph=True, graph_depth=1, graph_weight=0.2, use_entities=True, entity_types=None, keyword_weight=0.0, expand_query=False)`

Search all notes related to a topic and return an LLM-generated consolidated summary.

**Parameters:** Similar to `ask_vault` but defaults to `use_graph=True`, `use_entities=True`.

**Returns:** `str` — LLM-generated summary.

---

#### `tag_notes(query, top_k=5)`

Search notes matching a query and auto-suggest tags using the LLM.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | *(required)* | Search query |
| `top_k` | `int` | `5` | Number of notes to tag |

**Returns:** `str` — confirmation with tag map.

---

### Entity System

#### `search_entities(entity_name, entity_type=None, n=10, use_graph=False)`

Find notes mentioning a specific entity. Uses entity inverted index with ChromaDB fallback.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `entity_name` | `str` | *(required)* | Entity name to search (case-insensitive) |
| `entity_type` | `str` | `None` | Filter by type (Person, Technology, Project, etc.) |
| `n` | `int` | `10` | Max results |
| `use_graph` | `bool` | `False` | Expand via 1-hop graph traversal |

**Returns:** `list[dict]` — `[{path, title, entity_name, entity_type, snippet, confidence}, ...]`

---

#### `get_note_entities(path)`

Return all entities found in a specific note during indexing.

| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative note path |

**Returns:** `list[dict]` — `[{entity_name, entity_type, confidence}, ...]`

---

#### `get_entity_types()`

List all entity type labels present in the index. **Parameters:** None

**Returns:** `list[str]` — sorted entity types (e.g. `["Concept", "Event", "Hardware", ...]`)

---

### Index Management

#### `sync_index()`

Re-run the full indexing pipeline. Clears the embedding cache and BM25 index. **Parameters:** None

**Returns:** `str` — confirmation message.

---

#### `health_check()`

*(See Health section above)*

---

#### `get_index_stats()`

Return index statistics. **Parameters:** None

**Returns:** `str` — formatted stats with chunk count, unique notes, config info, and cache stats.

---

#### `switch_embedding_model(model_name)`

Switch the embedding model at runtime. Verifies the model exists in Ollama, clears all caches, and re-indexes the vault.

| Param | Type | Description |
|-------|------|-------------|
| `model_name` | `str` | Ollama model name (e.g. `"nomic-embed-text"`) |

**Returns:** `str` — confirmation with details.

---

### Todo Management

#### `get_todos(project="", status="", overdue=False, blocked=False, search="")`

List todos from `todos.md` with optional filters.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `project` | `str` | `""` | Filter by project name (case-sensitive) |
| `status` | `str` | `""` | `"pending"` or `"completed"` |
| `overdue` | `bool` | `False` | Only overdue pending todos |
| `blocked` | `bool` | `False` | Only blocked todos |
| `search` | `str` | `""` | Free-text search |

**Returns:** `list[dict]`

---

#### `add_todo(project, task, due="", priority="", tags=None)`

Add a new todo task to a project. Creates the project if it doesn't exist.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `project` | `str` | *(required)* | Project name |
| `task` | `str` | *(required)* | Task description |
| `due` | `str` | `""` | Due date (YYYY-MM-DD) |
| `priority` | `str` | `""` | `"high"`, `"medium"`, or `"low"` |
| `tags` | `list[str]` | `None` | Optional tags |

**Returns:** `dict` — `{success, todo_id, ...}`

---

#### `complete_todo(todo_id)`

Mark a todo as completed by its id.

| Param | Type | Description |
|-------|------|-------------|
| `todo_id` | `str` | Todo ID (returned by `get_todos` or `add_todo`) |

**Returns:** `dict` — `{success, ...}`

---

#### `update_todo(todo_id, task="", due="", priority="", tags=None, project="", status="")`

Update one or more fields of an existing todo. Only provided fields are changed.

**Returns:** `dict` — `{success, ...}`

---

#### `delete_todo(todo_id)`

Delete a todo by its id.

**Returns:** `dict` — `{success, ...}`

---

#### `sync_todos()`

Recalculate todo counts in the `todos.md` frontmatter and rewrite the file. **Parameters:** None

**Returns:** `dict` — `{success, ...}`

---

#### `get_todo_stats()`

Return aggregated statistics about all todos. **Parameters:** None

**Returns:** `dict` — `{total, completed, pending, overdue, per_project, per_priority, ...}`

---

#### `ensure_todo_file()`

Create a default `todos.md` file in the vault if it doesn't exist. **Parameters:** None

**Returns:** `str` — confirmation message.

---

## Error Handling

All tools catch exceptions and log them to `logs/mcp_calls.log` with the full traceback. List-returning tools return an empty list `[]` on failure. String-returning tools return an error message string.

---

## Logging

Every tool call is logged to `logs/mcp_calls.log`:

```
2026-05-31 14:23:01 [INFO] obsidian_ai.mcp_server — search_notes — query=python, n=5
2026-05-31 14:23:02 [INFO] obsidian_ai.mcp_server — read_note — path=Notes/topic.md
```

---

## Dependencies

- `fastmcp` — MCP server framework
- `pyyaml` — YAML frontmatter parsing
- `requests` — HTTP client for Obsidian API and Ollama
- All `obsidian_ai.*` modules (config, logger, frontmatter, obsidian_client, llm_client, chroma_store, indexer, pipelines, entity_store, graph_store, wiki_links, mcp_server)
