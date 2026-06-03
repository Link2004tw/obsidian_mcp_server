# MCP Server

The MCP server exposes Obsidian vault operations as callable tools for any MCP-compatible agent (Goose, Claude, Cursor, etc.). Built with [FastMCP](https://github.com/jlowin/fastmcp).

## Running

```bash
python -m obsidian_ai.mcp_server
```

The server runs on **stdio** transport — no HTTP port needed. Connect to it via an MCP client or agent.

## Tools

### `search_notes(query, n=5)`

Semantic search across all indexed notes.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | *(required)* | Natural language search query |
| `n` | `int` | `5` | Number of results to return |

**Returns:** `list[dict]` — deduplicated results with paths and distances.

```json
[
  {"path": "Notes/topic.md", "title": "topic", "chunk": 0, "distance": 0.42},
  {"path": "Notes/related.md", "title": "related", "chunk": 1, "distance": 0.51}
]
```

**How it works:**
1. Embeds the query via Ollama (`nomic-embed-text`)
2. Searches ChromaDB for top-k closest chunks
3. Deduplicates by note path (multiple chunks from same note → one result)

---

### `read_note(path)`

Fetch the full content of a note.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative path (e.g., `Notes/topic.md`) |

**Returns:** `str` — note content as Markdown.

---

### `write_note(path, content)`

Create or overwrite a note.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative path |
| `content` | `str` | Full note content (Markdown) |

**Returns:** `str` — confirmation message.

---

### `list_all_notes()`

List all note paths in the vault.

**Parameters:** None

**Returns:** `list[str]` — all `.md` file paths (excludes backup folders, `.excalidraw.md`, etc.).

---

### `list_folder(folder_path)`

List entries directly inside a specific folder (non-recursive — does not descend into subdirectories).
Returns both `.md` files and subdirectory names (with trailing `/`).

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `folder_path` | `str` | Vault-relative folder path (e.g., `Projects`) |

**Returns:** `list[str]` — entries in the folder (`.md` files and subdirectories).

---

### `list_folder_deep(folder_path)`

List all note paths within a specific folder (recursive — traverses all subdirectories).

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `folder_path` | `str` | Vault-relative folder path (e.g., `Projects`) |

**Returns:** `list[str]` — all `.md` file paths under the folder and its subdirectories.

---

### `read_note_by_title(title, folder_path=...)`

Look up a note by its title (filename without extension) and return its full content. Optionally scope to a folder to disambiguate duplicate titles.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | `str` | *(required)* | Note title without `.md` extension (e.g., `README`) |
| `folder_path` | `str` | `""` | Vault-relative folder path to narrow the search (e.g., `Projects`) |

**Returns:** `str` — note content as Markdown.

**Example:**
```
read_note_by_title("README")
# # Welcome...

read_note_by_title("README", folder_path="Projects")
# # Project Overview...
```

**Duplicate title handling:**
- **Single match** — returns the note content directly
- **Multiple matches, no folder** — returns all matching notes with `─── path ───` headers
- **Multiple matches with folder** — narrows to notes inside the given folder; returns just that one, or all inside the folder if still multiple
- **No match** — returns `Error: No note found with title: ...`

---

### `add_tags(path, tags)`

Add tags to a note's YAML frontmatter. Creates frontmatter if the note doesn't have any.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative path |
| `tags` | `list[str]` | Tags to add (e.g., `["python", "ml"]`) |

**Returns:** `str` — confirmation with updated tag list.

**Example:**
```
add_tags("Notes/topic.md", ["python", "machine-learning"])
# "Tags added to Notes/topic.md: ['python', 'machine-learning']"
```

**Behavior:**
- If note has no frontmatter → creates `---\ntags:\n- tag1\n---\n`
- If note has frontmatter but no `tags` field → adds `tags` field
- If note already has tags → appends new ones (no duplicates)
- If `tags` is a string in existing frontmatter → converts to list

---

### `create_backlink(path_a, path_b)`

Create mutual `[[backlinks]]` between two notes.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `path_a` | `str` | First note path |
| `path_b` | `str` | Second note path |

**Returns:** `str` — confirmation message.

**Behavior:**
- Extracts note names from paths (without `.md` extension)
- Appends `[[note_b]]` to note A if not already present
- Appends `[[note_a]]` to note B if not already present
- Links are added at the end of the note

---

### `sync_index()`

Re-run the full indexing pipeline on demand.

**Parameters:** None

**Returns:** `str` — confirmation message. Check `indexer.log` for details.

**What it does:**
1. Fetches all notes from Obsidian
2. Chunks, embeds, and stores in ChromaDB
3. Replaces any existing index data

---

### `ask_vault(question, top_k=3)`

Ask a natural language question about your vault. Searches relevant notes and uses the LLM to generate an answer.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | `str` | *(required)* | Natural language question |
| `top_k` | `int` | `3` | Number of notes to retrieve for context |

**Returns:** `str` — LLM-generated answer based on vault content.

**How it works:**
1. Embeds the question via Ollama
2. Searches ChromaDB for top-k relevant notes
3. Fetches full content of each note (truncated to 3000 words max)
4. Sends context + question to Qwen3:8b via Ollama
5. Returns the LLM's answer

**Example:**
```
ask_vault("What are my notes about machine learning?")
# "Your vault contains notes on neural networks, decision trees, and..."
```

---

### `tag_notes(query, top_k=5)`

Search notes matching a query and auto-suggest tags using the LLM.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | *(required)* | Search query to find relevant notes |
| `top_k` | `int` | `5` | Number of notes to tag |

**Returns:** `str` — confirmation with tag map.

**How it works:**
1. Embeds the query, finds top-k matching notes
2. Fetches full content of each note
3. Sends to LLM with instructions to suggest tags as JSON
4. Parses LLM response and applies tags via `add_tags`

**Example:**
```
tag_notes("machine learning")
# "Tagged 3 notes: {'Notes/nn.md': ['neural-network', 'deep-learning'], ...}"
```

---

## Error Handling

All tools catch exceptions and log them to `logs/mcp_calls.log` with the full traceback. List-returning tools (`search_notes`, `list_all_notes`, `list_folder`) return an empty list `[]` on failure to maintain type consistency. String-returning tools return an error message string.

---

## Logging

Every tool call is logged to `logs/mcp_calls.log`:

```
2026-05-31 14:23:01 [INFO] obsidian_ai.mcp_server — search_notes — query=python, n=5
2026-05-31 14:23:02 [INFO] obsidian_ai.mcp_server — read_note — path=Notes/topic.md
```

---

## Agent Configuration

Add this MCP server to any compatible agent. The server uses **stdio** transport.

### Common Config (all agents)

```json
{
  "mcpServers": {
    "obsidian-ai": {
      "command": "python",
      "args": ["-m", "obsidian_ai.mcp_server"],
      "cwd": "/path/to/obsidian-ai"
    }
  }
}
```

> **Note:** Replace `cwd` with your actual project path.

### Goose

Add to `~/.config/goose/profiles.yaml` or use the UI:

```json
{
  "mcpServers": {
    "obsidian-ai": {
      "command": "python",
      "args": ["-m", "obsidian_ai.mcp_server"],
      "cwd": "/path/to/obsidian-ai"
    }
  }
}
```

### Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "obsidian-ai": {
      "command": "python",
      "args": ["-m", "obsidian_ai.mcp_server"],
      "cwd": "/path/to/obsidian-ai"
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project root or global settings:

```json
{
  "mcpServers": {
    "obsidian-ai": {
      "command": "python",
      "args": ["-m", "obsidian_ai.mcp_server"],
      "cwd": "/path/to/obsidian-ai"
    }
  }
}
```

### Windsurf (Codeium)

Add to `~/.windsurf/mcp.json`:

```json
{
  "mcpServers": {
    "obsidian-ai": {
      "command": "python",
      "args": ["-m", "obsidian_ai.mcp_server"],
      "cwd": "/path/to/obsidian-ai"
    }
  }
}
```

### Cline (VS Code)

Add to VS Code settings or `.vscode/mcp.json`:

```json
{
  "mcpServers": {
    "obsidian-ai": {
      "command": "python",
      "args": ["-m", "obsidian_ai.mcp_server"],
      "cwd": "/path/to/obsidian-ai"
    }
  }
}
```

### opencode

Add to `opencode.json` in your project root:

```json
{
  "mcpServers": {
    "obsidian-ai": {
      "command": "python",
      "args": ["-m", "obsidian_ai.mcp_server"],
      "cwd": "/path/to/obsidian-ai"
    }
  }
}
```

### ChatGPT Desktop (OpenAI)

Add to `~/.chatgpt/mcp_servers.json`:

```json
{
  "obsidian-ai": {
    "command": "python",
    "args": ["-m", "obsidian_ai.mcp_server"],
    "cwd": "/path/to/obsidian-ai"
  }
}
```

### LibreChat

Add to `librechat.yaml` under `mcpServers`:

```yaml
mcpServers:
  obsidian-ai:
    command: python
    args:
      - "-m"
      - "obsidian_ai.mcp_server"
    cwd: "/path/to/obsidian-ai"
```

---

### `search_entities(entity_name, entity_type=..., n=10, use_graph=False)`

Find notes containing a specific entity (person, project, technology, etc.). Supports type filtering and graph-based expansion for finding connected notes.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `entity_name` | `str` | *(required)* | Entity name to search for (case-insensitive) |
| `entity_type` | `str` | `None` | Filter by entity type (Person, Project, Technology, etc.) |
| `n` | `int` | `10` | Maximum number of results |
| `use_graph` | `bool` | `False` | If True, expand results by following wiki-links to find connected notes |

**Returns:** `list[dict]` — results with path, title, entity_name, entity_type, snippet, confidence.

```json
[
  {"path": "Notes/python.md", "title": "python", "entity_name": "Python", "entity_type": "Technology", "snippet": "Python is a...", "confidence": 0.95}
]
```

**How it works:**
1. Searches the entity inverted index (`data/entities.json`) for matching entities by prefix
2. If few results, falls back to ChromaDB `$contains` on `entities_str` metadata
3. If `use_graph=True`, performs 1-hop BFS on each result's wiki-link graph

---

### `get_note_entities(path)`

Return all entities found in a specific note during indexing.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `path` | `str` | Vault-relative path (e.g., `Notes/topic.md`) |

**Returns:** `list[dict]` — entity records with entity_name, entity_type, confidence.

---

### `get_entity_types()`

List all entity type labels present in the index (e.g., Person, Technology, Project).

**Parameters:** None

**Returns:** `list[str]` — unique entity types sorted alphabetically.

---

## Dependencies

- `fastmcp` — MCP server framework
- `pyyaml` — YAML frontmatter parsing
- `requests` — HTTP client for Obsidian API and Ollama
- `obsidian_client.py` — Obsidian REST API wrapper
- `llm_client.py` — Ollama embedding + chat
- `chroma_store.py` — ChromaDB vector search
- `pipelines.py` — Query & action pipelines (LLM-powered)
- `indexer.py` — Indexing pipeline (for `sync_index`)
