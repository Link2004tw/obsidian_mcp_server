# Obsidian AI Knowledge System

A local, privacy-first knowledge management layer for Obsidian vaults. Provides semantic search, automatic tagging, entity extraction, graph-based RAG, summarization, and natural language querying — all running locally with no cloud dependency.

---

## Quick Start

### 1. Install dependencies

```bash
uv pip install -e .
```

### 2. Pull models

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

### 3. Configure environment

Create `.env` in the project root:

```env
OBSIDIAN_HOST=localhost
OBSIDIAN_PORT=27123
OBSIDIAN_API_KEY=your_api_key_here

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=qwen3:8b

VAULT_PATH=C:/Users/you/your-vault
CHROMA_PATH=data/chroma_db
```

### 4. Index your vault

```bash
python -m obsidian_ai.indexer
```

### 5. Start the MCP server

```bash
python -m obsidian_ai.mcp_server
```

---

## Commands

### Indexing

| Command | Description |
|---------|-------------|
| `python -m obsidian_ai.indexer` | One-shot full index of the Obsidian vault |
| `python -m obsidian_ai.indexer --watch` | Start file watcher (auto-indexes on changes) |
| `python -m obsidian_ai.indexer --skip-entities` | Index without entity extraction |
| `python -m obsidian_ai.indexer --skip-summaries` | Index without summary generation |

### CLI

All commands communicate with the MCP server via stdio — same code path as any MCP agent.

| Command | Description |
|---------|-------------|
| `python cli.py index` | Run full index |
| `python cli.py watch` | Start file watcher daemon |
| `python cli.py sync` | Re-run the full indexer pipeline |
| `python cli.py stats` | Show index statistics (chunks, notes, model) |
| `python cli.py search <query>` | Semantic search (`-n` for count, `--tags`, `--folder`, `--date-after`, etc.) |
| `python cli.py read <path>` | Read full note content by path |
| `python cli.py write <path> <content>` | Create or overwrite a note (use quotes for content) |
| `python cli.py list-all` | List all note paths in the vault |
| `python cli.py list-folder <path>` | List notes directly in a folder (non-recursive) |
| `python cli.py list-folder-deep <path>` | List all notes in a folder (recursive) |
| `python cli.py read-by-title <title>` | Look up a note by its title (`-f <folder>` to scope) |
| `python cli.py search-by-tags <tag> [tag...]` | Find notes by YAML frontmatter tags (`-n` for max results) |
| `python cli.py add-tags <path> <tag> [tag...]` | Add tags to a note's YAML frontmatter |
| `python cli.py create-backlink <a> <b>` | Create mutual `[[backlinks]]` between two notes |
| `python cli.py ask <question>` | Ask a natural-language question about the vault (`-k` for context) |
| `python cli.py tag-notes <query>` | Auto-suggest & apply tags (`-k` for note count) |

### MCP Server

| Command | Description |
|---------|-------------|
| `python -m obsidian_ai.mcp_server` | Start the MCP server (stdio transport) |

---

## MCP Tools (44 total)

### Search & Retrieval
| Tool | Description |
|------|-------------|
| `search_notes` | Semantic search across indexed notes with metadata filters |
| `batch_search` | Run multiple searches in one call |
| `retrieve_notes` | Multi-strategy retrieval (semantic + entity + graph) |
| `find_duplicate_notes` | Find near-duplicate notes via embedding similarity |
| `search_by_tags` | Find notes by YAML frontmatter tags |
| `search_entities` | Find notes mentioning a specific entity |
| `get_subject` | Get notes related to a free-form subject |

### Reading & Writing
| Tool | Description |
|------|-------------|
| `read_note` | Fetch full note content |
| `write_note` | Create or overwrite a note |
| `list_all_notes` | List all note paths in the vault |
| `list_folder` | List entries directly in a folder (non-recursive) |
| `list_folder_deep` | List all notes in a folder (recursive) |
| `read_note_by_title` | Look up a note by its filename |

### Tag Management
| Tool | Description |
|------|-------------|
| `add_tags` | Add tags to YAML frontmatter |
| `remove_tags` | Remove specific tags from YAML frontmatter |
| `set_tags` | Replace all tags on a note |
| `batch_tag_notes` | Add tags to multiple notes at once |
| `tag_notes` | Auto-suggest tags via LLM |

### Wiki-Link Graph
| Tool | Description |
|------|-------------|
| `create_backlink` | Create mutual backlinks between two notes |
| `get_backlinks` | Return notes linking TO a given note |
| `get_linked_notes` | Return notes a given note links TO |
| `get_broken_links` | Find unresolved wiki-links |
| `get_orphan_notes` | Find notes with no wiki-links |
| `get_graph_stats` | Graph statistics (nodes, edges, hubs, isolates) |
| `get_communities` | Detect communities via label propagation |
| `multi_hop_traversal` | BFS graph traversal from a seed note |
| `related_notes` | Find related notes via semantic + graph proximity |
| `export_graph` | Export wiki-link graph in DOT/JSON |

### LLM-Powered
| Tool | Description |
|------|-------------|
| `ask_vault` | Ask a question, get an LLM-powered answer |
| `ask_agent` | Route a query automatically to the best tool |
| `summarize_topic` | LLM-generated consolidated summary of a topic |
| `tag_notes` | Auto-suggest and apply tags via LLM |

### Entity System
| Tool | Description |
|------|-------------|
| `search_entities` | Find notes mentioning an entity |
| `get_note_entities` | Return all entities found in a note |
| `get_entity_types` | List all entity types in the index |

### Index Management
| Tool | Description |
|------|-------------|
| `sync_index` | Re-run the full indexer pipeline |
| `get_index_stats` | Show index statistics |
| `switch_embedding_model` | Switch embedding model at runtime |

### Todo Management
| Tool | Description |
|------|-------------|
| `get_todos` | List todos with filters |
| `add_todo` | Add a new todo task |
| `complete_todo` | Mark a todo as completed |
| `update_todo` | Update one or more fields of a todo |
| `delete_todo` | Delete a todo by id |
| `sync_todos` | Recalculate todo counts |
| `get_todo_stats` | Aggregated todo statistics |
| `ensure_todo_file` | Create todos.md if missing |

---

## Project Structure

```
obsidian-ai/
├── src/
│   └── obsidian_ai/
│       ├── __init__.py              # Package init
│       ├── config.py                # Environment variables and settings
│       ├── logger.py                # Shared logging module
│       ├── frontmatter.py           # YAML frontmatter parsing/manipulation
│       ├── obsidian_client.py       # Obsidian REST API wrapper
│       ├── llm_client.py            # Ollama embedding + chat wrapper
│       ├── chroma_store.py          # ChromaDB vector storage
│       ├── indexer.py               # Vault indexing pipeline + file watcher
│       ├── pipelines.py             # Query & action pipelines (LLM-powered)
│       ├── entity_store.py          # Entity extraction and inverted index
│       ├── graph_store.py           # Wiki-link graph storage and traversal
│       ├── wiki_links.py            # Wiki-link parsing utilities
│       └── mcp_server.py            # FastMCP server (44 tools)
├── cli.py                           # CLI wrapper (argparse, 15 commands)
├── docs/                            # User documentation
│   ├── setup.md
│   ├── architecture.md
│   ├── api.md
│   ├── mcp_server.md
│   ├── indexer.md
│   ├── files.md
│   └── troubleshooting.md
├── data/                            # Persistent data
│   ├── chroma_db/                   # Vector database (gitignored)
│   ├── content_hashes.json
│   ├── entity_cache.json
│   ├── summary_cache.json
│   ├── note_paths.json
│   ├── title_to_path.json
│   ├── graph.json
│   └── entities.json
├── logs/                            # Log files (gitignored)
│   ├── indexer.log
│   └── mcp_calls.log
├── tests/                           # Unit tests (226+)
├── .env                             # API keys and config (gitignored)
├── pyproject.toml                   # Project dependencies
└── README.md
```

---

## Documentation

- [Setup](docs/setup.md) — Installation and configuration
- [Architecture](docs/architecture.md) — System design and data flow
- [API Reference](docs/api.md) — Function signatures for all modules
- [MCP Server](docs/mcp_server.md) — Tool details and agent configuration
- [Indexer](docs/indexer.md) — Chunking, entity extraction, summaries
- [Files](docs/files.md) — Project structure reference
- [Troubleshooting](docs/troubleshooting.md) — Common errors and fixes

---

## Agent Configuration

Add this MCP server to any compatible AI agent. The server uses **stdio** transport.

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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSIDIAN_HOST` | `localhost` | Obsidian REST API host |
| `OBSIDIAN_PORT` | `27123` | Obsidian REST API port |
| `OBSIDIAN_API_KEY` | — | API key from Obsidian Local REST API plugin |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_CHAT_MODEL` | `qwen3:8b` | Chat/LLM model name |
| `VAULT_PATH` | — | Absolute path to the Obsidian vault (required for file watcher) |
| `CHROMA_PATH` | `data/chroma_db` | ChromaDB persistent storage path |

---

## Logs

| Log file | Contents |
|----------|----------|
| `logs/indexer.log` | Indexing errors with timestamps |
| `logs/mcp_calls.log` | MCP tool calls with timestamps |
