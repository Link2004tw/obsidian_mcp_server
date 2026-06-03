# Obsidian AI Knowledge System

A local, privacy-first knowledge management layer for Obsidian vaults. Provides semantic search, automatic tagging, backlink generation, and natural language querying — all running locally with no cloud dependency.

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

### CLI

All commands communicate with the MCP server via stdio — same code path as any MCP agent.

| Command | Description |
|---------|-------------|
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
| `python cli.py watch` | Start file watcher daemon (auto-indexes on changes) |
| `python cli.py sync` | Re-run the full indexer pipeline |
| `python cli.py stats` | Show index statistics (chunks, notes, model) |
| `python cli.py ask <question>` | Ask a natural-language question about the vault (`-k` for context) |
| `python cli.py tag-notes <query>` | Auto-suggest & apply tags (`-k` for note count) |

### MCP Server

| Command | Description |
|---------|-------------|
| `python -m obsidian_ai.mcp_server` | Start the MCP server (stdio transport) |

### MCP Tools (via agent)

| Tool | Parameters | Description |
|------|------------|-------------|
| `search_notes` | `query: str`, `n: int = 5` | Semantic search across indexed notes |
| `read_note` | `path: str` | Fetch full note content |
| `write_note` | `path: str`, `content: str` | Create or overwrite a note |
| `list_all_notes` | — | List all note paths in the vault |
| `list_folder` | `folder_path: str` | List note paths directly in a folder (non-recursive) |
| `list_folder_deep` | `folder_path: str` | List all note paths in a folder (recursive, includes subdirs) |
| `add_tags` | `path: str`, `tags: list[str]` | Add tags to YAML frontmatter |
| `create_backlink` | `path_a: str`, `path_b: str` | Create mutual `[[backlinks]]` |
| `sync_index` | — | Re-run the full indexer pipeline |
| `ask_vault` | `question: str`, `top_k: int = 3` | Ask a question, get an LLM-powered answer from vault content |
| `tag_notes` | `query: str`, `top_k: int = 5` | Auto-suggest tags for notes matching a query |
| `search_entities` | `entity_name: str`, `entity_type: str?`, `n: int = 10`, `use_graph: bool = False` | Find notes mentioning an entity, with optional graph expansion |
| `get_note_entities` | `path: str` | Return all entities found in a specific note |
| `get_entity_types` | — | List all entity types present in the index |

### Development

| Command | Description |
|---------|-------------|
| `uv pip install -e .` | Install project in editable mode |
| `ollama pull nomic-embed-text` | Pull/update the embedding model |
| `ollama list` | List installed Ollama models |
| `ollama serve` | Start Ollama server manually |

---

## Project Structure

```
obsidian-ai/
├── src/
│   └── obsidian_ai/
│       ├── __init__.py
│       ├── config.py              # Environment variables and settings
│       ├── logger.py              # Shared logging module
│       ├── obsidian_client.py     # Obsidian REST API wrapper
│       ├── llm_client.py          # Ollama embedding wrapper
│       ├── chroma_store.py        # ChromaDB vector storage
│       ├── indexer.py             # Vault indexing pipeline + file watcher
│       ├── pipelines.py           # Query & action pipelines (LLM-powered)
│       ├── entity_store.py        # Entity extraction and inverted index
│       ├── graph_store.py         # Wiki-link graph storage and traversal
│       └── mcp_server.py          # MCP server
├── cli.py                         # CLI wrapper (index, watch, search, tag, stats)
├── docs/                          # User documentation
│   ├── setup.md
│   ├── architecture.md
│   ├── api.md
│   ├── mcp_server.md
│   ├── indexer.md
│   └── troubleshooting.md
├── dev/                           # Internal project docs
│   └── tasks.md
├── data/
│   └── chroma_db/                 # Vector database (gitignored)
├── logs/                          # Log files (gitignored)
│   ├── indexer.log
│   └── mcp_calls.log
├── .env                           # API keys and config (gitignored)
├── pyproject.toml                 # Project dependencies
└── README.md
```

---

## Documentation

- [Setup](docs/setup.md) — Installation and configuration
- [Architecture](docs/architecture.md) — System design and data flow
- [API Reference](docs/api.md) — Function signatures for all modules
- [MCP Server](docs/mcp_server.md) — Tool details and agent configuration
- [Indexer](docs/indexer.md) — Chunking, sanitization, and indexing details
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

---

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
