# Obsidian AI Knowledge System

A local, privacy-first knowledge management layer for Obsidian vaults. Provides semantic search, automatic tagging, backlink generation, and natural language querying — all running locally with no cloud dependency.

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

| Command | Description |
|---------|-------------|
| `python cli.py index` | One-shot full index |
| `python cli.py watch` | Start file watcher daemon |
| `python cli.py search <query>` | Semantic search (`-n` for result count) |
| `python cli.py tag <query>` | Auto-tag notes (`-n` for note count) |
| `python cli.py stats` | Show total notes in index |

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
| `list_notes` | — | List all note paths in the vault |
| `add_tags` | `path: str`, `tags: list[str]` | Add tags to YAML frontmatter |
| `create_backlink` | `path_a: str`, `path_b: str` | Create mutual `[[backlinks]]` |
| `sync_index` | — | Re-run the full indexer pipeline |
| `ask_vault` | `question: str`, `top_k: int = 3` | Ask a question, get an LLM-powered answer from vault content |
| `tag_notes` | `query: str`, `top_k: int = 5` | Auto-suggest tags for notes matching a query |

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
│       └── mcp_server.py          # MCP server (9 tools)
├── cli.py                         # CLI wrapper (index, watch, search, tag, stats)
├── docs/                          # User documentation
│   ├── setup.md
│   ├── architecture.md
│   ├── api.md
│   ├── mcp_server.md
│   ├── indexer.md
│   └── troubleshooting.md
├── dev/                           # Internal project docs
│   ├── tasks.md
│   ├── description.md
│   └── word_limit_fix.md
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
| `CHROMA_PATH` | `data/chroma_db` | ChromaDB persistent storage path |

---

## Logs

| Log file | Contents |
|----------|----------|
| `logs/indexer.log` | Indexing errors with timestamps |
| `logs/mcp_calls.log` | MCP tool calls with timestamps |
