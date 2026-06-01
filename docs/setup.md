# Setup

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- [Obsidian](https://obsidian.md/) with the [Local REST API](https://github.com/coddingtonbear/obsidian-local-rest-api) plugin enabled

## 1. Install Ollama and pull models

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

Verify the embedding model works:

```bash
curl http://localhost:11434/api/embeddings -d '{"model": "nomic-embed-text", "prompt": "test"}'
```

## 2. Enable the Obsidian Local REST API plugin

1. Open Obsidian → Settings → Community Plugins → Browse
2. Search for **Local REST API** and install it
3. Enable the plugin
4. Copy the API key from the plugin settings
5. Confirm the API responds on port 27123:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:27123/vault/
```

## 3. Install Python dependencies

```bash
cd obsidian-ai
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

## 4. Configure environment variables

Create a `.env` file in the project root:

```env
OBSIDIAN_HOST=localhost
OBSIDIAN_PORT=27123
OBSIDIAN_API_KEY=your_api_key_here

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=qwen3:8b

CHROMA_PATH=data/chroma_db
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSIDIAN_HOST` | `localhost` | Obsidian REST API hostname |
| `OBSIDIAN_PORT` | `27123` | Obsidian REST API port |
| `OBSIDIAN_API_KEY` | *(required)* | Bearer token from the plugin |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_CHAT_MODEL` | `qwen3:8b` | Chat/LLM model name |
| `CHROMA_PATH` | `data/chroma_db` | ChromaDB storage path |

## 5. Run the indexer

### One-shot index

```bash
python -m obsidian_ai.indexer
```

### Watch mode (auto-index on changes)

```bash
python -m obsidian_ai.indexer --watch
```

This starts a file watcher that automatically re-indexes notes when they are created, modified, deleted, or renamed.

This will:
1. Connect to your Obsidian vault via the REST API
2. Fetch all `.md` notes (excluding backup folders and `.excalidraw.md` files)
3. Chunk notes into 500-word segments with 100-word overlap
4. Embed each chunk via Ollama
5. Store embeddings in ChromaDB

Expected output:

```
INFO: Starting index — 50 notes found
INFO: Indexed: path/to/note1.md
INFO: Indexed: path/to/note2.md (3 chunks)
...
INFO: Done — Indexed: 42, Skipped: 5, Failed: 0
```

## 6. Verify the index

Check that ChromaDB was created:

```bash
ls data/chroma_db
```

Or use the CLI stats command:

```bash
python cli.py stats
```

Errors during indexing are logged to `logs/indexer.log`.

## 7. Run the MCP server

```bash
python -m obsidian_ai.mcp_server
```

The server runs on stdio — no HTTP port. Connect via an MCP client or agent.

### Test the server

You can test tools directly via Python:

```python
from obsidian_ai.mcp_server import search_notes, list_notes, ask_vault

print(list_notes())
print(search_notes("python"))
print(ask_vault("What are my notes about?"))
```

## 8. Configure an agent (optional)

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

Add to your Goose MCP config:

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

Tool calls are logged to `logs/mcp_calls.log`.

## 9. CLI (optional)

A CLI wrapper is available for quick access to common operations:

```bash
python cli.py index          # Run full index
python cli.py watch          # Start file watcher
python cli.py search "query" # Semantic search
python cli.py tag "query"    # Auto-tag notes
python cli.py stats          # Show index stats
```
