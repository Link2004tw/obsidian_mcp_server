# Obsidian AI Knowledge System

A local, privacy-first knowledge management layer for Obsidian vaults. Provides semantic search, automatic tagging, entity extraction, graph-based RAG, summarization, and natural language querying — all running locally with no cloud dependency.

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
uv sync
```

### 2. Pull models

**Required:** The embedding model must be pulled before indexing:

```bash
ollama pull nomic-embed-text
```

**Required for LLM features** (query answering, entity extraction, summarization, auto-tagging):

```bash
ollama pull qwen3:4b
```

> You can substitute any Ollama model — set `OLLAMA_EMBED_MODEL` and `OLLAMA_CHAT_MODEL` in `.env`.

### 3. Configure environment

Create `.env` in the project root. For Ollama (default):

```env
OBSIDIAN_HOST=localhost
OBSIDIAN_PORT=27123
OBSIDIAN_API_KEY=your_api_key_here

LLM_PROVIDER=ollama
EMBED_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=qwen3:4b

VAULT_PATH=C:/Users/you/your-vault
CHROMA_PATH=data/chroma_db
```

Or for OpenAI / OpenAI-compatible (Groq, Together, vLLM):

```env
OBSIDIAN_HOST=localhost
OBSIDIAN_PORT=27123
OBSIDIAN_API_KEY=your_api_key_here

LLM_PROVIDER=openai
EMBED_PROVIDER=openai
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.groq.com/openai/v1  # swap for Groq

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
Use `obsidian-ai <command>` (if installed) or `python cli.py <command>`.

| Command | Description |
|---------|-------------|
| `index` | Run full index |
| `watch` | Start file watcher daemon |
| `sync` | Re-run the full indexer pipeline |
| `stats` | Show index statistics (chunks, notes, model) |
| `search <query>` | Semantic search (`-n` for count, `--tags`, `--folder`, `--date-after`, etc.) |
| `read <path>` | Read full note content by path |
| `write <path> <content>` | Create or overwrite a note (use quotes for content) |
| `list-all` | List all note paths in the vault |
| `list-folder <path>` | List notes directly in a folder (non-recursive) |
| `list-folder-deep <path>` | List all notes in a folder (recursive) |
| `read-by-title <title>` | Look up a note by its title (`-f <folder>` to scope) |
| `search-by-tags <tag> [tag...]` | Find notes by YAML frontmatter tags (`-n` for max results) |
| `add-tags <path> <tag> [tag...]` | Add tags to a note's YAML frontmatter |
| `create-backlink <a> <b>` | Create mutual `[[backlinks]]` between two notes |
| `ask <question>` | Ask a natural-language question about the vault (`-k` for context) |
| `tag-notes <query>` | Auto-suggest & apply tags (`-k` for note count) |
| `dashboard [--serve] [-o FILE]` | Generate or serve the HTML knowledge graph dashboard |
| `eval [--use-graph] [--use-summaries]` | Run retrieval evaluation benchmark |

### MCP Server

| Command | Description |
|---------|-------------|
| `python -m obsidian_ai.mcp_server` | Start the MCP server (stdio transport) |
| `obsidian-ai <command>` | CLI wrapper that talks to the MCP server |

---

## MCP Tools (9 total)

All tools that accept a `path` parameter auto-normalize it — you can pass absolute paths or vault-relative paths interchangeably. The vault prefix is stripped automatically.

Each tool (except `ask` and `tools`) takes an `action` parameter — this replaces ~50 specialized tools with 9 dispatch tools.

| Tool | Description |
|------|-------------|
| `ask(query)` | **Universal discovery tool.** Routes any query to the right internal capability via LLM intent detection — handles search, Q&A, entities, graph, summaries, tags, stats, and more |
| `notes(action, ...)` | Note CRUD: read, write, list, list_folder, search_by_tags, read_by_title, add_note_to_subject |
| `tags(action, ...)` | YAML frontmatter tags: add, remove, set, batch_add, auto_suggest |
| `links(action, ...)` | Wiki-link operations: create, backlinks, outgoing, broken |
| `graph(action, ...)` | Graph exploration: communities, community_of, orphans, path, stats, related, traverse, export |
| `entities(action, ...)` | Entity management: search, note_entities, list, aliases, timeline, related, add, merge, change_type, types, weights, import |
| `todo(action, ...)` | Task management: list, add, complete, update, delete, stats, suggestions, link, ask |
| `admin(action, ...)` | Administration: health, reindex, stats, switch_model, sync_todos |
| `tools()` | Tool discovery — list all tools with descriptions and parameter schemas |

### Quick Examples

```bash
# Ask anything — the tool figures out what to do
obsidian-ai ask "What do I have about machine learning?"

# Read a note via the notes tool
obsidian-ai notes read --path "Notes/topic.md"

# Search by tags
obsidian-ai notes search_by_tags --tags "python,ml"

# Find orphan notes
obsidian-ai graph orphans

# Check system health
obsidian-ai admin health
```

---

## Project Structure

```
obsidian_mcp_server_test/
├── src/
│   └── obsidian_ai/
│       ├── __init__.py              # Package init
│       ├── _index_utils.py          # Indexer utilities (sanitization, hash, tags)
│       ├── config.py                # Environment variables and settings
│       ├── logger.py                # Shared logging module
│       ├── frontmatter.py           # YAML frontmatter parsing/manipulation
│       ├── obsidian_client.py       # Obsidian REST API wrapper
│       ├── llm_client.py            # LLM provider abstraction (embeddings + chat)
│       ├── providers/               # Pluggable LLM backends
│       │   ├── __init__.py          # Provider registry & factory
│       │   ├── base.py              # Abstract BaseLLMProvider
│       │   ├── ollama.py            # Ollama provider (default)
│       │   └── openai_provider.py   # OpenAI / Groq / Together / vLLM provider
│       ├── chroma_store.py          # ChromaDB vector storage
│       ├── indexer.py               # Indexing orchestration + file watcher
│       ├── chunker.py               # Phase 1: chunk, embed, store (no LLM)
│       ├── entity_extractor.py      # Phase 2a: LLM entity extraction
│       ├── summarizer.py            # Phase 2b: LLM summary generation
│       ├── extract_entities_pipeline.py  # Entity extraction pipeline
│       ├── summarize_pipeline.py    # Summary generation pipeline
│       ├── ranker.py                # Unified ranking (semantic + entity + graph + keyword)
│       ├── entity_relations.py      # Entity relationship graph store
│       ├── entity_resolver.py       # Cross-vault entity resolution & import
│       ├── summary_store.py         # Summary embedding storage
│       ├── pipelines.py             # Query & action pipelines (LLM-powered)
│       ├── entity_store.py          # Entity extraction and inverted index
│       ├── graph_store.py           # Wiki-link graph storage and traversal
│       ├── keyword_search.py        # Keyword-based search over notes
│       ├── clustering.py            # Semantic clustering of notes
│       ├── wiki_links.py            # Wiki-link parsing utilities
│       ├── dashboard.py             # HTML knowledge graph dashboard
│       ├── todos.py                 # Todo implementation layer
│       ├── eval.py                  # Retrieval evaluation benchmark
│       ├── tools/                   # MCP tool submodules
│       │   ├── __init__.py          # Tool registration (register_all)
│       │   ├── _tool_base.py        # build_tool() decorator, TOOL_MODULES
│       │   ├── _shared.py           # Shared helpers (expand, rewrite, filter)
│       │   ├── ask.py               # ask — universal discovery tool
│       │   ├── notes.py             # notes — note CRUD (7 actions)
│       │   ├── tags.py              # tags — YAML frontmatter (5 actions)
│       │   ├── links.py             # links — wiki-link ops (4 actions)
│       │   ├── graph.py             # graph — exploration (8 actions)
│       │   ├── entities.py          # entities — entity management (13 actions)
│       │   ├── todo.py              # todo — task management (12 actions)
│       │   ├── admin.py             # admin — health, reindex, stats (5 actions)
│       │   └── tools.py             # tools — tool discovery
│       └── mcp_server.py            # FastMCP server (9 consolidated tools)
├── scripts/                         # Health monitoring scripts
│   ├── monitor_disk_temp.ps1        # Disk temperature monitor
│   └── temp_dashboard.py            # Temperature dashboard
├── docs/                            # User documentation
│   ├── setup.md
│   ├── architecture.md
│   ├── api.md
│   ├── mcp_server.md
│   ├── indexer.md
│   ├── files.md
│   └── troubleshooting.md
├── dev/                             # Developer notes and planning
│   ├── description.md
│   ├── final_product.md
│   ├── improvement.md
│   ├── optimizations.md
│   ├── tasks.md
│   └── word_limit_fix.md
├── data/                            # Persistent data
│   ├── chroma_db/                   # Vector database (gitignored)
│   ├── clusters.json
│   ├── combined_cache.json
│   ├── content_hashes.json
│   ├── embed_cache.json
│   ├── entity_cache.json
│   ├── expand_cache.json
│   ├── entity_relations.json
│   ├── graph.json
│   ├── mtime_map.json
│   ├── note_paths.json
│   └── summary_cache.json
├── logs/                            # Log files (gitignored)
│   ├── indexer.log
│   ├── mcp_calls.log
│   ├── todos.log
│   └── test_file.log
├── graphify-out/                    # Knowledge graph exports
│   ├── cache/
│   ├── cost.json
│   ├── GRAPH_REPORT.md
│   ├── graph.html
│   ├── graph.json
│   └── manifest.json
├── tests/                           # Unit tests (400+)
│   ├── test_chroma_store.py
│   ├── test_clustering.py
│   ├── test_config.py
│   ├── test_dashboard.py
│   ├── test_entity_relations.py
│   ├── test_entity_store.py
│   ├── test_eval.py
│   ├── test_frontmatter.py
│   ├── test_graph_store.py
│   ├── test_indexer.py
│   ├── test_keyword_search.py
│   ├── test_llm_client.py
│   ├── test_logger.py
│   ├── test_mcp_server.py
│   ├── test_modules.py
│   ├── test_obsidian_client.py
│   ├── test_pipelines.py
│   ├── test_ranker.py
│   ├── test_summary_store.py
│   ├── test_todos.py
│   └── test_wiki_links.py
├── .claude/                         # Claude agent settings
│   └── settings.local.json
├── .openclaude/                     # OpenClaude agent settings
│   └── settings.local.json
├── .env                             # API keys and config (gitignored)
├── .env.example                     # Environment variable template
├── .gitignore
├── .mcp.json                        # MCP server config for editors
├── cli.py                           # CLI entry point
├── pyproject.toml                   # Project dependencies
├── uv.lock                          # Locked dependency versions
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
| `LLM_PROVIDER` | `ollama` | Chat provider: `ollama` or `openai` |
| `EMBED_PROVIDER` | `ollama` | Embedding provider: `ollama` or `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model name |
| `OLLAMA_CHAT_MODEL` | `qwen3:4b` | Ollama chat/LLM model name |
| `OPENAI_API_KEY` | — | API key for OpenAI / compatible APIs |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL (swap for Groq, Together, vLLM) |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model name |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model name |
| `VAULT_PATH` | — | Absolute path to the Obsidian vault. **Required only for file watcher** (`–watch`). One-shot indexing works via the Obsidian REST API without setting this. |
| `CHROMA_PATH` | `data/chroma_db` | ChromaDB persistent storage path |
| `DATA_DIR` | `data` | Override data storage root |
| `READ_WORKERS` | `2` | Parallel note readers for initial fetch |
| `LLM_CHAT_CONCURRENCY` | `1` | Max concurrent chat calls during indexing |
| `INDEX_BATCH_SIZE` | `50` | Notes per index batch |
| `EMBED_WORKER_FLOOR` | `1` | Min embedding worker threads |
| `EMBED_WORKER_CEIL` | `2` | Max embedding worker threads |
| `LLM_CALL_DELAY` | `0.5` | Delay between LLM calls (seconds) |
| `LLM_CALL_HARD_TIMEOUT` | `30` | Max seconds per LLM call (prevents GPU TDR crash) |
| `GPU_TEMP_LIMIT` | `85` | GPU temperature limit (°C); extraction aborts if exceeded |
| `GPU_VRAM_LIMIT` | `80` | GPU VRAM usage limit (%) |
| `DISK_TEMP_LIMIT` | `80` | Disk temperature limit (°C) |
| `DISK_TEMP_CHECK_INTERVAL` | `30` | Disk temp check interval (seconds) |
| `EXPAND_CACHE_TTL` | `3600` | TTL in seconds for query expansion cache |
| `ENTITY_ALIASES_FILE` | — | Custom entity aliases JSON file |
| `TODO_FILE` | `todos.md` | Todo file name in vault |
| `RANKING_SEMANTIC` | `0.40` | Semantic search ranking weight |
| `RANKING_ENTITY` | `0.30` | Entity signal ranking weight |
| `RANKING_GRAPH` | `0.20` | Graph proximity ranking weight |
| `RANKING_KEYWORD` | `0.10` | Keyword BM25 ranking weight |

---

## Example Use Cases

### "What does my vault say about ESP32?"

```bash
obsidian-ai ask "What projects involve ESP32?"
# → LLM-powered answer synthesised from all ESP32-related notes
```

### "Find all notes about machine learning, tagged for review"

```bash
obsidian-ai search "machine learning" -n 20 --tags review
# → Semantic search filtered by YAML tag
```

### "Auto-tag all notes about embedded systems"

```bash
obsidian-ai tag-notes "embedded systems programming" -k 10
# → LLM reads top-10 matching notes, suggests + applies tags automatically
```

### "Show me the most connected notes in my vault"

```bash
obsidian-ai search-by-tags project
# or via MCP: get_graph_stats() → lists hub notes with most wiki-links
```

### "Create a note about my new PCB project and link it to ESP32"

```bash
obsidian-ai add-note-to-subject ESP32 "PCB Rev3 Design" \
  "Notes about the ESP32 PCB revision 3 design."
# → Creates note under Subjects/ESP32/, adds [[ESP32]] backlink
```

---

## Performance Notes

Tested on **RTX 3060 (6GB VRAM)** with Ollama:

| Operation | Typical Time |
|-----------|-------------|
| Full index (500 notes, ~3k chunks) | ~8-12 minutes |
| Incremental index (1 changed note) | ~2-5 seconds |
| Semantic search (top-5) | ~500ms |
| LLM query answer | ~3-8 seconds |
| Entity extraction per note | ~1-3 seconds |

**Hardware:** `embed_worker_ceil=3`, `llm_chat_concurrency=1`, `chunk_size=500` words. GPU temperature limit set to 85°C to prevent TDR crashes during long index runs.

### Delta Indexing Roadmap

Chunk-level delta indexing is the **primary optimization opportunity** and is targeted for **v0.2.0**. Current behavior: when a note changes, all its chunks are re-embedded. The planned improvement diffs old vs. new chunks and only re-embeds changed sections, delivering ~95% faster re-indexing for small edits. For vaults under 1,000 notes, the current full re-index is acceptable.

---

## Logs
