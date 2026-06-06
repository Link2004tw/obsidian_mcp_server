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

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b
```

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

## MCP Tools (70 total)

All tools that accept a `path` parameter auto-normalize it — you can pass absolute paths or vault-relative paths interchangeably. The vault prefix is stripped automatically.

### Search & Retrieval
| Tool | Description |
|------|-------------|
| `search_notes` | Semantic search with metadata filters, optional auto-rewrite, TTL-cached query expansion |
| `batch_search` | Run multiple searches in one call |
| `composite_search` | High-recall composite search (summary + entity + community) |
| `retrieve_notes` | Multi-strategy retrieval (semantic + entity + graph) |
| `find_duplicate_notes` | Find near-duplicate notes via embedding similarity |
| `search_by_tags` | Find notes by YAML frontmatter tags |
| `search_entities` | Find notes mentioning a specific entity |
| `get_subject` | Get notes related to a free-form subject |

### Reading & Writing
| Tool | Description |
|------|-------------|
| `read_note` | Fetch full note content |
| `write_note` | Create or overwrite a note (YAML-validated) |
| `list_all_notes` | List all note paths in the vault |
| `list_folder` | List entries directly in a folder (non-recursive) |
| `list_folder_deep` | List all notes in a folder (recursive) |
| `read_note_by_title` | Look up a note by its filename |
| `add_note_to_subject` | Create a note and auto-link it into a subject |

### Tag Management
| Tool | Description |
|------|-------------|
| `add_tags` | Add tags to YAML frontmatter |
| `remove_tags` | Remove specific tags from YAML frontmatter |
| `set_tags` | Replace all tags on a note |
| `batch_tag_notes` | Add tags to multiple notes at once |
| `tag_notes` | Auto-suggest and apply tags via LLM |

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
| `get_note_community` | Show which community a note belongs to |
| `multi_hop_traversal` | BFS graph traversal from a seed note |
| `related_notes` | Find related notes via semantic + graph proximity |
| `export_graph` | Export wiki-link graph in DOT/JSON |
| `get_shortest_path` | Find shortest path between two notes in the graph |

### LLM-Powered
| Tool | Description |
|------|-------------|
| `ask_vault` | Ask a question, get an LLM-powered answer |
| `ask_agent` | Route a query automatically to the best tool |
| `summarize_topic` | LLM-generated consolidated summary of a topic |

### Entity System
| Tool | Description |
|------|-------------|
| `search_entities` | Find notes mentioning an entity |
| `get_note_entities` | Return all entities found in a note |
| `get_entity_types` | List all entity types in the index |
| `get_entity_aliases` | List aliases for an entity |
| `list_entities` | List entities with optional type filter |
| `add_entity` | Register a new entity with metadata |
| `merge_entities` | Merge duplicate entities in the index |
| `import_entities` | Import entities from another vault with configurable dedup |
| `entity_timeline` | Show timeline of entity mentions across notes |
| `related_entities` | Find related entities via relationship graph |
| `get_ranking_weights` | View current ranking weights for entity/keyword/graph |
| `set_ranking_weights` | Customize ranking weights |

### Clustering
| Tool | Description |
|------|-------------|
| `get_clusters` | Return semantic clusters of notes based on embedding similarity |

### Health
| Tool | Description |
|------|-------------|
| `health_check` | Check backend service status (Ollama, ChromaDB, Obsidian API) |

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
| `get_todos_by_priority` | List todos grouped by priority |
| `add_todo_from_natural_language` | Add todo from plain text |
| `suggest_task_priority` | Suggest priority for a task via LLM |
| `suggest_due_date` | Suggest due date for a task via LLM |
| `suggest_task_splitting` | Suggest how to split a large task |
| `get_overdue_summary` | Summary of all overdue todos |
| `estimate_completion_date` | Estimate completion date for a project |
| `get_todos_for_note` | Find todos linked to a specific note |
| `get_notes_for_todo` | Find notes linked to a specific todo |
| `link_todo_to_notes` | Link a todo to one or more notes |
| `ask_vault_about_todo` | Ask about a specific todo in vault context |
| `ask_vault_about_todos` | Ask a question about all todos |

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
│       │   ├── __init__.py          # Tool registration
│       │   ├── _shared.py           # Shared helpers (expand, rewrite, filter)
│       │   ├── search.py            # 11 search/retrieval tools
│       │   ├── notes.py             # 14 note CRUD tools
│       │   ├── graph.py             # 20 entity/graph tools
│       │   ├── todos.py             # 20 todo management tools
│       │   └── misc.py              # get_clusters, health_check
│       └── mcp_server.py            # FastMCP server (67 tools)
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
| `VAULT_PATH` | — | Absolute path to the Obsidian vault (required for file watcher) |
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

## Logs

| Log file | Contents |
|----------|----------|
| `logs/indexer.log` | Indexing errors with timestamps |
| `logs/mcp_calls.log` | MCP tool calls with timestamps |
