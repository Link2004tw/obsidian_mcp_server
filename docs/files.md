# Project Structure

```
obsidian-ai/
├── src/
│   └── obsidian_ai/
│       ├── __init__.py              # Package init
│       ├── config.py                # Loads .env, exports all config values
│       ├── logger.py                # Shared logging module
│       ├── frontmatter.py           # YAML frontmatter parsing/manipulation
│       ├── obsidian_client.py       # Obsidian REST API wrapper (list/get/put notes)
│       ├── llm_client.py            # Ollama embedding + chat wrapper (with retry, caching, model switching)
│       ├── chroma_store.py          # ChromaDB read/write wrapper (upsert/query/reset/dedup)
│       ├── indexer.py               # Indexing pipeline + file watcher (chunking, entities, summaries)
│       ├── pipelines.py             # Query & action pipelines (LLM-powered, agent routing)
│       ├── entity_store.py          # Entity inverted index (extraction, search, persistence)
│       ├── graph_store.py           # Wiki-link graph (BFS, communities, orphans, export)
│       ├── wiki_links.py            # Wiki-link parsing/normalization utilities
│       └── mcp_server.py            # FastMCP server with 44 vault tools
├── cli.py                           # CLI wrapper (argparse, 15 commands, bridges to MCP)
├── docs/                            # Documentation
│   ├── setup.md
│   ├── architecture.md
│   ├── api.md
│   ├── mcp_server.md
│   ├── indexer.md
│   ├── files.md
│   └── troubleshooting.md
├── dev/                             # Internal project docs
│   └── tasks.md
├── data/                            # Persistent data storage
│   ├── chroma_db/                   # Vector database (ChromaDB, gitignored)
│   ├── content_hashes.json          # Content hash map for incremental indexing
│   ├── entity_cache.json            # Cached LLM entity extraction results
│   ├── summary_cache.json           # Cached LLM summary results
│   ├── note_paths.json              # Note path → title mapping
│   ├── title_to_path.json           # Title → path reverse mapping
│   ├── graph.json                   # Serialized wiki-link graph
│   └── entities.json                # Entity store persistence
├── logs/                            # Log files (gitignored)
│   ├── indexer.log
│   └── mcp_calls.log
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_chroma_store.py
│   ├── test_config.py
│   ├── test_entity_store.py
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
│   └── test_wiki_links.py
├── .env                             # API keys, ports, model names (gitignored)
├── .gitignore
├── pyproject.toml                   # Project dependencies + config
└── README.md
```
