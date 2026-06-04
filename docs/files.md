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
│       ├── summary_store.py         # Summary embedding storage
│       ├── ranker.py                # Ranking and composite search
│       ├── pipelines.py             # Query & action pipelines (LLM-powered, agent routing)
│       ├── entity_store.py          # Entity inverted index (extraction, search, persistence)
│       ├── graph_store.py           # Wiki-link graph (BFS, communities, orphans, export)
│       ├── keyword_search.py        # Keyword-based search (BM25)
│       ├── clustering.py            # Semantic clustering of notes
│       ├── wiki_links.py            # Wiki-link parsing/normalization utilities
│       ├── tools/                   # MCP tool submodules
│       │   ├── __init__.py          # Tool registration
│       │   ├── _shared.py           # Shared helpers
│       │   ├── search.py            # 11 search/retrieval tools
│       │   ├── notes.py             # 14 note CRUD tools
│       │   ├── graph.py             # 20 entity/graph tools
│       │   ├── todos.py             # 20 todo management tools
│       │   └── misc.py              # get_clusters, health_check
│       └── mcp_server.py            # FastMCP server with 67 vault tools
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
│   ├── clusters.json                # Semantic clustering cache
│   ├── combined_cache.json          # Combined entity+summary cache
│   ├── content_hashes.json          # Content hash map for incremental indexing
│   ├── embed_cache.json             # Embedding cache
│   ├── entity_cache.json            # Cached LLM entity extraction results
│   ├── expand_cache.json            # Query expansion cache (TTL-based)
│   ├── mtime_map.json               # File modification time map
│   ├── note_paths.json              # Note path → title mapping
│   ├── summary_cache.json           # Cached LLM summary results
│   ├── graph.json                   # Serialized wiki-link graph
│   └── entities.json                # Entity store persistence
├── logs/                            # Log files (gitignored)
│   ├── indexer.log
│   └── mcp_calls.log
├── tests/                           # Unit tests (381+)
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
├── .env                             # API keys, ports, model names (gitignored)
├── .gitignore
├── pyproject.toml                   # Project dependencies + config
└── README.md
```
