# Project Structure

```
obsidian_mcp_server_test/
├── src/
│   └── obsidian_ai/
│       ├── __init__.py              # Package init
│       ├── _index_utils.py          # Indexer utilities (sanitization, hash, tags)
│       ├── config.py                # Loads .env, exports all config values
│       ├── logger.py                # Shared logging module
│       ├── frontmatter.py           # YAML frontmatter parsing/manipulation
│       ├── obsidian_client.py       # Obsidian REST API wrapper (list/get/put notes)
│       ├── llm_client.py            # Ollama embedding + chat wrapper (retry, caching, model switching)
│       ├── chroma_store.py          # ChromaDB read/write wrapper (upsert/query/reset/dedup)
│       ├── indexer.py               # Indexing orchestration + file watcher
│       ├── chunker.py               # Phase 1: chunk, embed, store in ChromaDB (no LLM)
│       ├── entity_extractor.py      # Phase 2a: LLM entity extraction
│       ├── summarizer.py            # Phase 2b: LLM summary generation
│       ├── extract_entities_pipeline.py  # Entity extraction pipeline
│       ├── summarize_pipeline.py    # Summary generation pipeline
│       ├── ranker.py                # Unified ranking (semantic + entity + graph + keyword)
│       ├── entity_relations.py      # Entity relationship graph store
│       ├── summary_store.py         # Summary embedding storage
│       ├── pipelines.py             # Query & action pipelines (LLM-powered, agent routing)
│       ├── entity_store.py          # Entity inverted index (extraction, search, persistence)
│       ├── graph_store.py           # Wiki-link graph (BFS, communities, orphans, export)
│       ├── keyword_search.py        # Keyword-based search (BM25)
│       ├── clustering.py            # Semantic clustering of notes
│       ├── wiki_links.py            # Wiki-link parsing/normalization utilities
│       ├── dashboard.py             # HTML knowledge graph dashboard
│       ├── todos.py                 # Todo implementation (CRUD, NL parsing)
│       ├── eval.py                  # Retrieval evaluation benchmark
│       ├── tools/                   # MCP tool submodules
│       │   ├── __init__.py          # Tool registration
│       │   ├── _shared.py           # Shared helpers
│       │   ├── search.py            # 11 search/retrieval/LLM tools
│       │   ├── notes.py             # 15 note/write/tag/index tools
│       │   ├── graph.py             # 22 entity/graph tools
│       │   ├── todos.py             # 20 todo management tools
│       │   └── misc.py              # 2 tools (get_clusters, health_check)
│       └── mcp_server.py            # FastMCP server with 70 vault tools (total)
├── cli.py                           # CLI wrapper (argparse, 17 commands, bridges to MCP)
├── scripts/                         # Health monitoring scripts
│   ├── monitor_disk_temp.ps1        # Disk temperature monitor (PowerShell)
│   └── temp_dashboard.py            # Temperature dashboard
├── docs/                            # Documentation
│   ├── setup.md
│   ├── architecture.md
│   ├── api.md
│   ├── mcp_server.md
│   ├── indexer.md
│   ├── files.md
│   └── troubleshooting.md
├── dev/                             # Internal project docs
│   ├── description.md
│   ├── final_product.md
│   ├── improvement.md
│   ├── optimizations.md
│   ├── tasks.md
│   └── word_limit_fix.md
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
├── .env                             # API keys, ports, model names (gitignored)
├── .env.example                     # Environment variable template
├── .gitignore
├── .mcp.json                        # MCP server config for editors
├── cli.py                           # CLI entry point
├── pyproject.toml                   # Project dependencies + config
├── uv.lock                          # Locked dependency versions
└── README.md
```
