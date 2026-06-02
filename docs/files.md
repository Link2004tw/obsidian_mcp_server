# Project Structure

```
obsidian-ai/
├── src/
│   └── obsidian_ai/
│       ├── __init__.py          # Package init
│       ├── config.py            # Loads .env, exports all config values
│       ├── logger.py            # Shared logging module
│       ├── frontmatter.py       # YAML frontmatter parsing/manipulation
│       ├── obsidian_client.py   # Obsidian REST API wrapper (list/get/put notes)
│       ├── llm_client.py        # Ollama embedding + chat wrapper
│       ├── chroma_store.py      # ChromaDB read/write wrapper (upsert/query)
│       ├── indexer.py           # Indexing pipeline + file watcher
│       ├── pipelines.py         # Query & action pipelines (LLM-powered)
│       └── mcp_server.py        # FastMCP server with 13 vault tools
├── cli.py                       # CLI wrapper (index, watch, search, tag, stats)
├── docs/                        # Documentation
│   ├── setup.md
│   ├── architecture.md
│   ├── api.md
│   ├── mcp_server.md
│   ├── indexer.md
│   └── troubleshooting.md
├── dev/                         # Internal project docs
│   └── tasks.md
├── .env                         # API keys, ports, model names (gitignored)
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_frontmatter.py
│   ├── test_modules.py
│   └── test_obsidian_client.py
├── .gitignore
├── pyproject.toml               # Project dependencies + config
├── README.md
└── requirements.txt
```
