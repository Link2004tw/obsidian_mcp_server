# Obsidian AI Knowledge System

A local, privacy-first knowledge management layer that gives an Obsidian vault semantic search, automatic tagging, backlink generation, and natural language querying — all running on local hardware with no cloud dependency.

---

## Goals

- Turn a plain Obsidian vault into a queryable, self-organizing knowledge base
- Enable natural language interaction with notes ("find everything related to X")
- Automate tagging and backlink creation based on semantic meaning, not manual effort
- Keep all processing local — no data leaves the machine
- Build on an open, tool-based architecture that can be extended over time

---

## Stack

| Layer | Technology | Role |
|---|---|---|
| Note storage | Obsidian + Local REST API plugin | Source of truth; read/write interface |
| Embeddings | `nomic-embed-text` via Ollama | Convert note text to semantic vectors |
| Vector database | ChromaDB (local) | Store and search embeddings |
| MCP server | Python + `fastmcp` | Expose vault tools to any MCP-capable agent |
| Local LLM | Qwen3:8b via Ollama | Reasoning, tagging suggestions, summarization |
| Agent runner | Goose (Block) | Agentic multi-step task execution over MCP |
| Dev workflow | opencode + DeepSeek V3 API | AI-assisted coding while building the project |
| Language | Python 3.11+ | All backend and tooling |

**Key libraries:** `fastmcp`, `chromadb`, `requests`, `watchdog`

**Hardware target:** RTX 3060 (6GB VRAM), 16GB RAM

---

## Architecture

```
User (Goose / CLI)
       │
       ▼
  MCP Server (fastmcp)
  ┌────────────────────────────────┐
  │  search_notes(query)           │
  │  read_note(path)               │
  │  write_note(path, content)     │
  │  add_tags(path, tags)          │
  │  create_backlink(a, b)         │
  │  list_notes()                  │
  │  sync_index()                  │
  └────┬──────────────┬────────────┘
       │              │
       ▼              ▼
  ChromaDB       Obsidian REST API
  (embeddings)   (note CRUD)
       │
       ▼
  Ollama — nomic-embed-text
       +
  Ollama — Qwen3:8b (reasoning)
```

---

## Phases

### Phase 1 — Vault API + Embeddings Foundation
**Goal:** Establish the data pipeline. By the end, semantic search works from a Python script.

- Connect to Obsidian via the Local REST API plugin
- Pull all `.md` files and their content
- Embed each note using `nomic-embed-text` through Ollama
- Store embeddings in ChromaDB with metadata (`path`, `title`, `mtime`)
- Write a test query: `search("maria")` returns ranked relevant notes

**Exit criteria:** `python indexer.py` runs cleanly and a test semantic search returns sensible results.

---

### Phase 2 — MCP Server
**Goal:** Expose vault operations as callable tools any MCP-compatible agent can use.

- Build a `fastmcp` server with 7 core tools
- Each tool wraps either ChromaDB (for search) or the Obsidian REST API (for CRUD)
- Test each tool independently before agent integration
- Document tool signatures and expected inputs/outputs

**Exit criteria:** All tools callable via MCP inspector or direct HTTP; no agent required yet.

---

### Phase 3 — LLM + Agent Integration
**Goal:** Wire Qwen3:8b and Goose to the MCP server for end-to-end natural language workflows.

- Configure Goose to point at the MCP server
- Implement query mode: embed query → semantic search → stuff context → LLM response
- Implement action mode: "tag notes about X" → search → LLM reasons → `add_tags` called
- Test multi-step agent flows (find + tag + link in one run)

**Exit criteria:** "Find notes related to Maria and add #maria to all of them" works end-to-end.

---

### Phase 4 — Incremental Indexing + Polish
**Goal:** Make the system maintainable and responsive to vault changes over time.

- Add `watchdog`-based file watcher to re-embed only modified/new notes
- Expose `sync_index` as an MCP tool so the agent can trigger re-indexing
- Handle edge cases: deleted notes, renamed files, empty notes
- Write a basic `README.md` and usage guide
- Optional: small CLI wrapper for common commands

**Exit criteria:** Adding a new note to Obsidian automatically updates the index within seconds.

---

## File Structure

```
obsidian-ai/
├── indexer.py           # One-shot indexing + watch mode
├── mcp_server.py        # fastmcp tool definitions
├── chroma_store.py      # ChromaDB read/write wrapper
├── obsidian_client.py   # Obsidian REST API wrapper
├── llm_client.py        # Ollama embedding + chat wrapper
├── config.py            # Vault path, ports, API key, model names
├── description.md       # This file
└── tasks.md             # Full task list
```

---

## Design Principles

- **Embeddings handle memory** — ChromaDB knows what notes mean
- **LLM handles reasoning** — Qwen3 decides what to do with retrieved notes
- **MCP handles actions** — all vault mutations go through tools, never direct file I/O
- **Obsidian stays the source of truth** — the system enhances the vault, never replaces it
