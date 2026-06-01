# API Reference

## config.py

Module-level constants loaded from `.env`:

```python
obsidian_host: str          # OBSIDIAN_HOST, default "localhost"
obsidian_port: int          # OBSIDIAN_PORT, default 27123
obsidian_api_key: str       # OBSIDIAN_API_KEY, required
ollama_base_url: str        # OLLAMA_BASE_URL, default "http://localhost:11434"
ollama_embed_model: str     # OLLAMA_EMBED_MODEL, default "nomic-embed-text"
ollama_chat_model: str      # OLLAMA_CHAT_MODEL, default "qwen3:8b"
chroma_path: str            # CHROMA_PATH, default "data/chroma_db"
EXCLUDE_PATTERNS: list[str] # Hardcoded exclusion patterns
```

---

## obsidian_client.py

### `list_notes() -> list[str]`

Returns a flat list of all `.md` file paths in the Obsidian vault. Recursively walks the vault tree via the REST API. Skips entries matching `EXCLUDE_PATTERNS`.

```python
from obsidian_ai import obsidian_client
notes = obsidian_client.list_notes()
# ["Courses/notes/Topic.md", "Journal/2024-01-01.md", ...]
```

### `get_note(path: str) -> str`

Fetches the full content of a note by its vault path. Auto-appends `.md` extension if missing.

```python
content = obsidian_client.get_note("Courses/notes/Topic.md")
# "# Topic\n\nThis is the note content..."
# Also works: get_note("Courses/notes/Topic")
```

**Raises:** `requests.HTTPError` if the note doesn't exist or the API fails.

### `put_note(path: str, content: str) -> None`

Creates or overwrites a note at the given path. Content is encoded as UTF-8.

```python
obsidian_client.put_note("New Note.md", "# Hello\n\nNew content here.")
```

---

## llm_client.py

### Constants

```python
REQUEST_TIMEOUT = 120    # Timeout for chat requests (seconds)
EMBED_TIMEOUT = 180      # Timeout for embedding requests (seconds)
MAX_RETRIES = 3           # Max retry attempts on failure
INITIAL_BACKOFF = 2       # Initial backoff seconds (doubles each attempt)
MAX_CONTEXT_WORDS = 3000  # Max words for LLM context truncation
```

### Retry Behavior

All Ollama requests use `_request_with_retry()` — retries up to 3 times with exponential backoff (2s → 4s → 8s) on:
- `ReadTimeout` — server took too long to respond
- `ConnectionError` — server unreachable
- `HTTPError` with status 429 (rate limit), 502 (bad gateway), 503 (service unavailable)

### `embed(text: str) -> list[float]`

Converts text to a 768-dimensional embedding vector via Ollama.

```python
from obsidian_ai import llm_client
vector = llm_client.embed("What is machine learning?")
# [0.665, 0.270, -4.427, ...] (768 floats)
```

**Raises:** `requests.HTTPError` if Ollama fails after all retries (e.g., model not loaded, input too large).

### `chat(messages: list[dict], model: str = None, think: bool = True) -> str`

Send a chat completion request to Ollama and return the response text. Uses same retry logic as `embed()` (3 attempts, exponential backoff).

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]
response = llm_client.chat(messages)
# "Python is a high-level programming language..."
```

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | `list[dict]` | *(required)* | OpenAI-format message list |
| `model` | `str` | `config.ollama_chat_model` | Model to use |
| `think` | `bool` | `True` | Enable thinking mode (set `False` for speed) |

**Raises:** `requests.HTTPError` if Ollama fails after all retries.

### `truncate_to_budget(text: str, max_words: int = MAX_CONTEXT_WORDS) -> str`

Truncate text to a word budget, appending `[truncated]` if cut.

```python
long_text = "word " * 5000
short = llm_client.truncate_to_budget(long_text, max_words=1000)
# "word word ... (1000 words) [truncated]"
```

---

## chroma_store.py

### `upsert(path: str, chunk_idx: int, embedding: list[float], metadata: dict) -> None`

Inserts or updates a single chunk's embedding in ChromaDB.

- **Document ID:** `{path}::chunk_{chunk_idx}`
- **Metadata fields:** `path`, `title`, `chunk`, `word_count`

```python
chroma_store.upsert(
    path="Courses/notes/Topic.md",
    chunk_idx=0,
    embedding=[0.665, 0.270, ...],
    metadata={"path": "Courses/notes/Topic.md", "title": "Topic", "chunk": 0, "word_count": 500}
)
```

### `get_by_path(path: str) -> list[dict]`

Returns all metadata entries for a given note path.

```python
entries = chroma_store.get_by_path("Courses/notes/Topic.md")
# [{"path": "Courses/notes/Topic.md", "title": "Topic", "chunk": 0, "word_count": 500}]
```

### `delete_by_path(path: str) -> None`

Deletes all chunks belonging to a given note path. Used before re-indexing to wipe stale chunks.

```python
chroma_store.delete_by_path("Courses/notes/Topic.md")
```

### `count() -> int`

Returns the total number of indexed chunks in ChromaDB.

```python
total = chroma_store.count()
# 157
```

### `dedup_paths(results: list[dict]) -> list[tuple[str, str]]`

Deduplicates query results by note path. Returns `[(path, title), ...]`.

```python
pairs = chroma_store.dedup_paths(results)
# [("Courses/notes/Topic.md", "Topic"), ("Journal/2024-01-01.md", "2024-01-01")]
```

### `query(embedding: list[float], n: int = 5) -> list[dict]`

Semantic search. Returns the top-k closest chunks.

```python
results = chroma_store.query(embedding=[0.665, 0.270, ...], n=5)
# [
#   {"id": "path::chunk_0", "metadata": {...}, "distance": 0.42},
#   {"id": "path::chunk_1", "metadata": {...}, "distance": 0.51},
#   ...
# ]
```

Each result contains:
- `id` — ChromaDB document ID (`{path}::chunk_{N}`)
- `metadata` — dict with `path`, `title`, `chunk`, `word_count`
- `distance` — cosine distance (lower = more similar)

---

## indexer.py

### Constants

```python
SKIP_MIN_TOKENS = 20    # Notes under this word count are skipped
CHUNK_SIZE = 500         # Words per chunk
CHUNK_OVERLAP = 100      # Word overlap between chunks
```

### `chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]`

Splits text into overlapping word-based chunks.

```python
from indexer import chunk_text
chunks = chunk_text("word " * 1200, size=500, overlap=100)
# ["word word ... (500 words)", "word word ... (500 words)", "word word ... (300 words)"]
```

### `run_index() -> None`

Main indexing pipeline. Fetches all notes, chunks them, embeds, and stores in ChromaDB. Prints summary stats and logs errors to `indexer.log`.

```python
from indexer import run_index
run_index()
# Done. Indexed: 42, Skipped: 5, Failed: 0
```

### `_index_note(path: str) -> bool`

Index a single note. Used by the file watcher for incremental updates.

```python
from indexer import _index_note
_index_note("Notes/topic.md")
# True if successful
```

### `_delete_note(path: str) -> bool`

Delete a note from the index. Used by the file watcher on file deletion.

```python
from indexer import _delete_note
_delete_note("Notes/topic.md")
# True if successful
```

### `watch() -> None`

Start the file watcher daemon. Monitors the vault directory for changes and automatically indexes/deletes notes.

```python
from indexer import watch
watch()
# Press Ctrl+C to stop
```

**Events handled:**
- `on_created` → `_index_note(path)`
- `on_modified` → `_index_note(path)`
- `on_deleted` → `_delete_note(path)`
- `on_moved` → `_delete_note(old_path)` + `_index_note(new_path)`

### `add_tags_to_note(path: str, tags: list[str]) -> None`

Adds tags to a note's YAML frontmatter. Creates frontmatter if absent.

```python
from indexer import add_tags_to_note
add_tags_to_note("Notes/topic.md", ["python", "ml"])
```

**Behavior:**
- Parses existing YAML frontmatter (or creates new)
- Appends new tags (no duplicates)
- Converts string `tags` field to list if needed

---

## pipelines.py

LLM-powered pipelines for querying and tagging notes.

### `query(ask: str, top_k: int = 3) -> str`

Ask a question about the vault. Searches relevant notes, stuffs them into an LLM prompt, and returns the answer.

```python
from obsidian_ai import pipelines
answer = pipelines.query("What are my notes about Python?")
# "Your vault contains notes on Python basics, advanced patterns, ..."
```

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ask` | `str` | *(required)* | Natural language question |
| `top_k` | `int` | `3` | Number of notes to retrieve |

### `tag_notes(ask: str, top_k: int = 5) -> str`

Search notes matching a query and auto-suggest tags via LLM.

```python
result = pipelines.tag_notes("machine learning")
# "Tagged 3 notes: {'Notes/nn.md': ['neural-network', 'deep-learning'], ...}"
```

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ask` | `str` | *(required)* | Search query |
| `top_k` | `int` | `5` | Number of notes to process |

---

## mcp_server.py

FastMCP server exposing 9 vault tools. Run with `python -m obsidian_ai.mcp_server`.

### `search_notes(query: str, n: int = 5) -> list[dict]`

Semantic search. Embeds query, searches ChromaDB, deduplicates by path.

```python
result = mcp_client.call_tool("search_notes", {"query": "python", "n": 5})
# [{"path": "Notes/topic.md", "title": "topic", "chunk": 0, "distance": 0.42}]
```

### `read_note(path: str) -> str`

Fetches full note content from Obsidian.

```python
content = mcp_client.call_tool("read_note", {"path": "Notes/topic.md"})
```

### `write_note(path: str, content: str) -> str`

Creates or overwrites a note.

```python
mcp_client.call_tool("write_note", {"path": "New Note.md", "content": "# Hello"})
```

### `list_notes() -> list[str]`

Returns all `.md` paths in the vault.

```python
notes = mcp_client.call_tool("list_notes", {})
# ["Notes/topic.md", "Journal/2024-01-01.md", ...]
```

### `add_tags(path: str, tags: list[str]) -> str`

Adds tags to YAML frontmatter. Creates frontmatter if absent.

```python
mcp_client.call_tool("add_tags", {"path": "Notes/topic.md", "tags": ["python", "ml"]})
# "Tags added to Notes/topic.md: ['python', 'ml']"
```

### `create_backlink(path_a: str, path_b: str) -> str`

Creates mutual `[[backlinks]]` between two notes. Skips if link already exists.

```python
mcp_client.call_tool("create_backlink", {"path_a": "Notes/A.md", "path_b": "Notes/B.md"})
# "Linked: Notes/A.md ↔ Notes/B.md"
```

### `sync_index() -> str`

Re-runs the full indexing pipeline.

```python
mcp_client.call_tool("sync_index", {})
# "Index sync complete. Check indexer.log for details."
```

### `ask_vault(question: str, top_k: int = 3) -> str`

Ask a question, get an LLM-powered answer from vault content.

```python
mcp_client.call_tool("ask_vault", {"question": "What is machine learning?"})
# "Machine learning is a subset of AI that..."
```

### `tag_notes(query: str, top_k: int = 5) -> str`

Auto-suggest tags for notes matching a query.

```python
mcp_client.call_tool("tag_notes", {"query": "python"})
# "Tagged 2 notes: {'Notes/py.md': ['python', 'programming'], ...}"
```
