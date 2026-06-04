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
vault_path: str | None      # VAULT_PATH (required for file watcher)
data_dir: str               # Data storage root, default "data"
EXCLUDE_PATTERNS: list[str] # Hardcoded exclusion patterns
chunk_size: int             # Words per chunk, default 500
chunk_overlap: int          # Word overlap between chunks, default 100
read_workers: int           # READ_WORKERS, default 6 (parallel note readers)
llm_chat_concurrency: int   # LLM_CHAT_CONCURRENCY, default 2 (max concurrent LLM calls)
expand_cache_ttl: int       # EXPAND_CACHE_TTL, default 3600 (seconds, TTL for query expansion cache)
```

---

## obsidian_client.py

### `list_all_notes() -> list[str]`

Returns a flat list of all `.md` file paths in the Obsidian vault. Recursively walks the vault tree via the REST API. Skips entries matching `EXCLUDE_PATTERNS`.

```python
from obsidian_ai import obsidian_client
notes = obsidian_client.list_all_notes()
# ["Courses/notes/Topic.md", "Journal/2024-01-01.md", ...]
```

### `list_folder(folder_path: str) -> list[str]`

Returns entries directly inside a specific folder (non-recursive — does not descend into subdirectories).
Includes both `.md` files and subdirectory names (with trailing `/`).

```python
notes = obsidian_client.list_folder("Projects")
# ["Projects/active.md", "Projects/archive/", "Projects/notes.md"]
```

Raises the same HTTP errors as `get_note()` if the folder doesn't exist.

### `list_folder_deep(folder_path: str) -> list[str]`

Returns a list of `.md` file paths within a specific folder (recursive — traverses all subdirectories).

```python
notes = obsidian_client.list_folder_deep("Projects")
# ["Projects/active.md", "Projects/archive/old.md", ...]
```

### `list_dir(path: str) -> list[dict]`

Low-level REST API call that returns raw directory entries (with `name`, `type`, `uri` fields). Used internally by `list_folder` and `list_all_notes`.

```python
entries = obsidian_client.list_dir("Projects")
# [{"name": "active.md", "type": "file", "uri": "..."}, ...]
```

### `get_note(path: str) -> str`

Fetches the full content of a note by its vault path. Auto-appends `.md` extension if missing.

```python
content = obsidian_client.get_note("Courses/notes/Topic.md")
# "# Topic\n\nThis is the note content..."
```

**Data directory:** Cache files (`note_paths.json`, `title_to_path.json`) are stored in `config.data_dir`.

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

All Ollama requests use `_request_with_retry()` — retries up to 3 times with exponential backoff (`2s → 4s → 8s`) on:
- `ReadTimeout` — server took too long to respond
- `ConnectionError` — server unreachable
- `HTTPError` with status 429 (rate limit), 502 (bad gateway), 503 (service unavailable)

### `embed(text: str) -> list[float]`

Converts text to an embedding vector via Ollama. Length depends on the model (768 for `nomic-embed-text`).

```python
from obsidian_ai import llm_client
vector = llm_client.embed("What is machine learning?")
# [0.665, 0.270, -4.427, ...]
```

**Caching:** Embeddings are cached in memory by text hash to avoid redundant API calls.

**Raises:** `requests.HTTPError` if Ollama fails after all retries.

### `batch_embed(texts: list[str]) -> list[list[float]]`

Embed multiple texts in a single Ollama API call via ``/api/embed``. Checks the persistent cache first; only uncached texts are sent. Falls back to sequential ``embed()`` if the batch endpoint is unavailable (Ollama < 0.3).

```python
vectors = llm_client.batch_embed(["text one", "text two"])
# [[0.665, ...], [0.270, ...]]
```

**Indexer uses this internally to embed all chunks of a note in one API call.**

### `chat(messages: list[dict], model: str = None, think: bool = True) -> str`

Send a chat completion request to Ollama and return the response text. Uses same retry logic as `embed()`.

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

### `truncate_to_budget(text: str, max_words: int = MAX_CONTEXT_WORDS) -> str`

Truncate text to a word budget, appending `[truncated]` if cut.

```python
long_text = "word " * 5000
short = llm_client.truncate_to_budget(long_text, max_words=1000)
# "word word ... (1000 words) [truncated]"
```

### `clear_embed_cache() -> None`

Clear the in-memory embedding cache.

### `embed_cache_info() -> dict`

Return stats about the embedding cache (`size`, `hits`, `misses`).

### `switch_embed_model(model_name: str) -> None`

Switch the embedding model at runtime. Clears the embedding cache and updates `config.ollama_embed_model`. Does NOT verify the model exists in Ollama — caller should verify first.

---

## chroma_store.py

### `init(path: str = None) -> None`

Initializes or re-initializes the ChromaDB persistent client and the `"notes"` collection. Defaults to `config.chroma_path` if no path is given.

### `reset_collection() -> None`

Deletes all documents from the ChromaDB collection. Used before re-indexing after an embedding model switch.

### `upsert(path: str, chunk_idx: int, embedding: list[float], metadata: dict, document: str = None) -> None`

Inserts or updates a single chunk's embedding in ChromaDB.

- **Document ID:** `{path}::chunk_{chunk_idx}`
- **Metadata fields:** `path`, `title`, `chunk`, `word_count`, `heading`, `tags_str`, `links_str`, `entities_str`, `summary`, `mtime`, `fm_*`

```python
chroma_store.upsert(
    path="Courses/notes/Topic.md",
    chunk_idx=0,
    embedding=[0.665, 0.270, ...],
    metadata={"path": "Courses/notes/Topic.md", "title": "Topic", "chunk": 0, "word_count": 500}
)
```

### `delete_by_path(path: str) -> None`

Deletes all chunks belonging to a given note path. Used before re-indexing to wipe stale chunks.

```python
chroma_store.delete_by_path("Courses/notes/Topic.md")
```

### `get_by_path(path: str) -> list[dict]`

Returns all metadata entries for a given note path.

```python
entries = chroma_store.get_by_path("Courses/notes/Topic.md")
# [{"path": "Courses/notes/Topic.md", "title": "Topic", "chunk": 0, "word_count": 500}]
```

### `get_by_title(title: str) -> list[dict]`

Looks up notes by their title (filename without `.md` extension). Returns deduplicated metadata dicts (one per unique path).

```python
entries = chroma_store.get_by_title("README")
# [{"path": "README.md", "title": "README", ...}]
```

### `count() -> int`

Returns the total number of indexed chunks in ChromaDB.

```python
total = chroma_store.count()
# 157
```

### `get_all_documents() -> tuple[list[str], list[str | None], list[dict]]`

Returns a 3-tuple of `(ids, documents, metadatas)` for every entry in the index. Used by `entity_store.rebuild()`.

### `get_all_embeddings() -> dict[str, list[float]]`

Returns a dict mapping document IDs to their embedding vectors. Used by the clustering module.

```python
embs = chroma_store.get_all_embeddings()
# {"path::chunk_0": [0.665, 0.270, ...], ...}
```

### `get_metadata_by_ids(ids: list[str]) -> tuple[list[dict], list[str | None]]`

Fetches metadata and documents for specific ChromaDB document IDs.

### `query(embedding: list[float], n: int = 5, where: dict = None) -> list[dict]`

Semantic search. Returns the top-k closest chunks.

```python
results = chroma_store.query(embedding=[0.665, 0.270, ...], n=5)
# [
#   {"id": "path::chunk_0", "metadata": {...}, "distance": 0.42, "document": "..."},
#   ...
# ]
```

Each result contains:
- `id` — ChromaDB document ID (`{path}::chunk_{N}`)
- `metadata` — dict with `path`, `title`, `chunk`, `word_count`, `summary`, `entities_str`, etc.
- `distance` — cosine distance (lower = more similar)
- `document` — the chunk text content

### `find_duplicate_notes(threshold: float = 0.9, n: int = 20) -> list[dict]`

Find near-duplicate notes by comparing first-chunk embeddings via cosine distance. Returns up to `n` pairs with similarity scores, sorted descending.

```python
dupes = chroma_store.find_duplicate_notes(threshold=0.85)
# [{"path_a": "Notes/A.md", "path_b": "Notes/B.md", "similarity": 0.92}, ...]
```

### `get_index_stats() -> dict`

Returns index statistics (`total_chunks`, `unique_notes`).

### `search_by_tags(tags: list[str], n: int = 20) -> list[dict]`

Find notes containing ALL specified tags via client-side filtering (workaround for ChromaDB `$contains` bug). Returns deduplicated results with snippets.

```python
results = chroma_store.search_by_tags(["python", "ml"], n=5)
# [{"path": "Notes/topic.md", "title": "topic", "tags": ["python", "ml"], "snippet": "..."}]
```

---

## indexer.py

### Constants & Flags

```python
SKIP_MIN_TOKENS = 20      # Notes under this word count are skipped
CHUNK_SIZE = 500           # Words per chunk
CHUNK_OVERLAP = 100        # Word overlap between chunks
SKIP_ENTITIES = False      # Set True to skip entity extraction
SKIP_SUMMARIES = False     # Set True to skip summary generation
```

### `chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]`

Splits text into overlapping word-based chunks.

```python
from obsidian_ai.indexer import chunk_text
chunks = chunk_text("word " * 1200, size=500, overlap=100)
# ["word word ... (500 words)", "word word ... (500 words)", "word word ... (300 words)"]
```

### `split_by_headings(text: str) -> list[tuple[str, str]]`

Split note content by markdown headings. Returns `[(heading_text, section_content), ...]`. The first section (before any heading) has an empty string heading.

### `chunk_text_heading_aware(text: str, size: int = 500, overlap: int = 100) -> list[tuple[str, str]]`

Heading-aware chunking. Splits by headings first, then chunks large sections. Returns `[(heading, chunk_text), ...]`.

### `run_index() -> None`

Main indexing pipeline. Fetches all notes, chunks them, extracts entities, generates summaries, embeds, and stores in ChromaDB. Prints summary stats and logs errors to `indexer.log`.

**Optimizations:**
- Notes are read in parallel (``READ_WORKERS`` workers, default 6)
- Entity extraction and summary generation use a single combined LLM call per note
- LLM chat calls are limited to ``LLM_CHAT_CONCURRENCY`` concurrent calls (default 2)
- All chunks of a note are embedded in a single Ollama ``/api/embed`` batch call
- Per-note wiki-links are cached during graph update and reused in indexing
- ``delete_by_path`` is skipped for first-time notes
- Mtime map is stored separately from ChromaDB (avoids expensive full-scan)

```python
from obsidian_ai.indexer import run_index
run_index()
# Done. Indexed: 42, Skipped: 5, Failed: 0
```

### `_index_note(path: str, content: str = None) -> bool`

Index a single note. Used by the file watcher for incremental updates. Extracts entities and generates summary if not skipped.

```python
from obsidian_ai.indexer import _index_note
_index_note("Notes/topic.md")
# True if successful
```

### `_delete_note(path: str) -> bool`

Delete a note from the index. Used by the file watcher on file deletion.

```python
from obsidian_ai.indexer import _delete_note
_delete_note("Notes/topic.md")
# True if successful
```

### `watch() -> None`

Start the file watcher daemon. Monitors the vault directory for changes and automatically indexes/deletes notes.

```python
from obsidian_ai.indexer import watch
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
from obsidian_ai.indexer import add_tags_to_note
add_tags_to_note("Notes/topic.md", ["python", "ml"])
```

**Behavior:**
- Parses existing YAML frontmatter (or creates new)
- Appends new tags (no duplicates)
- Converts string `tags` field to list if needed

---

## pipelines.py

LLM-powered pipelines for querying and tagging notes.

### `query(ask: str, top_k: int = 3, ...) -> str`

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
| `use_graph` | `bool` | `False` | Expand via wiki-link graph |
| `graph_depth` | `int` | `1` | Max hops for graph expansion |
| `use_entities` | `bool` | `False` | Expand via entity lookup |
| `keyword_weight` | `float` | `0.0` | BM25 keyword blend |
| `expand_query` | `bool` | `False` | LLM query expansion |
| `entity_types` | `list[str] \| None` | `None` | Filter entity types |
| `min_similarity` | `float \| None` | `None` | Min similarity threshold |
| `expand_entities` | `bool` | `False` | Follow entity relationship edges |
| `use_summaries` | `bool` | `False` | Include summary-embedding results |
| `summary_threshold` | `float` | `0.7` | Min similarity for summary results |
| `auto_weights` | `bool` | `False` | Auto-detect intent, adjust weights |
| `auto_rewrite` | `bool` | `False` | Rewrite query using vault terminology |

### `tag_notes(ask: str, top_k: int = 5) -> str`

Search notes matching a query and auto-suggest tags via LLM.

```python
result = pipelines.tag_notes("machine learning")
# "Tagged 3 notes: {'Notes/nn.md': ['neural-network', 'deep-learning'], ...}"
```

### `summarize_topic(topic: str, top_k: int = 5, use_graph=True, use_entities=True, keyword_weight=0.0, expand_query=False, expand_entities=False, use_summaries=False, summary_threshold=0.7, auto_weights=False, auto_rewrite=False) -> str`

Search all notes related to a topic and return an LLM-generated consolidated summary. Supports auto-rewrite and auto-weight detection.

```python
summary = pipelines.summarize_topic("machine learning")
# "Your vault covers neural networks, decision trees, and reinforcement learning..."
```

### `expand_query(query: str) -> str`

Use LLM to generate alternative query phrasings for broader search. Results are cached with a configurable TTL (`EXPAND_CACHE_TTL`) and persisted to `data/expand_cache.json`.

```python
expanded = pipelines.expand_query("python libraries")
# "python libraries python packages python modules pip conda"
```

### `_rewrite_query(query: str) -> str`

Rewrites a user query using known vault terminology (entity names, note titles). Uses `REWRITE_SYSTEM` prompt. Called when `auto_rewrite=True`.

### `_get_vault_terminology() -> str`

Collects entity names and note titles from the vault to provide context for query rewriting.

### `route_query(query: str, system: str = AGENT_SYSTEM, top_k: int = 5) -> str`

Route a free-form query to the appropriate tool automatically. The LLM decides which tool to call (search, read, graph, etc.) based on the `AGENT_SYSTEM` prompt.

```python
result = pipelines.route_query("Find notes about neural networks and show me related topics")
```

### `extract_entities(text: str, path: str) -> list[dict]`

Extract entities from text using the LLM. Cached per content hash. Returns `[{"name": "...", "type": "...", "confidence": 0.0-1.0}]`.

```python
entities = pipelines.extract_entities("Python is a programming language created by Guido van Rossum.", "Notes/lang.md")
# [{"name": "Python", "type": "Technology", "confidence": 0.95}, {"name": "Guido van Rossum", "type": "Person", "confidence": 0.9}]
```

---

## entity_store.py

Persistent entity inverted index. Data is stored in `data/entities.json` alongside the ChromaDB directory.

### `EntityStore`

```python
from obsidian_ai.entity_store import EntityStore
store = EntityStore()
```

### `add(name: str, type: str, confidence: float, path: str, chunk_idx: int, context: str) -> None`

Add or update an entity mention. Deduplicates by casefolded name — subsequent mentions update confidence, type, aliases, and path list.

```python
store.add("Python", "Technology", 0.95, "Notes/lang.md", 0, "Python is a programming language...")
```

### `search(query: str, type: str = None, n: int = 10) -> list[dict]`

Search entities by name prefix (case-insensitive). Optionally filter by type. Returns top-n records sorted by confidence.

```python
results = store.search("python", n=5)
# [{"name": "Python", "canonical": "Python", "type": "Technology", "confidence": 0.95, ...}]
```

### `search_by_type(entity_type: str, n: int = 50) -> list[dict]`

Return all entities of a given type, sorted by confidence descending.

```python
people = store.search_by_type("Person", n=10)
```

### `get_note_entities(path: str) -> list[dict]`

Return all entities found in a specific note.

```python
entities = store.get_note_entities("Notes/lang.md")
# [{"entity_name": "Python", "entity_type": "Technology", "confidence": 0.95}]
```

### `clear() -> None`

Remove all records from the in-memory store.

### `rebuild() -> None`

Rebuild the entity index from ChromaDB metadata. Walks all chunks, parses `entities_str`, and repopulates the store. Uses lazy import to avoid circular dependency.

### `stats() -> dict`

Return index statistics: total unique entities, entity count, and per-type breakdown.

### `entity_types() -> set[str]`

Return the set of entity type labels present in the index.

### Module-level convenience functions

```python
from obsidian_ai import entity_store
entity_store.add(name, type, confidence, path, chunk_idx, context)
entity_store.search(query, type=None, n=10)
entity_store.search_by_type(entity_type, n=50)
entity_store.get_note_entities(path)
entity_store.clear()
entity_store.rebuild()
entity_store.stats()
entity_store.entity_types()
```

These wrap a global `EntityStore` singleton (`_store`).

---

## graph_store.py

In-memory wiki-link graph built during indexing. Uses a dict-based adjacency list.

### `GraphStore`

```python
from obsidian_ai.graph_store import GraphStore
store = GraphStore()
```

### `add_edge(source: str, target: str) -> None`

Add a directed edge from `source` to `target` note path.

### `remove_node(path: str) -> None`

Remove a node and all its edges. Also removes entity edges referencing this path.

### `rename_node(old_path: str, new_path: str) -> None`

Rename a node and update all edges referencing it.

### `register_title(path: str, title: str) -> None`

Associate a note path with its display title.

### `get_backlinks(path: str) -> list[dict]`

Return all notes linking TO the given note. Returns `[{"path": "...", "title": "..."}]`.

### `get_linked_notes(path: str) -> list[dict]`

Return all notes the given note links TO. Returns `[{"path": "...", "title": "..."}]`.

### `bfs(start: str, max_depth: int = 2) -> list[dict]`

BFS traversal from a seed note up to N hops. Returns reachable notes with path traces.

```python
results = store.bfs("Notes/A.md", max_depth=2)
# [{"path": "Notes/B.md", "title": "B", "depth": 1, "trace": ["Notes/A.md", "Notes/B.md"]}, ...]
```

### `get_broken_links() -> list[dict]`

Find wiki-links that don't resolve to any existing note.

### `get_orphans() -> list[str]`

Find notes with no incoming or outgoing wiki-links.

### `get_stats() -> dict`

Return graph statistics: `nodes`, `edges`, `avg_degree`, `isolated` notes, `hubs` (top 5 by degree).

### `get_communities() -> dict[str, list[str]]`

Detect communities using label propagation. Returns `{community_id: [note_paths]}`.

### `to_dict() -> dict`

Serialize graph to dict for export.

### `to_dot() -> str`

Export graph in Graphviz DOT format.

### `add_entity_edge(entity_id: str, note_path: str) -> None`

Add an edge from an entity node to a note that mentions it.

### `remove_entity_edges(note_path: str) -> None`

Remove all entity edges referencing a given note path.

### Module-level convenience functions

```python
from obsidian_ai import graph_store
graph_store.rebuild()
graph_store.get_backlinks(path)
graph_store.get_linked_notes(path)
graph_store.bfs(start, max_depth=2)
graph_store.get_broken_links()
graph_store.get_orphans()
graph_store.get_stats()
graph_store.get_communities()
graph_store.export_graph(format="json")
```

---

## clustering.py

Semantic clustering module for grouping notes by meaning.

### `get_clusters(force_recompute: bool = False, similarity_threshold: float = 0.6) -> list[dict]`

Returns semantic clusters of notes based on embedding cosine similarity. Connected-components clustering (no scikit-learn dependency).

**Parameters:**
- `force_recompute` — if True, ignore cached results and re-cluster
- `similarity_threshold` — minimum cosine similarity (0–1) for two notes to be connected (default 0.6)

**Returns:** List of clusters, each with:
- `label` — auto-generated descriptive label from the most central note
- `notes` — list of note paths in the cluster
- `size` — number of notes
- `central_note` — path of the most central (highest-degree) note

```python
from obsidian_ai.clustering import get_clusters
clusters = get_clusters(similarity_threshold=0.7)
# [{"label": "Machine Learning", "notes": [...], "size": 5, "central_note": "..."}, ...]
```

Results are cached with TTL and persisted to `data/clusters.json`.

---

## frontmatter.py

YAML frontmatter parsing and manipulation utilities.

### `parse(text: str) -> tuple[dict, str, int]`

Parse YAML frontmatter from note text. Returns `(metadata_dict, body_text, end_line)`.

```python
from obsidian_ai.frontmatter import parse
meta, body, end = parse("---\ntags: [python]\n---\n# Content")
# meta = {"tags": ["python"]}, body = "# Content"
```

### `build(metadata: dict, body: str) -> str`

Build full note text from metadata dict and body.

### `add_tags_to_meta(metadata: dict, tags: list[str]) -> dict`

Add tags to a metadata dict (no duplicates, converts string to list).

### `add_tags(text: str, tags: list[str]) -> str`

High-level function: parse, add tags, rebuild. Creates frontmatter if absent.

```python
result = frontmatter.add_tags("# Note", ["python"])
# "---\ntags:\n- python\n---\n# Note"
```

### `remove_tags(text: str, tags: list[str]) -> str`

Remove specific tags from frontmatter.

### `set_tags(text: str, tags: list[str]) -> str`

Replace all tags with the given list.

---

## wiki_links.py

Wiki-link parsing utilities.

### `extract_wiki_links(text: str) -> list[str]`

Extract all `[[wikilinks]]` from note text. Returns deduplicated link targets (first-seen order). Ignores image embeds (`![[image.png]]`) and empty targets (`[[]]`).

```python
from obsidian_ai.wiki_links import extract_wiki_links
links = extract_wiki_links("See [[Note A]] and [[Note B|display text]]")
# ["Note A", "Note B"]
```

### `normalize_wiki_link_target(target: str) -> str`

Normalize a wiki-link target: strips display text after `|`, strips section anchors after `#`, normalizes folder path separators.

---

## mcp_server.py

FastMCP server exposing **67 vault tools**. Tool implementations are organized in `tools/` submodules (`search.py`, `notes.py`, `graph.py`, `todos.py`, `misc.py`) and registered via `register_all(mcp)`. All path parameters are auto-normalized (absolute or vault-relative accepted). Run with `python -m obsidian_ai.mcp_server`. See [MCP Server](mcp_server.md) for detailed tool documentation.

### Tools by category

**Search & Retrieval:** `search_notes`, `batch_search`, `composite_search`, `retrieve_notes`, `find_duplicate_notes`, `search_by_tags`, `get_subject`, `search_entities`

**Read & Write:** `read_note`, `write_note`, `list_all_notes`, `list_folder`, `list_folder_deep`, `read_note_by_title`

**Tags:** `add_tags`, `remove_tags`, `set_tags`, `batch_tag_notes`, `tag_notes`

**Graph:** `create_backlink`, `get_backlinks`, `get_linked_notes`, `get_broken_links`, `get_orphan_notes`, `get_graph_stats`, `get_communities`, `get_note_community`, `multi_hop_traversal`, `related_notes`, `export_graph`, `get_shortest_path`

**LLM:** `ask_vault`, `ask_agent`, `summarize_topic`, `tag_notes`

**Entities:** `search_entities`, `get_note_entities`, `get_entity_types`, `get_entity_aliases`, `merge_entities`, `entity_timeline`, `related_entities`, `set_ranking_weights`, `get_ranking_weights`

**Clustering:** `get_clusters`

**Health:** `health_check`

**Index:** `sync_index`, `get_index_stats`, `switch_embedding_model`

**Todos:** `get_todos`, `add_todo`, `complete_todo`, `update_todo`, `delete_todo`, `sync_todos`, `get_todo_stats`, `ensure_todo_file`, `get_todos_by_priority`, `add_todo_from_natural_language`, `suggest_task_priority`, `suggest_due_date`, `suggest_task_splitting`, `get_overdue_summary`, `estimate_completion_date`, `get_todos_for_note`, `get_notes_for_todo`, `link_todo_to_notes`, `ask_vault_about_todo`, `ask_vault_about_todos`

---

## dev/tasks.md

Internal project task tracking. Lists all tasks, their status, and progress.
