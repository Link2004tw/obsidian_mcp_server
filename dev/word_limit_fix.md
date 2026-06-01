## Fix Plan: Chunked Indexing

### Step 1 — Update `chroma_store.py` schema

Change the document ID from `path` to `path::chunk_N` so multiple chunks per note can coexist. Keep `path` as a metadata field so you can still group/deduplicate by note later.

```python
id = f"{path}::chunk_{i}"
metadata = { "path": path, "title": title, "chunk": i, "mtime": mtime }
```

Also add a `delete_by_path(path)` method — needed so re-indexing a modified note wipes all its old chunks before reinserting.

---

### Step 2 — Add a `chunk_text()` utility

Simple sliding window, word-based:

```python
def chunk_text(text, size=1000, overlap=200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start+size]))
        start += size - overlap
    return chunks
```

`size=1000` words, `overlap=200` words. Well within the 6000-word safe limit per chunk.

---

### Step 3 — Update `indexer.py`

Replace the current truncation logic with:

1. Call `chunk_text(content)`
2. Embed each chunk separately
3. Upsert each as its own ChromaDB entry
4. Call `delete_by_path(path)` before reinserting on re-index

---

### Step 4 — Update `search_notes` in `mcp_server.py`

After getting top-k results from ChromaDB, deduplicate by `path` — return each note once even if multiple chunks matched. Optionally surface the best-matching chunk's snippet as the preview.

---

### Order

Do **1 → 2 → 3 → 4** in sequence. Each is independently testable. Total scope is small — only touches 3 files.
