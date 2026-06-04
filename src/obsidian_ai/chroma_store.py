import chromadb

from . import config
from .logger import get_logger

log = get_logger("obsidian_ai.chroma_store")

_client = None
_collection = None


def _ensure_init():
    """Lazily initialize the ChromaDB client on first use."""
    global _client, _collection
    if _collection is not None:
        return
    init()


def reset_collection() -> None:
    """Delete all data from the collection and re-create it."""
    _ensure_init()
    _collection.delete(ids=_collection.get()["ids"])
    log.info("ChromaDB collection reset")


_HNSW_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 100,
    "hnsw:M": 8,
    "hnsw:search_ef": 100,
}


def init(path: str | None = None) -> None:
    """Initialize or reinitialize the ChromaDB client.

    Args:
        path: directory for the persistent ChromaDB store.
              Defaults to ``config.chroma_path``.
    """
    global _client, _collection
    _client = chromadb.PersistentClient(path=path or config.chroma_path)
    _collection = _client.get_or_create_collection("notes", metadata=_HNSW_METADATA)


def upsert(path: str, chunk_idx: int, embedding: list[float], metadata: dict, document: str | None = None) -> None:
    _ensure_init()
    doc_id = f"{path}::chunk_{chunk_idx}"
    docs = [document] if document else None
    _collection.upsert(ids=[doc_id], embeddings=[embedding], metadatas=[metadata], documents=docs)  # type: ignore[arg-type]


def delete_by_path(path: str) -> None:
    _ensure_init()
    results = _collection.get(where={"path": path})
    if results["ids"]:
        _collection.delete(ids=results["ids"])


def get_by_path(path: str) -> list[dict]:
    _ensure_init()
    results = _collection.get(where={"path": path})
    metadatas = results["metadatas"]
    if metadatas is None:
        return []
    return metadatas  # type: ignore[return-value]


def get_chunks_by_path(path: str) -> tuple[list[str], list[dict], list[str | None]]:
    """Return (ids, metadatas, documents) for all chunks of a note."""
    _ensure_init()
    results = _collection.get(where={"path": path}, include=["documents", "metadatas"])  # type: ignore[arg-type]
    ids: list[str] = results["ids"] if results["ids"] else []
    metadatas: list[dict] = results["metadatas"] if results["metadatas"] else []  # type: ignore[assignment]
    docs: list[str | None] = results["documents"] if results["documents"] else []  # type: ignore[assignment]
    return ids, metadatas, docs


def get_by_title(title: str) -> list[dict]:
    """Look up notes by title (basename without extension). Returns metadata dicts."""
    _ensure_init()
    results = _collection.get(where={"title": title})
    raw = results["metadatas"]
    metadatas: list[dict] = raw if raw is not None else []  # type: ignore[assignment]
    seen = set()
    unique = []
    for m in metadatas:
        p = m["path"]
        if p not in seen:
            seen.add(p)
            unique.append(m)
    return unique


def count() -> int:
    _ensure_init()
    return _collection.count()


def get_all_documents() -> tuple[list[str], list[str | None], list[dict]]:
    """Return all document IDs, text content, and metadata from the index.

    Returns:
        (ids, documents, metadatas) where ids are the ChromaDB document IDs,
        documents are the stored chunk texts (may be None for old entries),
        and metadatas are the metadata dicts for each entry.
    """
    _ensure_init()
    all_data = _collection.get(include=["documents", "metadatas"])  # type: ignore[arg-type]
    ids: list[str] = all_data["ids"] if all_data["ids"] else []
    docs: list[str | None] = all_data["documents"] if all_data["documents"] else []  # type: ignore[assignment]
    metadatas: list[dict] = all_data["metadatas"] if all_data["metadatas"] else []  # type: ignore[assignment]
    return ids, docs, metadatas


def get_metadata_by_ids(ids: list[str]) -> tuple[list[dict], list[str | None]]:
    """Fetch metadata and documents for specific document IDs.

    Returns:
        (metadatas, documents) aligned with the requested IDs.
    """
    _ensure_init()
    all_data = _collection.get(ids=ids, include=["metadatas", "documents"])  # type: ignore[arg-type]
    metadatas: list[dict] = all_data["metadatas"] if all_data["metadatas"] else []  # type: ignore[assignment]
    docs: list[str | None] = all_data["documents"] if all_data["documents"] else []  # type: ignore[assignment]
    return metadatas, docs


def find_duplicate_notes(threshold: float = 0.9, n: int = 20) -> list[dict]:
    """Find near-duplicate notes via embedding similarity.

    Compares the first chunk embedding of each note against all others.
    Returns list of ``{"path_a": str, "path_b": str, "similarity": float}``
    sorted by similarity descending, limited to ``n`` pairs.
    """
    _ensure_init()
    all_data = _collection.get(include=["embeddings", "metadatas"])
    ids = all_data["ids"]
    embeddings = all_data["embeddings"]
    metadatas = all_data["metadatas"]

    if not ids or not embeddings:
        return []

    # First chunk embedding per note
    note_emb: dict[str, list[float]] = {}
    for meta, emb in zip(metadatas, embeddings, strict=False):
        path = meta["path"]
        if path not in note_emb:
            note_emb[path] = emb

    note_paths = list(note_emb.keys())

    pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()

    for path in note_paths:
        emb = note_emb[path]
        results = _collection.query(
            query_embeddings=[emb],
            n_results=min(n * 3, len(note_paths)),
        )
        for j, doc_id in enumerate(results["ids"][0]):
            other_path = doc_id.split("::")[0]
            if other_path == path:
                continue
            distance = results["distances"][0][j]
            similarity = 1.0 - distance
            if similarity >= threshold:
                pair = tuple(sorted([path, other_path]))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append((pair[0], pair[1], similarity))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return [
        {"path_a": p[0], "path_b": p[1], "similarity": round(p[2], 4)}
        for p in pairs[:n]
    ]


def get_index_stats() -> dict:
    """Return index statistics as a dict."""
    _ensure_init()
    total_chunks = _collection.count()
    # Get all unique note paths
    all_metadatas = _collection.get()["metadatas"] or []
    unique_notes = len({m["path"] for m in all_metadatas}) if all_metadatas else 0
    return {
        "total_chunks": total_chunks,
        "unique_notes": unique_notes,
    }


def search_by_tags(tags: list[str], n: int = 20) -> list[dict]:
    """Search for notes that have ALL of the given tags.

    Tags are stored as a comma-delimited string like ",tag1,tag2," in the
    ``tags_str`` metadata field. Performs client-side filtering as a
    workaround for ChromaDB 1.5.9 ``$contains`` bug.

    Returns deduplicated results (one per note path) with metadata + snippet.
    """
    if not tags:
        return []
    _ensure_init()
    raw = _collection.get(include=["metadatas", "documents"])  # type: ignore[arg-type]
    all_metadatas: list[dict] = raw["metadatas"] if raw["metadatas"] else []  # type: ignore[assignment]
    all_documents: list[str] = raw["documents"] if raw["documents"] else []  # type: ignore[assignment]

    # Client-side tag filtering (workaround for ChromaDB $contains bug)
    seen = set()
    unique = []
    for i, m in enumerate(all_metadatas):
        tags_str = m.get("tags_str", "")
        if not tags_str:
            continue
        if all(f",{tag}," in tags_str for tag in tags):
            p = m["path"]
            if p not in seen:
                seen.add(p)
                result = dict(m)
                if i < len(all_documents) and all_documents[i]:
                    doc = all_documents[i]
                    snippet = doc[:300] + ("..." if len(doc) > 300 else "")
                    result["snippet"] = snippet
                else:
                    result["snippet"] = ""
                unique.append(result)
                if len(unique) >= n:
                    break
    return unique


def _dedup_paths(results: list[dict]) -> list[tuple[str, str]]:
    """Deduplicate query results by path. Returns [(path, title), ...]."""
    seen = {}
    for r in results:
        path = r["metadata"]["path"]
        if path not in seen:
            seen[path] = r["metadata"].get("title", path)
    return list(seen.items())


def get_all_embeddings() -> list[dict]:
    """Return all chunk embeddings with metadata.

    Returns:
        List of {"id", "embedding", "metadata"} for every chunk in the index.
        Embeddings are list[float] from ChromaDB.
    """
    _ensure_init()
    raw = _collection.get(include=["embeddings", "metadatas"])
    ids = raw["ids"] or []
    embeddings = raw["embeddings"]
    metadatas = raw["metadatas"]
    result = []
    for i, doc_id in enumerate(ids):
        result.append({
            "id": doc_id,
            "embedding": embeddings[i] if embeddings else None,
            "metadata": metadatas[i] if metadatas else {},
        })
    return result


def query(embedding: list[float], n: int = 5, where: dict | None = None) -> list[dict]:
    _ensure_init()
    kwargs: dict = {"query_embeddings": [embedding], "n_results": n}
    if where is not None:
        kwargs["where"] = where
    raw = _collection.query(**kwargs)  # type: ignore[arg-type]
    ids = raw["ids"][0]
    metadatas = raw["metadatas"][0] if raw["metadatas"] else []
    distances = raw["distances"][0] if raw["distances"] else []
    documents = raw["documents"][0] if raw["documents"] else []
    items = []
    for i in range(len(ids)):
        items.append({
            "id": ids[i],
            "metadata": metadatas[i] if metadatas else {},
            "distance": distances[i] if distances else None,
            "document": documents[i] if i < len(documents) else None,
        })
    return items
