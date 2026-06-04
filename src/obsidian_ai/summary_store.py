"""ChromaDB-backed store for note summaries.

Each entry maps a summary embedding to a note path + title, enabling
summary-first retrieval: query summaries before falling back to
chunk-level semantic search.
"""

from . import config, llm_client
from .logger import get_logger

log = get_logger(__name__)

_SUMMARY_COLLECTION = "note_summaries"

_client = None
_collection = None

_HNSW_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 100,
    "hnsw:M": 8,
    "hnsw:search_ef": 100,
}


def ensure_init():
    global _client, _collection
    if _collection is not None:
        return
    import chromadb
    _client = chromadb.PersistentClient(path=config.chroma_path)
    _collection = _client.get_or_create_collection(
        _SUMMARY_COLLECTION, metadata=_HNSW_METADATA
    )


def clear():
    ensure_init()
    _collection.delete(ids=_collection.get()["ids"])
    log.info("Summary store cleared")


def add(path: str, title: str, summary: str) -> None:
    """Embed *summary* and store it in the summary collection.

    Args:
        path: vault-relative note path (used as document ID).
        title: note title (basename without extension).
        summary: the summary text to embed and store.
    """
    if not summary:
        return
    ensure_init()
    embedding = llm_client.embed(summary)
    metadata = {"path": path, "title": title}
    _collection.upsert(
        ids=[path],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[summary],
    )


def delete_by_path(path: str) -> None:
    ensure_init()
    _collection.delete(ids=[path])


def query(query_text: str, n: int = 5) -> list[dict]:
    """Search summaries by semantic similarity to *query_text*.

    Returns results sorted by distance ascending (closest first), each with:
    ``path``, ``title``, ``distance``, ``similarity``, ``summary``.
    """
    ensure_init()
    embedding = llm_client.embed(query_text)
    raw = _collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["metadatas", "documents", "distances"],
    )
    ids = raw["ids"][0] if raw["ids"] else []
    distances = raw["distances"][0] if raw["distances"] else []
    metadatas = raw["metadatas"][0] if raw["metadatas"] else []
    documents = raw["documents"][0] if raw["documents"] else []

    results = []
    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else 1.0
        summary = documents[i] if i < len(documents) else ""
        results.append({
            "path": meta.get("path", ids[i]),
            "title": meta.get("title", ""),
            "distance": dist,
            "similarity": round(1.0 / (1.0 + dist), 4),
            "summary": summary,
        })
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results


def count() -> int:
    ensure_init()
    return _collection.count()
