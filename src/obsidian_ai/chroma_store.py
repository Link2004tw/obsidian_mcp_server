import chromadb

from . import config

_client = chromadb.PersistentClient(path=config.chroma_path)
_collection = _client.get_or_create_collection("notes")


def upsert(path: str, chunk_idx: int, embedding: list[float], metadata: dict, document: str | None = None) -> None:
    doc_id = f"{path}::chunk_{chunk_idx}"
    docs = [document] if document else None
    _collection.upsert(ids=[doc_id], embeddings=[embedding], metadatas=[metadata], documents=docs)  # type: ignore[arg-type]


def delete_by_path(path: str) -> None:
    results = _collection.get(where={"path": path})
    if results["ids"]:
        _collection.delete(ids=results["ids"])


def get_by_path(path: str) -> list[dict]:
    results = _collection.get(where={"path": path})
    metadatas = results["metadatas"]
    if metadatas is None:
        return []
    return metadatas  # type: ignore[return-value]


def get_by_title(title: str) -> list[dict]:
    """Look up notes by title (basename without extension). Returns metadata dicts."""
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
    return _collection.count()


def get_all_documents() -> tuple[list[str], list[str | None], list[dict]]:
    """Return all document IDs, text content, and metadata from the index.

    Returns:
        (ids, documents, metadatas) where ids are the ChromaDB document IDs,
        documents are the stored chunk texts (may be None for old entries),
        and metadatas are the metadata dicts for each entry.
    """
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
    all_data = _collection.get(ids=ids, include=["metadatas", "documents"])  # type: ignore[arg-type]
    metadatas: list[dict] = all_data["metadatas"] if all_data["metadatas"] else []  # type: ignore[assignment]
    docs: list[str | None] = all_data["documents"] if all_data["documents"] else []  # type: ignore[assignment]
    return metadatas, docs


def get_index_stats() -> dict:
    """Return index statistics as a dict."""
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
    ``tags_str`` metadata field. Uses ChromaDB's ``$contains`` to match.

    Returns deduplicated results (one per note path) with metadata + snippet.
    """
    if not tags:
        return []
    conditions = [{"tags_str": {"$contains": f",{tag},"}} for tag in tags]
    where: dict = {"$and": conditions} if len(conditions) > 1 else conditions[0]

    raw = _collection.get(where=where, include=["metadatas", "documents"])  # type: ignore[arg-type]
    metadatas: list[dict] = raw["metadatas"] if raw["metadatas"] else []  # type: ignore[assignment]
    documents: list[str] = raw["documents"] if raw["documents"] else []  # type: ignore[assignment]

    # Deduplicate by path, picking first chunk as snippet
    seen = set()
    unique = []
    for i, m in enumerate(metadatas):
        p = m["path"]
        if p not in seen:
            seen.add(p)
            result = dict(m)
            if i < len(documents) and documents[i]:
                doc = documents[i]
                snippet = doc[:300] + ("..." if len(doc) > 300 else "")
                result["snippet"] = snippet
            else:
                result["snippet"] = ""
            unique.append(result)
            if len(unique) >= n:
                break
    return unique


def dedup_paths(results: list[dict]) -> list[tuple[str, str]]:
    """Deduplicate query results by path. Returns [(path, title), ...]."""
    seen = {}
    for r in results:
        path = r["metadata"]["path"]
        if path not in seen:
            seen[path] = r["metadata"].get("title", path)
    return list(seen.items())


def query(embedding: list[float], n: int = 5, where: dict | None = None) -> list[dict]:
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
