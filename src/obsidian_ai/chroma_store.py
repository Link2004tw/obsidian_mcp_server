import chromadb
from . import config

_client = chromadb.PersistentClient(path=config.chroma_path)
_collection = _client.get_or_create_collection("notes")


def upsert(path: str, chunk_idx: int, embedding: list[float], metadata: dict) -> None:
    doc_id = f"{path}::chunk_{chunk_idx}"
    _collection.upsert(ids=[doc_id], embeddings=[embedding], metadatas=[metadata])


def delete_by_path(path: str) -> None:
    results = _collection.get(where={"path": path})
    if results["ids"]:
        _collection.delete(ids=results["ids"])


def get_by_path(path: str) -> list[dict]:
    results = _collection.get(where={"path": path})
    return results["metadatas"] if results["ids"] else []


def count() -> int:
    return _collection.count()


def dedup_paths(results: list[dict]) -> list[tuple[str, str]]:
    """Deduplicate query results by path. Returns [(path, title), ...]."""
    seen = {}
    for r in results:
        path = r["metadata"]["path"]
        if path not in seen:
            seen[path] = r["metadata"].get("title", path)
    return list(seen.items())


def query(embedding: list[float], n: int = 5) -> list[dict]:
    results = _collection.query(query_embeddings=[embedding], n_results=n)
    items = []
    for i in range(len(results["ids"][0])):
        items.append({
            "id": results["ids"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results["distances"] else None,
        })
    return items
