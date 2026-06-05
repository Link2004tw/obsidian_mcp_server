"""BM25 keyword search index that works alongside ChromaDB semantic search.

Builds and caches a BM25 index from all chunk documents stored in ChromaDB.
Used by ``search_notes`` to blend keyword and semantic scores for hybrid search.
"""

import re
import threading
import time

from rank_bm25 import BM25Okapi

from . import chroma_store
from .logger import get_logger

log = get_logger(__name__)

# Global BM25 index (lazily built, cached, thread-safe)
_bm25: BM25Okapi | None = None
_bm25_corpus_ids: list[str] | None = None
_bm25_doc_count: int = 0
_bm25_lock = threading.Lock()

# Cooldown to avoid repeated full-index rebuilds (which call get_all_documents)
_bm25_last_rebuild: float = 0.0
_BM25_REBUILD_INTERVAL = 60.0  # seconds


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric tokens, remove short words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def _rebuild_index() -> None:
    """Fetch all documents from ChromaDB and rebuild the BM25 index."""
    global _bm25, _bm25_corpus_ids, _bm25_doc_count

    ids, docs, metadatas = chroma_store.get_all_documents()
    text_docs: list[str] = []
    non_null_ids: list[str] = []

    for i, d in enumerate(docs):
        if d is not None:
            title = metadatas[i].get("title", "") if i < len(metadatas) else ""
            text = f"{title}: {d}" if title else d
            text_docs.append(text)
            non_null_ids.append(ids[i])

    if not text_docs:
        _bm25 = None
        _bm25_corpus_ids = []
        _bm25_doc_count = 0
        log.info("BM25 index: empty corpus")
        return

    tokenized = [_tokenize(d) for d in text_docs]
    _bm25 = BM25Okapi(tokenized)
    _bm25_corpus_ids = non_null_ids
    _bm25_doc_count = len(non_null_ids)
    log.info(f"BM25 index rebuilt — {len(non_null_ids)} documents, {sum(len(t) for t in tokenized)} total tokens")


def ensure_index() -> None:
    """Rebuild the BM25 index if it is stale or missing (with cooldown)."""
    global _bm25_last_rebuild
    with _bm25_lock:
        current_count = chroma_store.count()
        if _bm25 is None or current_count != _bm25_doc_count:
            _rebuild_index()
            _bm25_last_rebuild = time.time()


def get_document_id(path: str, chunk_idx: int) -> str:
    """Return the stable ChromaDB document ID for a given note path + chunk."""
    return f"{path}::chunk_{chunk_idx}"


def search(query: str, n: int = 20) -> list[dict]:
    """Run BM25 keyword search against the cached index.

    Returns up to *n* results sorted by BM25 score (highest first).
    Each result dict matches the ``chroma_store.query()`` shape so it can
    be blended seamlessly::

        {
            "id": str,
            "metadata": dict,
            "distance": None,   # not used for keyword
            "document": str | None,
        }

    BM25 scores are stored under ``"bm25_score"`` in the result dict;
    callers should normalise them before blending with semantic scores.
    """
    with _bm25_lock:
        global _bm25_last_rebuild
        current_count = chroma_store.count()
        needs_rebuild = _bm25 is None or current_count != _bm25_doc_count
        is_on_cooldown = (time.time() - _bm25_last_rebuild) < _BM25_REBUILD_INTERVAL
        if needs_rebuild and not is_on_cooldown:
            _rebuild_index()
            _bm25_last_rebuild = time.time()
        elif needs_rebuild and is_on_cooldown:
            log.debug("BM25 rebuild deferred — on cooldown")
        if _bm25 is None or not _bm25_corpus_ids:
            return []
        corpus_ids = list(_bm25_corpus_ids)
        bm25 = _bm25

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)

    # Get metadata for the top-n results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    top_ids = [corpus_ids[i] for i in top_indices]

    # Fetch metadata + docs from ChromaDB for these IDs
    metadatas, documents = chroma_store.get_metadata_by_ids(top_ids)

    results = []
    for i, idx in enumerate(top_indices):
        doc_id = corpus_ids[idx]
        results.append({
            "id": doc_id,
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "distance": None,
            "document": documents[i] if i < len(documents) else None,
            "bm25_score": float(scores[idx]),
        })
    return results


def normalise_scores(results: list[dict]) -> list[dict]:
    """Min-max normalise ``bm25_score`` values to 0..1 in place.

    Returns the same list for chaining.
    """
    scores = [r["bm25_score"] for r in results]
    if not scores:
        return results
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        for r in results:
            r["bm25_score"] = 1.0
    else:
        for r in results:
            r["bm25_score"] = (r["bm25_score"] - mn) / (mx - mn)
    return results
