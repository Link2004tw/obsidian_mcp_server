"""Unified ranking pipeline combining semantic, entity, graph, and keyword retrieval signals.

Provides a ``Ranker`` class that accepts tunable weights for all four signals,
normalizes scores, and returns note-level results with ``matched_by`` metadata.
Thread-safe for concurrent access from MCP workers.
"""

import os
import threading

from . import chroma_store, llm_client
from .logger import get_logger

log = get_logger(__name__)

SNIPPET_MAX_CHARS = 300

DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.40,
    "entity": 0.30,
    "graph": 0.20,
    "keyword": 0.10,
}


def _truncate_snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _auto_detect_entities(query: str) -> list[dict]:
    """Check if the query contains entity names and return matching results.

    Tries the full query first, then individual words (>2 chars),
    then consecutive word pairs (bigrams) to catch multi-word entities
    like ``"Alice Johnson"`` or ``"ESP32 Reader"``.

    Returns deduplicated entity-store results (path, entity_name, snippet, etc.)
    or an empty list if no match is found.
    """
    from . import entity_store

    seen_paths: set[str] = set()
    results: list[dict] = []

    def _merge(entity_results: list[dict]) -> None:
        for r in entity_results:
            p = r["path"]
            if p not in seen_paths:
                seen_paths.add(p)
                results.append(r)

    # 1. Full query as a single entity name (e.g. "ESP32" or "Alice Johnson")
    _merge(entity_store.search(query.strip()))

    # 2. Individual words (skip very short tokens)
    words = [w.strip(".,!?;:'\"()") for w in query.split()]
    for word in words:
        if len(word) > 2:
            _merge(entity_store.search(word))

    # 3. Consecutive word pairs (bigrams)
    if len(words) > 1:
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 3:
                _merge(entity_store.search(bigram))

    return results


def _expand_entity_names(entity_results: list[dict]) -> list[str]:
    """Expand auto-detected entity results with related entities from the relationship graph.

    For each entity detected in the query, looks up related entities via
    ``entity_relations.get_related()`` and returns any new entity names
    not already in the original result set.

    Args:
        entity_results: output from ``_auto_detect_entities()``.

    Returns:
        A deduplicated list of related entity names (original casing,
        lowercased for dedup).
    """
    from . import entity_relations

    detected_names: set[str] = set()
    for r in entity_results:
        name = r.get("entity_name", "").strip()
        if name:
            detected_names.add(name)

    if not detected_names:
        return []

    seen: set[str] = set()
    expanded: list[str] = []
    for name in detected_names:
        try:
            related = entity_relations.get_related(name, depth=1)
        except Exception:
            continue
        for rel in related:
            ename = rel["entity_name"]
            if ename not in seen and ename.casefold() not in {n.casefold() for n in detected_names}:
                seen.add(ename)
                expanded.append(ename)

    return expanded


class Ranker:
    """Unified ranking pipeline blending semantic, entity, graph, and keyword signals.

    Each retrieval strategy contributes a normalised score (0–1). Scores are
    blended using configurable weights and results are returned as note-level
    entries with ``matched_by`` metadata for transparency.

    Thread-safe: all weight mutations are protected by a lock.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)
        self._lock = threading.Lock()

    # ── Weight management ──────────────────────────────────────────

    @property
    def weights(self) -> dict[str, float]:
        with self._lock:
            return dict(self._weights)

    def set_weights(
        self,
        semantic: float | None = None,
        entity: float | None = None,
        graph: float | None = None,
        keyword: float | None = None,
    ) -> dict[str, float]:
        """Update one or more ranking weights at runtime.

        Returns the new weights dict. Only keys that are not ``None`` are updated.
        """
        with self._lock:
            if semantic is not None:
                self._weights["semantic"] = semantic
            if entity is not None:
                self._weights["entity"] = entity
            if graph is not None:
                self._weights["graph"] = graph
            if keyword is not None:
                self._weights["keyword"] = keyword
            return dict(self._weights)

    def _normalize(self, weights: dict[str, float] | None = None) -> dict[str, float]:
        """Return a copy with inactive (0) signals removed, remaining summing to 1.0."""
        w = dict(self._weights)
        if weights:
            w.update(weights)
        w = {k: v for k, v in w.items() if v > 0}
        total = sum(w.values())
        if total == 0:
            return {"semantic": 1.0}
        return {k: v / total for k, v in w.items()}

    # ── Main search API ────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = 5,
        use_entities: bool = False,
        use_graph: bool = False,
        graph_depth: int = 1,
        weights: dict[str, float] | None = None,
        where: dict | None = None,
        exclude_tags: list[str] | None = None,
        expand_queries: list[str] | None = None,
        min_similarity: float | None = None,
        expand_entities: bool = False,
    ) -> list[dict]:
        """Multi-strategy search returning blended, note-level results.

        Arguments match the combined param space of ``search_notes``,
        ``retrieve_notes``, and ``ask_vault`` for drop-in compatibility.

        Args:
            query: the search query string.
            n: max results to return.
            use_entities: if True, include entity-index results.
            use_graph: if True, include wiki-link graph traversal results.
            graph_depth: max BFS hops when *use_graph* is True.
            weights: optional per-call weight overrides
                (e.g. ``{"semantic": 0.6, "entity": 0.4}``).
            where: ChromaDB ``where`` clause for metadata filtering.
            exclude_tags: tags to exclude from results.
            expand_queries: additional query phrasings for broader semantic search.
            min_similarity: minimum blended score threshold (0–1).
            expand_entities: if True, when entities are auto-detected in the
                query, also search for related entities via the entity
                relationship graph and include their results.

        Returns:
            List of dicts with ``path``, ``title``, ``similarity_score``,
            ``matched_by`` (list of signal names), and ``snippet``.
        """
        from . import entity_store, graph_store, keyword_search

        w = self._normalize(weights)

        # note_scores maps path -> {score, title, snippet, matched_by}
        note_scores: dict[str, dict] = {}

        # ── 1. Semantic search ──────────────────────────────────────
        if w.get("semantic", 0) > 0:
            queries = [query]
            if expand_queries:
                queries.extend(expand_queries)
            for q in queries:
                try:
                    results = chroma_store.query(llm_client.embed(q), n=n * 3, where=where)
                except Exception as exc:
                    log.warning("ranker — semantic query failed: %s", exc)
                    continue
                for r in results:
                    metadata = r["metadata"]
                    path = metadata["path"]
                    if exclude_tags:
                        tags_str = metadata.get("tags_str", "")
                        if any(f",{et}," in tags_str for et in exclude_tags):
                            continue
                    distance = r["distance"]
                    similarity = 1.0 / (1.0 + distance) if distance is not None else 0.0
                    entry = note_scores.get(path)
                    if entry is None:
                        title = os.path.splitext(os.path.basename(path))[0]
                        entry = {"score": 0.0, "title": title, "snippet": "", "matched_by": []}
                        note_scores[path] = entry
                    entry["score"] += similarity * w["semantic"]
                    if "semantic" not in entry["matched_by"]:
                        entry["matched_by"].append("semantic")
                    snippet = r.get("document") or ""
                    if len(snippet) > len(entry["snippet"]):
                        entry["snippet"] = _truncate_snippet(snippet)

        # ── 2. BM25 keyword search ──────────────────────────────────
        if w.get("keyword", 0) > 0:
            try:
                kw_results = keyword_search.search(query, n=n * 3)
                keyword_search.normalise_scores(kw_results)
            except Exception as exc:
                log.warning("ranker — keyword search failed: %s", exc)
                kw_results = []
            for r in kw_results:
                metadata = r["metadata"]
                path = metadata.get("path", "")
                if not path:
                    continue
                if exclude_tags:
                    tags_str = metadata.get("tags_str", "")
                    if any(f",{et}," in tags_str for et in exclude_tags):
                        continue
                kw_score = r["bm25_score"]
                entry = note_scores.get(path)
                if entry is None:
                    title = os.path.splitext(os.path.basename(path))[0]
                    entry = {"score": 0.0, "title": title, "snippet": "", "matched_by": []}
                    note_scores[path] = entry
                entry["score"] += kw_score * w["keyword"]
                if "keyword" not in entry["matched_by"]:
                    entry["matched_by"].append("keyword")
                snippet = r.get("document") or ""
                if len(snippet) > len(entry["snippet"]):
                    entry["snippet"] = _truncate_snippet(snippet)

        # ── 3. Entity lookup (auto-detect + explicit + relationship expansion) ─
        entity_weight = w.get("entity", 0)
        if entity_weight > 0:
            entity_results: list[dict] = []

            # Auto-detect: always check if the query mentions an entity
            try:
                entity_results = _auto_detect_entities(query)
            except Exception as exc:
                log.warning("ranker — auto entity detection failed: %s", exc)

            # Relationship expansion: when entities are found, expand via graph
            expand_names: list[str] = []
            if expand_entities and entity_results:
                try:
                    expand_names = _expand_entity_names(entity_results)
                except Exception as exc:
                    log.warning("ranker — entity expansion failed: %s", exc)

            # Explicit opt-in: also search when use_entities=True
            if use_entities:
                try:
                    explicit = entity_store.search(query)
                    seen = {r["path"] for r in entity_results}
                    for r in explicit:
                        if r["path"] not in seen:
                            entity_results.append(r)
                except Exception as exc:
                    log.warning("ranker — explicit entity search failed: %s", exc)

            # Search for expanded entity names from relationship graph
            if expand_names:
                log.info("ranker — expanded query with related entities: %s", expand_names)
                seen_paths = {r["path"] for r in entity_results}
                for ename in expand_names:
                    try:
                        expanded_results = entity_store.search(ename)
                        for r in expanded_results:
                            if r["path"] not in seen_paths:
                                seen_paths.add(r["path"])
                                entity_results.append(r)
                    except Exception as exc:
                        log.warning("ranker — expansion search for %s failed: %s", ename, exc)

            if entity_results:
                log.info("ranker — auto-detected %d entity results for query: %s",
                         len(entity_results), query)

            for r in entity_results:
                path = r["path"]
                confidence = float(r.get("confidence", 0.5))
                entity_name = r.get("entity_name", "")
                entity_type = r.get("entity_type", "Concept")
                entry = note_scores.get(path)
                if entry is None:
                    title = os.path.splitext(os.path.basename(path))[0]
                    entry = {"score": 0.0, "title": title, "snippet": "", "matched_by": []}
                    note_scores[path] = entry
                entry["score"] += confidence * entity_weight
                if "entity" not in entry["matched_by"]:
                    entry["matched_by"].append("entity")
                snippet = r.get("snippet") or f"[{entity_type}] {entity_name}"
                if len(snippet) > len(entry["snippet"]):
                    entry["snippet"] = _truncate_snippet(snippet)

        if not note_scores:
            return []

        # ── 4. Graph traversal (requires seed paths) ────────────────
        if use_graph and w.get("graph", 0) > 0:
            try:
                seed_paths = list(note_scores.keys())
                graph_connected: dict[str, float] = {}
                for seed_path in seed_paths:
                    neighbors = graph_store.bfs(seed_path, max_depth=graph_depth)
                    for neighbor_path, trace in neighbors.items():
                        if neighbor_path not in note_scores:
                            depth = len(trace) - 1
                            proximity = 1.0 / max(depth, 1)
                            graph_connected[neighbor_path] = max(
                                graph_connected.get(neighbor_path, 0), proximity
                            )
                for path, proximity in graph_connected.items():
                    title = os.path.splitext(os.path.basename(path))[0]
                    entry = {"score": proximity * w["graph"], "title": title,
                             "snippet": "", "matched_by": ["graph"]}
                    note_scores[path] = entry
            except Exception as exc:
                log.warning("ranker — graph traversal failed: %s", exc)

        # ── 5. Sort, threshold, trim ────────────────────────────────
        scored: list[tuple[str, float]] = [(path, info["score"])
                                            for path, info in note_scores.items()]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for path, score in scored[:n * 2]:
            if min_similarity is not None and score < min_similarity:
                continue
            info = note_scores[path]
            results.append({
                "path": path,
                "title": info["title"],
                "similarity_score": round(score, 4),
                "matched_by": info["matched_by"],
                "snippet": info["snippet"],
            })
            if len(results) >= n:
                break

        return results


# ── Module-level singleton ──────────────────────────────────────────

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker()
    return _ranker


def search(
    query: str,
    n: int = 5,
    use_entities: bool = False,
    use_graph: bool = False,
    graph_depth: int = 1,
    weights: dict[str, float] | None = None,
    where: dict | None = None,
    exclude_tags: list[str] | None = None,
    expand_queries: list[str] | None = None,
    min_similarity: float | None = None,
    expand_entities: bool = False,
) -> list[dict]:
    """Convenience wrapper around ``Ranker.search()`` using the module-level singleton."""
    return _get_ranker().search(
        query=query, n=n,
        use_entities=use_entities, use_graph=use_graph,
        graph_depth=graph_depth,
        weights=weights,
        where=where, exclude_tags=exclude_tags,
        expand_queries=expand_queries,
        min_similarity=min_similarity,
        expand_entities=expand_entities,
    )


def weights() -> dict[str, float]:
    """Return the current ranking weights (for status / debugging)."""
    return _get_ranker().weights


def set_weights(
    semantic: float | None = None,
    entity: float | None = None,
    graph: float | None = None,
    keyword: float | None = None,
) -> dict[str, float]:
    """Update ranking weights at runtime. Returns the new weights dict."""
    return _get_ranker().set_weights(
        semantic=semantic, entity=entity,
        graph=graph, keyword=keyword,
    )
