import atexit
import json
import os
import time
from datetime import datetime

from .. import (
    chroma_store,
    config,
    entity_store,
    graph_store,
    keyword_search,
    llm_client,
    obsidian_client,
    ranker,
)
from ..frontmatter import parse as fm_parse
from ..logger import get_logger

log = get_logger("obsidian_ai.tools._shared")

# ── vault file scan ───────────────────────────────────────────────


def _find_notes_mentioning(name: str) -> list[str]:
    """Walk vault and return relative paths of ``.md`` files whose content contains ``name``.

    Uses a plain-text substring check — no LLM, no API calls.
    Returns paths sorted alphabetically, excluding the entity/subject hub itself
    to avoid a redundant re-index cycle.
    """
    vault = config.vault_path
    if not vault or not os.path.isdir(vault):
        return []
    vault_norm = vault.replace("\\", "/").rstrip("/") + "/"
    matches: list[str] = []
    for root, _, files in os.walk(vault):
        for f in files:
            if not f.endswith(".md"):
                continue
            abs_path = os.path.join(root, f)
            rel_path = abs_path.replace("\\", "/")
            if not rel_path.startswith(vault_norm):
                continue
            rel_path = rel_path[len(vault_norm):]
            try:
                with open(abs_path, "r", encoding="utf-8") as fh:
                    if name in fh.read():
                        matches.append(rel_path)
            except Exception:
                continue
    matches.sort()
    return matches


# ── filter builder ─────────────────────────────────────────────────


QUERY_EXPANSION_SYSTEM = (
    "You are a search query expansion assistant. Given a query, generate 3-5 "
    "alternative phrasings that would help find relevant documents. Focus on "
    "synonyms, technical terms, acronyms (both expanded and contracted forms), "
    "and related concepts. Return each phrasing on its own line, no numbering, "
    "no extra text."
)

REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a personal knowledge vault. "
    "Given a user query and a list of known entity names and note titles from "
    "the vault, rewrite the query to use the vault's own terminology when "
    "possible. If the query is already well-phrased for the vault, return it "
    "unchanged. Reply with ONLY the rewritten query, nothing else, no quotes."
)

SUBJECT_EXPANSION_SYSTEM = (
    "You are a subject expansion assistant. Given a subject or topic, generate "
    "5-8 related terms, synonyms, and sub-topics that would help find relevant "
    "notes about that subject. Include technical terms, abbreviations, and "
    "related concepts. Return each term on its own line, no numbering, no extra text."
)

SNIPPET_MAX_CHARS = 300


def _truncate_snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Truncate text to max_chars, appending '...' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _normalize_path(path: str) -> str:
    """Strip vault prefix so the LLM can pass absolute or relative paths."""
    vault = config.vault_path.replace("\\", "/").rstrip("/")
    clean = path.replace("\\", "/")
    if clean.startswith(vault + "/"):
        return clean[len(vault) + 1:]
    if clean.startswith(vault):
        return clean[len(vault):]
    return clean


def _get_vault_terminology() -> str:
    """Build a compact string of known entity names + note titles for rewriting context."""
    terms: list[str] = []
    try:
        from .. import entity_store
        entities = entity_store.get_all_entities()
        name_counts: dict[str, int] = {}
        for ent in entities:
            name_counts[ent["name"]] = name_counts.get(ent["name"], 0) + 1
        names = sorted(name_counts, key=name_counts.get, reverse=True)
        terms.append("Entities: " + ", ".join(names[:60]))
    except Exception:
        pass
    try:
        vault_path = config.vault_path
        if vault_path and os.path.isdir(vault_path):
            titles = []
            for entry in os.scandir(vault_path):
                if entry.name.endswith(".md") and entry.is_file():
                    titles.append(entry.name.removesuffix(".md"))
            if titles:
                terms.append("Notes: " + ", ".join(sorted(titles)[:80]))
    except Exception:
        pass
    return "\n".join(terms)


def _rewrite_query(query: str) -> str:
    """Rewrite a user query using known vault terminology for better retrieval."""
    try:
        context = _get_vault_terminology()
        if not context.strip():
            return query
        user_msg = f"Vault context:\n{context}\n\nQuery: {query}"
        response = llm_client.chat(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            think=False,
        )
        rewritten = response.strip().strip("\"'")
        if rewritten and rewritten.lower() != query.lower():
            log.info(f"Query rewritten: {query!r} → {rewritten!r}")
            return rewritten
        return query
    except Exception as e:
        log.warning(f"Query rewriting failed: {e}")
        return query


# TTL-based query expansion cache with disk persistence.
_EXPAND_QUERY_CACHE: dict[str, tuple[float, list[str]]] = {}
_EXPAND_QUERY_CACHE_PATH = os.path.join(config.data_dir, "expand_cache.json")


def _load_expand_cache() -> dict[str, tuple[float, list[str]]]:
    try:
        if os.path.isfile(_EXPAND_QUERY_CACHE_PATH):
            with open(_EXPAND_QUERY_CACHE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            return {k: (v[0], v[1]) for k, v in raw.items()}
    except Exception as e:
        log.warning(f"Failed to load expand cache: {e}")
    return {}


def _save_expand_cache() -> None:
    try:
        os.makedirs(os.path.dirname(_EXPAND_QUERY_CACHE_PATH), exist_ok=True)
        serializable = {k: [ts, results] for k, (ts, results) in _EXPAND_QUERY_CACHE.items()}
        with open(_EXPAND_QUERY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Failed to save expand cache: {e}")


_EXPAND_QUERY_CACHE.update(_load_expand_cache())
atexit.register(_save_expand_cache)


def _expand_query(query: str) -> list[str]:
    """Use the LLM to generate alternative query phrasings for broader search.

    Results are cached with a TTL (default 1 hour) and persisted to disk.
    Returns a list of expanded query strings, or an empty list if expansion fails.
    """
    now = time.time()
    ttl = config.expand_cache_ttl

    # Check cache
    cached = _EXPAND_QUERY_CACHE.get(query)
    if cached is not None:
        ts, results = cached
        if now - ts < ttl:
            log.debug(f"Query expansion cache hit: {query}")
            return results
        else:
            del _EXPAND_QUERY_CACHE[query]

    try:
        response = llm_client.chat(
            [
                {"role": "system", "content": QUERY_EXPANSION_SYSTEM},
                {"role": "user", "content": f"Query: {query}"},
            ],
            think=False,
        )
        lines = [
            line.strip(" -\"'.,;:!?")
            for line in response.strip().split("\n")
            if line.strip()
        ]
        seen = {query.lower()}
        unique = []
        for phrase in lines:
            lower = phrase.lower().rstrip(".")
            if lower and lower not in seen and len(lower) > 3:
                seen.add(lower)
                unique.append(phrase)
        results = unique[:5]
        _EXPAND_QUERY_CACHE[query] = (now, results)
        log.info(f"Query expansion: {query} → {results}")
        return results
    except Exception as e:
        log.warning(f"Query expansion failed: {e}")
        return []


def _build_search_where(
    tags: list[str] | None = None,
    folder: str | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
) -> dict | None:
    """Build a ChromaDB ``where`` clause from optional search filters.
    Returns ``None`` when no filters are active."""
    conditions: list[dict] = []

    # Tags: $contains on tags_str with ",tag," delimiter
    if tags:
        for tag in tags:
            conditions.append({"tags_str": {"$contains": f",{tag},"}})

    # Folder: $contains on path (trailing slash ensures folder prefix match)
    if folder:
        f = folder.replace("\\", "/").strip("/")
        conditions.append({"path": {"$contains": f"{f}/"}})

    # Date range: mtime as Unix timestamp
    try:
        if date_after:
            ts = datetime.fromisoformat(date_after).timestamp()
            conditions.append({"mtime": {"$gte": ts}})
        if date_before:
            ts = datetime.fromisoformat(date_before).timestamp()
            conditions.append({"mtime": {"$lte": ts}})
    except ValueError as e:
        raise ValueError(f"Invalid ISO date format: {e}") from e

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _matches_where(metadata: dict, where: dict | None) -> bool:
    """Check whether a metadata dict satisfies a ChromaDB-style where clause."""
    if where is None:
        return True
    conditions = where.get("$and", [where])
    for cond in conditions:
        for field, op in cond.items():
            val = metadata.get(field)
            if val is None:
                return False
            if "$contains" in op and op["$contains"] not in str(val):
                return False
            if "$gte" in op and float(val) < float(op["$gte"]):
                return False
            if "$lte" in op and float(val) > float(op["$lte"]):
                return False
    return True


# ── shared search helpers ───────────────────────────────────────────


def _hybrid_search(
    queries: list[str],
    n: int = 5,
    keyword_weight: float = 0.0,
    min_similarity: float | None = None,
    diversity_penalty: float = 0.0,
    where: dict | None = None,
    exclude_tags: list[str] | None = None,
) -> list[dict]:
    """Shared hybrid semantic + BM25 search used by search_notes and get_subject.

    Args:
        queries: list of query strings to embed and search semantically.
        n: max results to return.
        keyword_weight: blend ratio for BM25 (0.0 = pure semantic, 1.0 = pure keyword).
        min_similarity: minimum similarity score threshold.
        diversity_penalty: penalise same-note repetition.
        where: ChromaDB where clause for metadata filtering.
        exclude_tags: tags to exclude from results.

    Returns:
        List of passage dicts with path, title, matched_chunk_idx, similarity_score, snippet.
    """
    # Step 1: Semantic search
    semantic_merged: dict[tuple[str, int], dict] = {}
    for q in queries:
        results = chroma_store.query(llm_client.embed(q), n=n + 10, where=where)
        for r in results:
            metadata = r["metadata"]
            path = metadata["path"]
            chunk_idx = metadata.get("chunk", 0)
            key = (path, chunk_idx)

            if exclude_tags:
                tags_str = metadata.get("tags_str", "")
                if any(f",{et}," in tags_str for et in exclude_tags):
                    continue

            distance = r["distance"]
            similarity = 1.0 / (1.0 + distance) if distance is not None else 0.0

            if key not in semantic_merged or similarity > semantic_merged[key]["similarity_score"]:
                semantic_merged[key] = {
                    "path": path,
                    "title": metadata.get("title", ""),
                    "matched_chunk_idx": chunk_idx,
                    "similarity_score": round(similarity, 4),
                    "snippet": _truncate_snippet(r.get("document") or ""),
                }

    # Step 2: BM25 keyword search
    uses_keyword = keyword_weight > 0.0
    if uses_keyword:
        kw_query = " ".join(queries)
        kw_results = keyword_search.search(kw_query, n=n + 15)
        keyword_search.normalise_scores(kw_results)

    # Step 3: Merge and blend scores
    merged: dict[tuple[str, int], dict] = {}
    semantic_w = 1.0 - keyword_weight
    for key, entry in semantic_merged.items():
        merged[key] = {**entry, "similarity_score": entry["similarity_score"] * semantic_w}

    if uses_keyword:
        for r in kw_results:
            metadata = r["metadata"]
            path = metadata.get("path", "")
            chunk_idx = metadata.get("chunk", 0)
            key = (path, chunk_idx)
            if not path:
                continue
            if exclude_tags:
                tags_str = metadata.get("tags_str", "")
                if any(f",{et}," in tags_str for et in exclude_tags):
                    continue
            if not _matches_where(metadata, where):
                continue
            kw_score = r["bm25_score"] * keyword_weight
            if key in merged:
                merged[key]["similarity_score"] += kw_score
            else:
                merged[key] = {
                    "path": path,
                    "title": metadata.get("title", ""),
                    "matched_chunk_idx": chunk_idx,
                    "similarity_score": round(kw_score, 4),
                    "snippet": _truncate_snippet(r.get("document") or ""),
                }

    for entry in merged.values():
        entry["similarity_score"] = round(entry["similarity_score"], 4)

    # Step 4: Apply min_similarity threshold
    passages = sorted(merged.values(), key=lambda p: p["similarity_score"], reverse=True)
    if min_similarity is not None:
        passages = [p for p in passages if p["similarity_score"] >= min_similarity]

    # Step 5: Diversity penalty
    if diversity_penalty > 0.0 and passages:
        selected: list[dict] = []
        path_counts: dict[str, int] = {}
        remaining = list(passages)
        while remaining and len(selected) < n:
            best = remaining.pop(0)
            path = best["path"]
            count = path_counts.get(path, 0)
            if count > 0:
                penalty_mult = max(0.0, 1.0 - diversity_penalty * count)
                new_score = round(best["similarity_score"] * penalty_mult, 4)
                if new_score <= 0.0:
                    continue
                best["similarity_score"] = new_score
                remaining.append(best)
                remaining.sort(key=lambda p: p["similarity_score"], reverse=True)
                continue
            path_counts[path] = count + 1
            selected.append(best)
        passages = selected
    else:
        passages = passages[:n]

    return passages


def _group_by_note(passages: list[dict], n: int) -> list[dict]:
    """Collapse chunk-level passages into note-level results.

    For each note, keeps the highest similarity score and the snippet from
    the best-matching chunk. Results are sorted by score descending.
    """
    grouped: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for p in passages:
        path = p["path"]
        score = p["similarity_score"]
        counts[path] = counts.get(path, 0) + 1
        if path not in grouped or score > grouped[path]["similarity_score"]:
            grouped[path] = {
                "path": path,
                "title": p["title"],
                "similarity_score": score,
                "snippet": p["snippet"],
            }

    for entry in grouped.values():
        path = entry["path"]
        entry["similarity_score"] = round(entry["similarity_score"], 4)
        entry["chunk_count"] = counts[path]

    results = sorted(grouped.values(), key=lambda x: x["similarity_score"], reverse=True)
    return results[:n]


def _apply_entity_search(
    passages: list[dict],
    query: str,
    entity_types: list[str] | None = None,
    where: dict | None = None,
    force: bool = False,
    expand_entities: bool = False,
) -> list[dict]:
    """Expand results by auto-detecting entities in the query.

    Uses ``ranker._auto_detect_entities`` to check if the query mentions
    any known entity (by full string, individual words, or bigrams).
    If *force* is True, also runs the broader ``entity_store.search()``
    even when no word-level match is found.

    If *expand_entities* is True, when entities are auto-detected, also
    fetches related entities via the entity-relationship graph and
    searches for those as well.

    Merges results with existing passages, deduplicating by path and
    keeping the higher score.
    """
    entity_results = ranker._auto_detect_entities(query)

    if force:
        explicit = entity_store.search(query)
        seen = {r["path"] for r in entity_results}
        for r in explicit:
            if r["path"] not in seen:
                entity_results.append(r)

    # Relationship expansion: find entities related to detected ones
    expand_names: list[str] = []
    if expand_entities and entity_results:
        try:
            expand_names = ranker._expand_entity_names(entity_results)
        except Exception as exc:
            log.warning("_apply_entity_search — entity expansion failed: %s", exc)

    if expand_names:
        log.info("_apply_entity_search — expanded with related entities: %s", expand_names)
        seen_paths = {r["path"] for r in entity_results}
        for ename in expand_names:
            try:
                expanded_results = entity_store.search(ename)
                for r in expanded_results:
                    if r["path"] not in seen_paths:
                        seen_paths.add(r["path"])
                        entity_results.append(r)
            except Exception as exc:
                log.warning("_apply_entity_search — related search for %s failed: %s", ename, exc)

    if not entity_results:
        return passages

    if entity_types:
        entity_results = [r for r in entity_results if r["entity_type"] in entity_types]

    existing = {p["path"]: p for p in passages}

    added = 0
    for r in entity_results:
        path = r["path"]
        title = os.path.splitext(os.path.basename(path))[0]
        score = round(r["confidence"] * 0.9, 4)
        snippet = r.get("snippet") or f"[{r['entity_type']}] {r['entity_name']}"

        if path in existing:
            existing_entry = existing[path]
            if score > existing_entry["similarity_score"]:
                existing_entry["similarity_score"] = score
                existing_entry["is_entity_match"] = True
        else:
            passages.append({
                "path": path,
                "title": title,
                "matched_chunk_idx": 0,
                "similarity_score": score,
                "snippet": _truncate_snippet(snippet),
                "is_entity_match": True,
            })
            added += 1

    if added:
        passages.sort(key=lambda p: p["similarity_score"], reverse=True)
        log.info("_apply_entity_search — %s entity results added", added)
    return passages


def _apply_graph_boost(passages: list[dict], graph_depth: int = 1, graph_weight: float = 0.2) -> list[dict]:
    """Expand search results by following wiki-links from result notes.

    For each result note, BFS up to graph_depth hops to find connected notes.
    Connected notes get a score boost proportional to graph_weight.
    Returns deduplicated, re-sorted results.
    """
    # Collect unique paths
    result_paths = set(p["path"] for p in passages)

    # BFS from each result note
    graph_connected: dict[str, float] = {}  # path -> max proximity score (1/depth)
    for path in result_paths:
        neighbors = graph_store.bfs(path, max_depth=graph_depth)
        for neighbor_path, trace in neighbors.items():
            if neighbor_path not in result_paths:
                depth = len(trace) - 1
                proximity = 1.0 / depth if depth > 0 else 0.5
                graph_connected[neighbor_path] = max(
                    graph_connected.get(neighbor_path, 0), proximity
                )

    if not graph_connected:
        return passages

    # Fetch content for connected notes and create passage entries
    for neighbor_path, proximity in graph_connected.items():
        try:
            content = obsidian_client.get_note(neighbor_path)
            _, body = fm_parse(content)
            snippet = _truncate_snippet(body[:500])
            title = os.path.splitext(os.path.basename(neighbor_path))[0]
            passages.append({
                "path": neighbor_path,
                "title": title,
                "matched_chunk_idx": 0,
                "similarity_score": round(proximity * graph_weight, 4),
                "snippet": snippet,
                "is_graph_connected": True,
            })
        except Exception:
            pass

    # Re-sort by score
    return sorted(passages, key=lambda x: x["similarity_score"], reverse=True)


def _filter_subjects(results: list[dict]) -> list[dict]:
    """Remove any results from the Subjects/ structural folder."""
    return [r for r in results if not r.get("path", "").startswith("Subjects/")]


def _apply_summary_search(
    passages: list[dict],
    query: str,
    n: int = 5,
    summary_threshold: float = 0.7,
) -> list[dict]:
    """Augment results with summary-embedding matches.

    Queries the summary store for each query; results with similarity
    above *summary_threshold* are added with ``is_summary_match`` flag.
    Merges with existing passages, deduplicating by path and keeping
    the higher score.
    """
    try:
        from obsidian_ai import summary_store
        results = summary_store.query(query, n=n)
    except Exception as exc:
        log.warning("_apply_summary_search — summary query failed: %s", exc)
        return passages

    existing = {p["path"]: p for p in passages}
    added = 0
    for r in results:
        sim = r.get("similarity", 0.0)
        if sim < summary_threshold:
            continue
        path = r["path"]
        title = r.get("title", os.path.splitext(os.path.basename(path))[0])
        snippet = r.get("summary", "")[:300]
        if path in existing:
            existing_entry = existing[path]
            if sim > existing_entry.get("similarity_score", 0.0):
                existing_entry["similarity_score"] = sim
                existing_entry["is_summary_match"] = True
        else:
            passages.append({
                "path": path,
                "title": title,
                "matched_chunk_idx": 0,
                "similarity_score": round(sim, 4),
                "snippet": _truncate_snippet(snippet) if snippet else "",
                "is_summary_match": True,
            })
            added += 1

    if added:
        passages.sort(key=lambda p: p["similarity_score"], reverse=True)
        log.info("_apply_summary_search — %s summary results added", added)
    return passages
