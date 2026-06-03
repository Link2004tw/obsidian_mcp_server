import os
import re
from datetime import datetime

import requests

from fastmcp import FastMCP

from . import chroma_store, config, entity_store, graph_store, indexer, keyword_search, llm_client, obsidian_client, pipelines
from .frontmatter import add_tags as fm_add_tags
from .frontmatter import parse as fm_parse
from .frontmatter import remove_tags as fm_remove_tags
from .frontmatter import set_tags as fm_set_tags
from .logger import get_logger, log_error
from .todos import (
    add_todo as td_add,
    complete_todo as td_complete,
    delete_todo as td_delete,
    ensure_todos_file_exists as td_ensure,
    get_todos as td_get,
    get_todo_stats as td_stats,
    sync_todos as td_sync,
    update_todo as td_update,
)

log = get_logger("obsidian_ai.mcp_server", log_file="mcp_calls.log")

mcp = FastMCP("obsidian-ai")

# ── filter builder ─────────────────────────────────────────────────


QUERY_EXPANSION_SYSTEM = (
    "You are a search query expansion assistant. Given a query, generate 3-5 "
    "alternative phrasings that would help find relevant documents. Focus on "
    "synonyms, technical terms, acronyms (both expanded and contracted forms), "
    "and related concepts. Return each phrasing on its own line, no numbering, "
    "no extra text."
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


# Cache expanded queries to reduce repeated LLM expansion cost.
_EXPAND_QUERY_CACHE_SIZE = 256


from functools import lru_cache


@lru_cache(maxsize=_EXPAND_QUERY_CACHE_SIZE)
def _expand_query(query: str) -> list[str]:
    """Use the LLM to generate alternative query phrasings for broader search.

    Returns a list of expanded query strings, or an empty list if expansion fails.
    """
    try:
        response = llm_client.chat(
            [
                {"role": "system", "content": QUERY_EXPANSION_SYSTEM},
                {"role": "user", "content": f"Query: {query}"},
            ],
            think=False,
        )
        # Split by newlines first, then clean each line
        lines = [
            line.strip(" -\"'.,;:!?")
            for line in response.strip().split("\n")
            if line.strip()
        ]
        # Deduplicate and filter out anything too similar to original
        seen = {query.lower()}
        unique = []
        for phrase in lines:
            lower = phrase.lower().rstrip(".")
            if lower and lower not in seen and len(lower) > 3:
                seen.add(lower)
                unique.append(phrase)
        log.info(f"Query expansion: {query} → {unique}")
        return unique[:5]
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
            if "$contains" in op:
                if op["$contains"] not in str(val):
                    return False
            if "$gte" in op:
                if float(val) < float(op["$gte"]):
                    return False
            if "$lte" in op:
                if float(val) > float(op["$lte"]):
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


# ── tools ──────────────────────────────────────────────────────────


@mcp.tool()
def get_index_stats() -> str:
    """Return index statistics (total chunks, unique notes, config info, cache stats)."""
    log.info("get_index_stats")
    try:
        stats = chroma_store.get_index_stats()
        cache = llm_client.embed_cache_info()
        ent_stats = entity_store.stats()
        lines = [
            f"Unique notes indexed: {stats['unique_notes']}",
            f"Total chunks stored:  {stats['total_chunks']}",
            f"Embedding model:      {config.ollama_embed_model}",
            f"ChromaDB path:        {config.chroma_path}",
            f"Embedding cache:      {cache['currsize']}/{cache['maxsize']} (hits={cache['hits']}, misses={cache['misses']})",
            f"Entities extracted:   {ent_stats['total_entities']} ({ent_stats['total_mentions']} mentions)",
        ]
        result = "\n".join(lines)
        log.info(f"get_index_stats — {result.replace(chr(10), ' | ')}")
        return result
    except Exception as e:
        log_error(log, "get_index_stats FAILED", exc=e)
        return f"Error: {e}"


@mcp.tool()
def find_duplicate_notes(threshold: float = 0.9, n: int = 20) -> list[dict]:
    """Find near-duplicate notes via embedding similarity.

    Args:
        threshold: cosine similarity threshold (0.0-1.0), default 0.9. Higher = must be more similar.
        n: maximum number of duplicate pairs to return (default 20).
    """
    log.info(f"find_duplicate_notes — threshold={threshold}, n={n}")
    try:
        results = chroma_store.find_duplicate_notes(threshold=threshold, n=n)
        log.info(f"find_duplicate_notes — {len(results)} pairs found")
        return results
    except Exception as e:
        log_error(log, "find_duplicate_notes FAILED", exc=e)
        return []


def _apply_entity_search(
    passages: list[dict],
    query: str,
    entity_types: list[str] | None = None,
    where: dict | None = None,
) -> list[dict]:
    """Expand results by looking up entities matching the query.

    Queries the entity store for the query text. For each matching entity,
    adds all notes that mention it. Merges with existing passages,
    deduplicating by path and keeping the higher score.
    """
    entity_results = entity_store.search(query)
    if not entity_results:
        return passages

    if entity_types:
        entity_results = [r for r in entity_results if r["entity_type"] in entity_types]

    # Index existing passages by path for fast dedup
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


@mcp.tool()
def search_notes(
    query: str,
    n: int = 5,
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    folder: str | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
    expand_query: bool = False,
    keyword_weight: float = 0.0,
    min_similarity: float | None = None,
    diversity_penalty: float = 0.0,
    use_graph: bool = False,
    graph_depth: int = 1,
    graph_weight: float = 0.2,
    use_entities: bool = False,
    entity_types: list[str] | None = None,
    group_by_note: bool = False,
) -> list[dict]:
    """Search notes semantically with optional metadata filters.

    By default returns passage-level results — each result is a single
    matching chunk. Multiple passages from the same note may appear.

    When ``group_by_note=True``, collapses chunk-level results into one
    result per note with the highest similarity score, best snippet, and
    a ``chunk_count`` field showing how many chunks matched.

    Filters (all optional):
    - ``tags`` — only notes having ALL of these YAML tags
    - ``exclude_tags`` — exclude notes having ANY of these tags
    - ``folder`` — only notes inside this vault-relative folder
    - ``date_after`` / ``date_before`` — ISO date strings (e.g. ``2024-06-01``) to filter by file mtime
    - ``expand_query`` — if True, use the LLM to generate alternative query phrasings
      for broader search (adds ~1-2s per search)
    - ``keyword_weight`` — blend ratio for BM25 keyword search (0.0 = pure semantic,
      1.0 = pure keyword)
    - ``min_similarity`` — minimum final similarity score (0-1); results below this
      threshold are filtered out
    - ``diversity_penalty`` — diversity penalty factor (0.0 = none, 0.5 = moderate,
      1.0 = aggressive). Penalises passages from a note that already has results
      selected, encouraging diverse sources.
    - ``use_graph`` — if True, expand results via wiki-link graph traversal
    - ``graph_depth`` — max hops for graph traversal (default 1)
    - ``graph_weight`` — weight for graph proximity boost (0.0-1.0, default 0.2)
    - ``use_entities`` — if True, also search the entity index for notes matching the query entity name
    - ``entity_types`` — optional list of entity types to filter by (e.g. ``["Person"]``)
    - ``group_by_note`` — if True, collapse chunk-level results into note-level (default False)

    Returns:
        Passage-level results (default) or note-level results (when ``group_by_note=True``).
        Each dict has ``path``, ``title``, ``similarity_score``, ``snippet``,
        and when grouped: ``chunk_count`` (how many chunks matched this note).
    """
    log.info(
        "search_notes — query=%s, n=%s, tags=%s, exclude_tags=%s, folder=%s, date_after=%s, date_before=%s, expand=%s, kw_weight=%s, min_sim=%s, div_penalty=%s, use_graph=%s, graph_depth=%s, graph_weight=%s, use_entities=%s, entity_types=%s, group_by_note=%s",
        query, n, tags, exclude_tags, folder, date_after, date_before,
        expand_query, keyword_weight, min_similarity, diversity_penalty,
        use_graph, graph_depth, graph_weight, use_entities, entity_types, group_by_note,
    )
    try:
        where = _build_search_where(tags=tags, folder=folder, date_after=date_after, date_before=date_before)
        queries_to_embed = [query]
        if expand_query:
            expanded = _expand_query(query)
            if expanded:
                log.info(f"search_notes — query expanded: {expanded}")
                queries_to_embed.extend(expanded)

        passages = _hybrid_search(
            queries=queries_to_embed,
            n=n,
            keyword_weight=keyword_weight,
            min_similarity=min_similarity,
            diversity_penalty=diversity_penalty,
            where=where,
            exclude_tags=exclude_tags,
        )

        # Entity-augmented expansion
        if use_entities:
            passages = _apply_entity_search(passages, query, entity_types=entity_types)

        # Graph-augmented expansion
        if use_graph and passages:
            passages = _apply_graph_boost(passages, graph_depth=graph_depth, graph_weight=graph_weight)

        # Group by note (collapse chunks) before final trim
        if group_by_note and passages:
            passages = _group_by_note(passages, n)
        else:
            passages = passages[:n]

        log.info("search_notes — %s results returned", len(passages))
        return passages
    except Exception as e:
        log_error(log, "search_notes FAILED", exc=e, query=query, n=n)
        return []


@mcp.tool()
def batch_search(
    queries: list[str],
    n: int = 5,
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    folder: str | None = None,
    keyword_weight: float = 0.0,
    min_similarity: float | None = None,
) -> dict[str, list[dict]]:
    """Run multiple searches in one call. Returns ``dict[query, results]``.

    Each search runs ``search_notes``-style hybrid search (semantic + BM25)
    with optional metadata filters. For per-query fine-grained params
    (expand_query, graph, entities), use individual ``search_notes`` calls.
    """
    log.info("batch_search — %d queries, n=%s", len(queries), n)
    where = _build_search_where(tags=tags, folder=folder)
    results: dict[str, list[dict]] = {}
    for q in queries:
        try:
            passages = _hybrid_search(
                queries=[q],
                n=n,
                keyword_weight=keyword_weight,
                min_similarity=min_similarity,
                diversity_penalty=0.0,
                where=where,
                exclude_tags=exclude_tags,
            )
            results[q] = passages[:n]
        except Exception as e:
            log_error(log, f"batch_search — query failed: {q}", exc=e)
            results[q] = []
    return results


@mcp.tool()
def read_note(path: str) -> str:
    """Read the full content of a note by path."""
    log.info(f"read_note — {path}")
    try:
        content = obsidian_client.get_note(path)
        log.info(f"read_note — {path} — {len(content)} chars")
        return content
    except Exception as e:
        log_error(log, f"read_note FAILED: {path}", exc=e)
        return f"Error: {e}"


@mcp.tool()
def write_note(path: str, content: str) -> str:
    """Create or overwrite a note with the given content."""
    log.info(f"write_note — {path} — {len(content)} chars")
    try:
        obsidian_client.put_note(path, content)
        return f"Written: {path}"
    except Exception as e:
        log_error(log, f"write_note FAILED: {path}", exc=e, content_len=len(content))
        return f"Error: {e}"


@mcp.tool()
def list_all_notes() -> list[str]:
    """Return a list of all note paths in the vault."""
    log.info("list_all_notes")
    try:
        notes = obsidian_client.list_all_notes()
        log.info(f"list_all_notes — {len(notes)} notes returned")
        return notes
    except Exception as e:
        log_error(log, "list_all_notes FAILED", exc=e)
        return []


@mcp.tool()
def list_folder(folder_path: str) -> list[str]:
    """Return entries directly inside a specific folder (non-recursive)."""
    log.info(f"list_folder — {folder_path}")
    try:
        notes = obsidian_client.list_folder(folder_path)
        log.info(f"list_folder — {len(notes)} notes returned")
        return notes
    except Exception as e:
        log_error(log, f"list_folder FAILED: {folder_path}", exc=e)
        return []


@mcp.tool()
def list_folder_deep(folder_path: str) -> list[str]:
    """Return a list of note paths within a specific folder (recursive)."""
    log.info(f"list_folder_deep — {folder_path}")
    try:
        notes = obsidian_client.list_folder_deep(folder_path)
        log.info(f"list_folder_deep — {len(notes)} notes returned")
        return notes
    except Exception as e:
        log_error(log, f"list_folder_deep FAILED: {folder_path}", exc=e)
        return []


@mcp.tool()
def search_by_tags(tags: list[str], n: int = 10) -> list[dict]:
    """Find notes that have ALL of the given YAML frontmatter tags."""
    log.info(f"search_by_tags — tags={tags}, n={n}")
    try:
        results = chroma_store.search_by_tags(tags, n=n)
        out = []
        for r in results:
            raw = r.get("tags_str", "")
            note_tags = [t for t in raw.strip(",").split(",") if t] if raw else []
            out.append(
                {
                    "path": r["path"],
                    "title": r.get("title", ""),
                    "tags": note_tags,
                    "snippet": r.get("snippet", ""),
                }
            )
        log.info(f"search_by_tags — {len(out)} results")
        return out
    except Exception as e:
        log_error(log, "search_by_tags FAILED", exc=e, tags=tags, n=n)
        return []


@mcp.tool()
def read_note_by_title(title: str, folder_path: str = "") -> str:
    """Look up a note by its title (filename without extension) and return the full content."""
    log.info(f"read_note_by_title — title={title}, folder_path={folder_path or '(none)'}")
    try:
        matches = chroma_store.get_by_title(title)
        if not matches:
            msg = f"No note found with title: {title}"
            log.warning(f"read_note_by_title — {msg}")
            return f"Error: {msg}"

        paths = [m["path"] for m in matches]

        if folder_path:
            prefix = folder_path.replace("\\", "/").rstrip("/") + "/"
            filtered = [p for p in paths if p.startswith(prefix)]
            if not filtered:
                msg = f'No note with title "{title}" found in folder: {folder_path}'
                log.warning(f"read_note_by_title — {msg}")
                return f"Error: {msg}"
            paths = filtered

        if len(paths) == 1:
            content = obsidian_client.get_note(paths[0])
            log.info(
                f"read_note_by_title — {title} → {paths[0]} — {len(content)} chars"
            )
            return content

        log.info(f"read_note_by_title — {title} — multiple matches ({len(paths)}): {paths}")
        parts = []
        for p in paths:
            content = obsidian_client.get_note(p)
            parts.append(f"─── {p} ───\n{content}")
        return f'Found {len(paths)} notes with title "{title}":\n\n' + "\n\n".join(parts)
    except Exception as e:
        log_error(log, f"read_note_by_title FAILED: title={title}, folder_path={folder_path}", exc=e)
        return f"Error: {e}"


@mcp.tool()
def add_tags(path: str, tags: list[str]) -> str:
    """Add tags to a note's YAML frontmatter. Creates frontmatter if absent. Path must be relative to vault root."""
    log.info(f"add_tags — {path} — tags={tags}")
    try:
        content = obsidian_client.get_note(path)
        new_content = fm_add_tags(content, tags)
        obsidian_client.put_note(path, new_content)
        meta, _ = fm_parse(new_content)
        log.info(f"add_tags — {path} — final tags={meta.get('tags', [])}")
        return f"Tags added to {path}: {meta.get('tags', [])}"
    except Exception as e:
        log_error(log, f"add_tags FAILED: {path}", exc=e, tags=tags)
        return f"Error: {e}"


@mcp.tool()
def remove_tags(path: str, tags: list[str]) -> str:
    """Remove specific tags from a note's YAML frontmatter. Path must be relative to vault root."""
    log.info(f"remove_tags — {path} — tags={tags}")
    try:
        content = obsidian_client.get_note(path)
        new_content = fm_remove_tags(content, tags)
        obsidian_client.put_note(path, new_content)
        meta, _ = fm_parse(new_content)
        log.info(f"remove_tags — {path} — final tags={meta.get('tags', [])}")
        return f"Tags removed from {path}. Remaining: {meta.get('tags', [])}"
    except Exception as e:
        log_error(log, f"remove_tags FAILED: {path}", exc=e, tags=tags)
        return f"Error: {e}"


@mcp.tool()
def set_tags(path: str, tags: list[str]) -> str:
    """Replace all tags on a note with the given list. Path must be relative to vault root."""
    log.info(f"set_tags — {path} — tags={tags}")
    try:
        content = obsidian_client.get_note(path)
        new_content = fm_set_tags(content, tags)
        obsidian_client.put_note(path, new_content)
        meta, _ = fm_parse(new_content)
        log.info(f"set_tags — {path} — final tags={meta.get('tags', [])}")
        return f"Tags set on {path}: {meta.get('tags', [])}"
    except Exception as e:
        log_error(log, f"set_tags FAILED: {path}", exc=e, tags=tags)
        return f"Error: {e}"


@mcp.tool()
def batch_tag_notes(note_paths: list[str], tags: list[str]) -> dict[str, str]:
    """Add tags to multiple notes at once. Returns ``dict[path, result_message]``."""
    log.info("batch_tag_notes — %d notes, tags=%s", len(note_paths), tags)
    results: dict[str, str] = {}
    for path in note_paths:
        try:
            content = obsidian_client.get_note(path)
            new_content = fm_add_tags(content, tags)
            obsidian_client.put_note(path, new_content)
            meta, _ = fm_parse(new_content)
            results[path] = f"Tags added: {meta.get('tags', [])}"
        except Exception as e:
            log_error(log, f"batch_tag_notes FAILED: {path}", exc=e, tags=tags)
            results[path] = f"Error: {e}"
    return results


@mcp.tool()
def create_backlink(path_a: str, path_b: str) -> str:
    """Create mutual [[backlinks]] between two notes."""
    log.info(f"create_backlink — {path_a} <-> {path_b}")
    try:
        name_a = os.path.splitext(os.path.basename(path_a))[0]
        name_b = os.path.splitext(os.path.basename(path_b))[0]
        link_to_b = f"[[{name_b}]]"
        link_to_a = f"[[{name_a}]]"

        content_a = obsidian_client.get_note(path_a)
        if not re.search(re.escape(link_to_b), content_a):
            content_a = content_a.rstrip() + f"\n\n{link_to_b}"
            obsidian_client.put_note(path_a, content_a)

        content_b = obsidian_client.get_note(path_b)
        if not re.search(re.escape(link_to_a), content_b):
            content_b = content_b.rstrip() + f"\n\n{link_to_a}"
            obsidian_client.put_note(path_b, content_b)

        return f"Linked: {path_a} <-> {path_b}"
    except Exception as e:
        log_error(log, f"create_backlink FAILED: {path_a} <-> {path_b}", exc=e)
        return f"Error: {e}"


@mcp.tool()
def sync_index() -> str:
    """Re-run the full indexer pipeline. Clears the embedding cache and BM25 index."""
    log.info("sync_index — starting")
    try:
        indexer.run_index()
        entity_store.rebuild()
        llm_client.clear_embed_cache()
        keyword_search.ensure_index()  # force BM25 rebuild
        log.info("sync_index — caches cleared")
        log.info("sync_index — complete")
        return "Index sync complete. Caches cleared. Check indexer.log for details."
    except Exception as e:
        log_error(log, "sync_index FAILED", exc=e)
        return f"Error: {e}"


@mcp.tool()
def switch_embedding_model(model_name: str) -> str:
    """Switch the embedding model at runtime.

    Updates the config, clears the embed cache, resets the ChromaDB collection,
    and re-indexes the entire vault with the new model.

    The model must already exist in Ollama (pull it first with ``ollama pull <name>``).
    """
    log.info(f"switch_embedding_model — {model_name}")
    try:
        # Verify the model exists in Ollama
        resp = requests.post(
            f"{config.ollama_base_url}/api/embeddings",
            json={"model": model_name, "prompt": "test"},
            timeout=30,
        )
        if resp.status_code != 200:
            return f"Model '{model_name}' not found in Ollama. Pull it first: ollama pull {model_name}"

        old_model = config.ollama_embed_model
        llm_client.switch_embed_model(model_name)
        chroma_store.reset_collection()
        keyword_search.ensure_index()
        indexer.run_index()
        entity_store.rebuild()
        keyword_search.ensure_index()

        log.info(f"switch_embedding_model — switched {old_model} -> {model_name}")
        return f"Switched embedding model: {old_model} → {model_name}. Vault re-indexed."
    except Exception as e:
        log_error(log, f"switch_embedding_model FAILED: {model_name}", exc=e)
        return f"Error: {e}"


@mcp.tool()
def ask_agent(query: str) -> str:
    """Route a query to the best tool automatically using an LLM agent.

    The agent decides whether to use search_notes, summarize_topic,
    search_entities, related_notes, read_note, or ask_vault based on
    the user's intent. Tool results are returned directly.
    """
    log.info(f"ask_agent — {query}")
    try:
        result = pipelines.route_query(query)
        log.info(f"ask_agent — done, {len(result)} chars")
        return result
    except Exception as e:
        log_error(log, "ask_agent FAILED", exc=e, query=query)
        return f"Error: {e}"


@mcp.tool()
def ask_vault(
    question: str,
    top_k: int = 3,
    use_graph: bool = False,
    graph_depth: int = 1,
    use_entities: bool = False,
    entity_types: list[str] | None = None,
    keyword_weight: float = 0.0,
    expand_query: bool = False,
) -> str:
    """Ask a question about your Obsidian vault. Searches relevant notes and uses LLM to answer.

    Uses the multi-strategy retrieval pipeline combining semantic search,
    entity lookup, and wiki-link graph traversal.

    Args:
        question: the question to answer.
        top_k: number of top notes to retrieve.
        use_graph: if True, expand results by following wiki-links to find connected notes.
        graph_depth: max hops for graph traversal (default 1).
        use_entities: if True, also search the entity index for matching entities.
        entity_types: optional list of entity types to filter by (e.g. ``["Person"]``).
        keyword_weight: blend ratio for BM25 keyword search (0.0 = pure semantic, 1.0 = pure keyword).
        expand_query: if True, use the LLM to generate alternative query phrasings for broader search.
    """
    log.info(
        "ask_vault — question=%s, top_k=%s, use_graph=%s, graph_depth=%s, use_entities=%s, entity_types=%s, keyword_weight=%s, expand_query=%s",
        question, top_k, use_graph, graph_depth, use_entities, entity_types, keyword_weight, expand_query,
    )
    try:
        answer = pipelines.query(
            ask=question, top_k=top_k,
            use_graph=use_graph, graph_depth=graph_depth,
            use_entities=use_entities, entity_types=entity_types,
            keyword_weight=keyword_weight, expand_query=expand_query,
        )
        log.info(f"ask_vault — done, {len(answer)} chars")
        return answer
    except Exception as e:
        log_error(log, "ask_vault FAILED", exc=e, question=question)
        return f"Error: {e}"


@mcp.tool()
def retrieve_notes(
    query: str,
    top_k: int = 5,
    use_graph: bool = False,
    graph_depth: int = 1,
    graph_weight: float = 0.2,
    use_entities: bool = False,
    entity_types: list[str] | None = None,
    keyword_weight: float = 0.0,
    min_similarity: float | None = None,
    expand_query: bool = False,
) -> list[dict]:
    """Multi-strategy retrieval pipeline combining semantic search, entity lookup,
    and wiki-link graph traversal into a single unified result set.

    Returns note-level results (not chunks), each tagged with which strategy
    found it (``matched_by`` field) and a blended similarity score.

    Args:
        query: the search query or topic.
        top_k: max notes to return (default 5).
        use_graph: if True, expand via wiki-link graph traversal.
        graph_depth: max BFS hops when use_graph is True (default 1).
        graph_weight: weight for graph proximity boost, 0.0-1.0 (default 0.2).
        use_entities: if True, search the entity index for matching entities.
        entity_types: optional list of entity types to filter by (e.g. ``["Person"]``).
        keyword_weight: BM25 keyword blend, 0.0-1.0 (0.0 = pure semantic, 1.0 = pure keyword).
        min_similarity: minimum similarity score threshold (0-1). Results below are filtered out.
        expand_query: if True, use LLM to expand the query with synonyms for broader search.

    Returns:
        A list of dicts, each with:
        - ``path`` — vault-relative note path
        - ``title`` — note title (basename without extension)
        - ``content`` — full note content (truncated to context budget)
        - ``similarity_score`` — 0-to-1 blended score (higher = more relevant)
        - ``matched_by`` — list of strategies that found this note (e.g. ``["semantic", "entity"]``)
    """
    log.info(
        "retrieve_notes — query=%s, top_k=%s, use_graph=%s, graph_depth=%s, graph_weight=%s, use_entities=%s, entity_types=%s, keyword_weight=%s, min_similarity=%s, expand_query=%s",
        query, top_k, use_graph, graph_depth, graph_weight, use_entities, entity_types, keyword_weight, min_similarity, expand_query,
    )
    try:
        result = pipelines.retrieve(
            query=query, top_k=top_k,
            use_graph=use_graph, graph_depth=graph_depth, graph_weight=graph_weight,
            use_entities=use_entities, entity_types=entity_types,
            keyword_weight=keyword_weight, min_similarity=min_similarity,
            expand_query=expand_query,
        )
        notes = result["notes"] if result else []
        log.info("retrieve_notes — %s notes returned", len(notes))
        return notes
    except Exception as e:
        log_error(log, "retrieve_notes FAILED", exc=e, query=query)
        return []


@mcp.tool()
def tag_notes(query: str, top_k: int = 5) -> str:
    """Search notes matching a query and auto-suggest tags using LLM.

    Args:
        query: search query to find relevant notes.
        top_k: number of notes to process (default 5).

    Returns:
        Confirmation message with the tag map.
    """
    log.info(f"tag_notes — query={query!r}, top_k={top_k}")
    try:
        result = pipelines.tag_notes(query, top_k=top_k)
        log.info(f"tag_notes — done")
        return result
    except Exception as e:
        log_error(log, "tag_notes FAILED", exc=e, query=query)
        return f"Error: {e}"


@mcp.tool()
def summarize_topic(
    topic: str,
    top_k: int = 5,
    use_graph: bool = True,
    graph_depth: int = 1,
    graph_weight: float = 0.2,
    use_entities: bool = True,
    entity_types: list[str] | None = None,
    keyword_weight: float = 0.0,
    expand_query: bool = False,
) -> str:
    """Search all notes related to a topic and return an LLM-generated consolidated summary.

    Uses the multi-strategy retrieval pipeline combining semantic search,
    entity lookup, and wiki-link graph traversal.

    Args:
        topic: the topic or subject to summarize.
        top_k: number of notes to retrieve for context (default 5).
        use_graph: if True, expand results via wiki-link graph traversal (default True).
        graph_depth: max hops for graph traversal (default 1).
        graph_weight: weight for graph proximity boost, 0.0-1.0 (default 0.2).
        use_entities: if True, also search the entity index (default True).
        entity_types: optional list of entity types to filter by (e.g. ``["Person"]``).
        keyword_weight: BM25 keyword blend, 0.0-1.0 (0.0 = pure semantic, 1.0 = pure keyword).
        expand_query: if True, use LLM to expand the query with synonyms for broader search.

    Returns:
        An LLM-generated summary of the topic across related notes.
    """
    log.info(
        "summarize_topic — topic=%s, top_k=%s, use_graph=%s, graph_depth=%s, graph_weight=%s, use_entities=%s, entity_types=%s, keyword_weight=%s, expand_query=%s",
        topic, top_k, use_graph, graph_depth, graph_weight, use_entities, entity_types, keyword_weight, expand_query,
    )
    try:
        result = pipelines.summarize_topic(
            topic=topic, top_k=top_k,
            use_graph=use_graph, graph_depth=graph_depth, graph_weight=graph_weight,
            use_entities=use_entities, entity_types=entity_types,
            keyword_weight=keyword_weight, expand_query=expand_query,
        )
        log.info(f"summarize_topic — done, {len(result)} chars")
        return result
    except Exception as e:
        log_error(log, "summarize_topic FAILED", exc=e, topic=topic)
        return f"Error: {e}"


@mcp.tool()
def get_subject(subject: str, top_k: int = 10, keyword_weight: float = 0.3, group_by_note: bool = False) -> list[dict]:
    """Get notes related to a free-form subject.

    Uses LLM to expand the subject with related terms, then performs hybrid
    search (semantic + BM25) across the vault.

    Args:
        subject: the subject or topic to search for.
        top_k: max results to return.
        keyword_weight: blend ratio for BM25 keyword search (0.0 = pure semantic, 1.0 = pure keyword).
        group_by_note: if True, collapse chunk-level results into note-level (default False).

    Returns:
        List of dicts with path, title, similarity_score, snippet.
        When ``group_by_note=True``, also includes ``chunk_count``.
    """
    log.info(f"get_subject — subject={subject}, top_k={top_k}, keyword_weight={keyword_weight}, group_by_note={group_by_note}")
    try:
        expanded = _expand_query(subject)
        queries = [subject] + expanded
        log.info(f"get_subject — expanded to: {queries}")

        passages = _hybrid_search(
            queries=queries,
            n=top_k,
            keyword_weight=keyword_weight,
        )

        if group_by_note and passages:
            passages = _group_by_note(passages, top_k)

        log.info("get_subject — %s results returned", len(passages))
        return passages
    except Exception as e:
        log_error(log, "get_subject FAILED", exc=e, subject=subject)
        return []


# ── Entity Tools ────────────────────────────────────────────────────


@mcp.tool()
def search_entities(
    entity_name: str,
    entity_type: str | None = None,
    n: int = 10,
    use_graph: bool = False,
) -> list[dict]:
    """Find notes mentioning a specific entity.

    Uses the entity index store for fast lookups, with ChromaDB fallback
    via the ``entities_str`` metadata field.

    Args:
        entity_name: name of the entity to search for (e.g. ``"ESP32"``, ``"Alice"``).
        entity_type: optional filter (e.g. ``"Person"``, ``"Hardware"``). One of:
                     Person, Project, Hardware, Technology, Location, Concept, Event.
        n: max results to return.
        use_graph: if True, also traverse wiki-links from matching notes to
                   find connected notes that may mention the entity indirectly.

    Returns:
        List of dicts with path, title, entity_name, entity_type, snippet, confidence.
    """
    log.info(f"search_entities — {entity_name}, type={entity_type}, n={n}, use_graph={use_graph}")
    try:
        results = entity_store.search(entity_name, type=entity_type, n=n * 2)

        # If entity store has few results, try ChromaDB $contains fallback
        if len(results) < n and chroma_store._collection is not None:
            where = {"entities_str": {"$contains": f",{entity_name},"}} if entity_type is None else {}
            if entity_type:
                where = {"entities_str": {"$contains": f",{entity_type}:{entity_name},"}}
            try:
                raw = chroma_store._collection.get(where=where, include=["metadatas", "documents"])
                seen_paths = {r["path"] for r in results}
                for i in range(len(raw["ids"])):
                    path = raw["metadatas"][i]["path"]
                    if path not in seen_paths:
                        seen_paths.add(path)
                        results.append({
                            "path": path,
                            "entity_name": entity_name,
                            "entity_type": entity_type or "Concept",
                            "snippet": _truncate_snippet((raw["documents"] or [""])[i] or ""),
                            "confidence": 0.5,
                        })
            except Exception:
                pass

        results = results[:n]

        # Graph expansion
        if use_graph and results:
            result_paths = {r["path"] for r in results}
            for r in list(results):
                neighbors = graph_store.bfs(r["path"], max_depth=1)
                for neighbor_path in neighbors:
                    if neighbor_path not in result_paths:
                        result_paths.add(neighbor_path)
                        title = os.path.splitext(os.path.basename(neighbor_path))[0]
                        results.append({
                            "path": neighbor_path,
                            "title": title,
                            "entity_name": entity_name,
                            "entity_type": entity_type or "Concept",
                            "snippet": "(graph-connected)",
                            "confidence": 0.3,
                        })
            results = results[:n]

        log.info(f"search_entities — {len(results)} results")
        return results
    except Exception as e:
        log_error(log, "search_entities FAILED", exc=e, entity_name=entity_name)
        return []


@mcp.tool()
def get_note_entities(path: str) -> list[dict]:
    """Return all entities found in a specific note during indexing.

    Args:
        path: vault-relative path to the note.

    Returns:
        List of dicts with entity_name, entity_type, confidence.
    """
    log.info(f"get_note_entities — {path}")
    try:
        results = entity_store.get_note_entities(path)
        log.info(f"get_note_entities — {len(results)} entities")
        return results
    except Exception as e:
        log_error(log, "get_note_entities FAILED", exc=e, path=path)
        return []


@mcp.tool()
def get_entity_types() -> list[str]:
    """Return all available entity types used for classification.

    Returns:
        List of entity type strings.
    """
    return entity_store.entity_types()


# ── Graph RAG Tools ──────────────────────────────────────────────────


@mcp.tool()
def get_backlinks(path: str) -> list[dict]:
    """Return all notes linking TO the given note (incoming wiki-link edges).

    Args:
        path: vault-relative path to the note (e.g., "Folder/Note.md").

    Returns:
        List of dicts with path, title, and trace (the path from source to target).
    """
    log.info(f"get_backlinks — {path}")
    try:
        sources = graph_store.get_backlinks(path)
        results = []
        for src in sources:
            title = os.path.splitext(os.path.basename(src))[0]
            results.append({"path": src, "title": title})
        log.info(f"get_backlinks — {len(results)} results")
        return results
    except Exception as e:
        log_error(log, "get_backlinks FAILED", exc=e, path=path)
        return []


@mcp.tool()
def get_linked_notes(path: str) -> list[dict]:
    """Return all notes the given note links TO (outgoing wiki-link edges).

    Args:
        path: vault-relative path to the note.

    Returns:
        List of dicts with path and title.
    """
    log.info(f"get_linked_notes — {path}")
    try:
        targets = graph_store.get_outgoing(path)
        results = []
        for tgt in targets:
            title = os.path.splitext(os.path.basename(tgt))[0]
            results.append({"path": tgt, "title": title})
        log.info(f"get_linked_notes — {len(results)} results")
        return results
    except Exception as e:
        log_error(log, "get_linked_notes FAILED", exc=e, path=path)
        return []


@mcp.tool()
def get_broken_links() -> list[dict]:
    """Find wiki-links across all notes that don't resolve to any existing note.

    Returns:
        List of dicts with source_path and link_target (the unresolved wiki-link).
    """
    log.info("get_broken_links")
    try:
        notes = obsidian_client.list_all_notes()
        all_contents: dict[str, str] = {}
        for path in notes:
            try:
                all_contents[path] = obsidian_client.get_note(path)
            except Exception:
                pass
        broken = graph_store.get_broken_links(all_contents)
        log.info(f"get_broken_links — {len(broken)} broken links")
        return broken
    except Exception as e:
        log_error(log, "get_broken_links FAILED", exc=e)
        return []


@mcp.tool()
def get_graph_stats() -> dict:
    """Return graph statistics: total nodes, edges, average degree, isolated notes, and hubs.

    Returns:
        Dict with nodes, edges, avg_degree, isolated_count, isolated (list), hubs (top 5 by degree).
    """
    log.info("get_graph_stats")
    try:
        stats = graph_store.stats()
        log.info(f"get_graph_stats — {stats['nodes']} nodes, {stats['edges']} edges")
        return stats
    except Exception as e:
        log_error(log, "get_graph_stats FAILED", exc=e)
        return {}


@mcp.tool()
def multi_hop_traversal(path: str, max_depth: int = 2) -> list[dict]:
    """Perform BFS graph traversal from a seed note up to N hops.

    Returns all reachable notes with path traces for explainability
    (e.g., A -> B -> C shows the chain of wiki-links).

    Args:
        path: vault-relative path to the seed note.
        max_depth: maximum number of hops to traverse (default 2).

    Returns:
        List of dicts with path, title, depth (hop count), and trace (list of paths from seed).
    """
    log.info(f"multi_hop_traversal — {path}, depth={max_depth}")
    try:
        results = graph_store.bfs(path, max_depth=max_depth)
        output = []
        for tgt_path, trace in results.items():
            title = os.path.splitext(os.path.basename(tgt_path))[0]
            output.append({
                "path": tgt_path,
                "title": title,
                "depth": len(trace) - 1,
                "trace": trace,
            })
        log.info(f"multi_hop_traversal — {len(output)} reachable notes")
        return output
    except Exception as e:
        log_error(log, "multi_hop_traversal FAILED", exc=e, path=path)
        return []


@mcp.tool()
def related_notes(path: str, k: int = 10, graph_weight: float = 0.3) -> list[dict]:
    """Find notes related to a given note using both semantic similarity and graph proximity.

    Combines embedding-based semantic search with wiki-link graph traversal.
    Notes connected via wiki-links get a proximity boost proportional to graph_weight.

    Args:
        path: vault-relative path to the source note.
        k: number of results to return.
        graph_weight: how much to weight graph proximity (0.0 = pure semantic, 1.0 = pure graph).

    Returns:
        List of dicts with path, title, similarity_score, and is_graph_connected (bool).
    """
    log.info(f"related_notes — {path}, k={k}, graph_weight={graph_weight}")
    try:
        # Get the source note's title
        title = os.path.splitext(os.path.basename(path))[0]

        # Semantic search: get notes with same title (the source note's chunks)
        source_docs = chroma_store.get_by_title(title)
        if not source_docs:
            log.warning(f"related_notes — source note not found in index: {path}")
            return []

        # Use the first chunk's embedding to search
        # Since we don't store embeddings separately, we search by title proximity
        # Get graph neighbors
        graph_neighbors = set(graph_store.get_outgoing(path))
        graph_neighbors.update(graph_store.get_backlinks(path))
        graph_neighbors.discard(path)

        # Semantic search using the source note's content
        source_content = obsidian_client.get_note(path)
        meta, body = fm_parse(source_content)
        query_text = body[:500]  # Use first 500 chars as query

        semantic_results = _hybrid_search(
            queries=[query_text],
            n=k * 2,
            keyword_weight=0.3,
        )

        # Filter out the source note itself
        semantic_results = [r for r in semantic_results if r["path"] != path]

        # Score blending
        seen: dict[str, dict] = {}
        for rank, result in enumerate(semantic_results):
            p = result["path"]
            semantic_score = result.get("similarity_score", 0.5)
            graph_score = 1.0 if p in graph_neighbors else 0.0
            combined = semantic_score * (1 - graph_weight) + graph_score * graph_weight

            if p not in seen or combined > seen[p]["similarity_score"]:
                seen[p] = {
                    "path": p,
                    "title": result["title"],
                    "similarity_score": round(combined, 4),
                    "is_graph_connected": p in graph_neighbors,
                }

        # Add graph-only neighbors not in semantic results
        for neighbor in graph_neighbors:
            if neighbor not in seen:
                seen[neighbor] = {
                    "path": neighbor,
                    "title": os.path.splitext(os.path.basename(neighbor))[0],
                    "similarity_score": round(graph_weight, 4),
                    "is_graph_connected": True,
                }

        results = sorted(seen.values(), key=lambda x: x["similarity_score"], reverse=True)[:k]
        log.info(f"related_notes — {len(results)} results")
        return results
    except Exception as e:
        log_error(log, "related_notes FAILED", exc=e, path=path)
        return []


@mcp.tool()
def export_graph(format: str = "json") -> str:
    """Export the wiki-link graph in DOT or JSON format for external visualization.

    Args:
        format: ``"dot"`` for Graphviz DOT format, ``"json"`` for JSON.

    Returns:
        The graph representation as a string.
    """
    log.info(f"export_graph — format={format}")
    try:
        if format == "dot":
            return graph_store.to_dot()
        data = graph_store.to_dict()
        import json
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        log_error(log, "export_graph FAILED", exc=e, format=format)
        return f"Error: {e}"


@mcp.tool()
def get_orphan_notes() -> list[str]:
    """Find notes with no incoming or outgoing wiki-links (orphans).

    Useful for vault cleanup — these notes are disconnected from the rest.
    """
    log.info("get_orphan_notes")
    try:
        orphans = graph_store.get_orphans()
        log.info(f"get_orphan_notes — {len(orphans)} orphans")
        return orphans
    except Exception as e:
        log_error(log, "get_orphan_notes FAILED", exc=e)
        return []


@mcp.tool()
def get_communities() -> dict[str, list[str]]:
    """Detect communities in the wiki-link graph using label propagation.

    Notes in the same community are densely connected via wiki-links.
    Useful for understanding vault structure and grouping related notes.

    Returns:
        Dict mapping community IDs (integers as strings) to lists of note paths.
    """
    log.info("get_communities")
    try:
        communities = graph_store.label_propagation()
        result: dict[str, list[str]] = {}
        for path, cid in communities.items():
            key = str(cid)
            result.setdefault(key, []).append(path)
        for k in result:
            result[k].sort()
        log.info(f"get_communities — {len(result)} communities found")
        return result
    except Exception as e:
        log_error(log, "get_communities FAILED", exc=e)
        return {}


# ── Todo Tools ───────────────────────────────────────────────────────


@mcp.tool()
def ensure_todo_file() -> str:
    """Create a default todos.md file in the vault if it doesn't exist."""
    log.info("ensure_todo_file")
    try:
        result = td_ensure()
        log.info(f"ensure_todo_file — {result}")
        return result
    except Exception as e:
        log_error(log, "ensure_todo_file FAILED", exc=e)
        return f"Error: {e}"


@mcp.tool()
def get_todos(project: str = "", status: str = "", overdue: bool = False, blocked: bool = False, search: str = "") -> list[dict]:
    """List todos from todos.md, optionally filtered by project and/or status.

    Args:
        project: filter by project name (case-sensitive). Empty string = all projects.
        status: ``"pending"`` or ``"completed"``. Empty string = all statuses.
        overdue: if True, only return pending todos past their due date.
        blocked: if True, only return todos with "blocked"/"blocking" in the task text or a "blocked" tag.
        search: free-text search against task description, tags, and project name (case-insensitive).

    Returns:
        List of todo dicts, each with id, task, status, due, priority, tags, project.
    """
    log.info(f"get_todos — project={project!r}, status={status!r}, overdue={overdue}, blocked={blocked}, search={search!r}")
    try:
        proj = project if project else None
        st = status if status else None
        todos = td_get(project=proj, status=st, overdue=overdue, blocked=blocked, search=search)
        log.info(f"get_todos — {len(todos)} results")
        return todos
    except Exception as e:
        log_error(log, "get_todos FAILED", exc=e)
        return []


@mcp.tool()
def add_todo(project: str, task: str, due: str = "", priority: str = "", tags: list[str] | None = None) -> dict:
    """Add a new todo task to a project.

    Args:
        project: project name (e.g. ``"Work"``, ``"Personal"``). Created if it doesn't exist.
        task: the task description.
        due: optional due date in ``YYYY-MM-DD`` format.
        priority: ``"high"``, ``"medium"``, or ``"low"``.
        tags: optional list of tag strings.

    Returns:
        The created todo dict with its assigned id.
    """
    log.info(f"add_todo — project={project!r}, task={task!r}")
    try:
        todo = td_add(
            project=project,
            task=task,
            due=due if due else None,
            priority=priority if priority else None,
            tags=tags or None,
        )
        log.info(f"add_todo — created {todo['id']}")
        return todo
    except Exception as e:
        log_error(log, "add_todo FAILED", exc=e)
        return {"error": str(e)}


@mcp.tool()
def complete_todo(todo_id: str) -> dict:
    """Mark a todo as completed by its id.

    Args:
        todo_id: the id of the todo (returned by get_todos or add_todo).

    Returns:
        The updated todo dict, or an error dict if the id is not found.
    """
    log.info(f"complete_todo — {todo_id}")
    try:
        todo = td_complete(todo_id)
        if todo is None:
            return {"error": f"Todo not found: {todo_id}"}
        log.info(f"complete_todo — {todo_id} done")
        return todo
    except Exception as e:
        log_error(log, "complete_todo FAILED", exc=e)
        return {"error": str(e)}


@mcp.tool()
def update_todo(todo_id: str, task: str = "", due: str = "", priority: str = "", tags: list[str] | None = None, project: str = "", status: str = "") -> dict:
    """Update one or more fields of an existing todo.

    Args:
        todo_id: the id of the todo to update.
        task: new task description (leave empty to keep current).
        due: new due date (leave empty to keep current).
        priority: new priority (leave empty to keep current).
        tags: new tags list (leave empty to keep current).
        project: move to a different project (leave empty to keep current).
        status: ``"pending"`` or ``"completed"`` (leave empty to keep current).

    Returns:
        The updated todo dict, or an error dict if the id is not found.
    """
    log.info(f"update_todo — {todo_id}")
    try:
        kwargs: dict = {}
        if task:
            kwargs["task"] = task
        if due:
            kwargs["due"] = due
        if priority:
            kwargs["priority"] = priority
        if tags is not None:
            kwargs["tags"] = tags
        if project:
            kwargs["project"] = project
        if status:
            kwargs["status"] = status
        todo = td_update(todo_id, **kwargs)
        if todo is None:
            return {"error": f"Todo not found: {todo_id}"}
        log.info(f"update_todo — {todo_id} updated")
        return todo
    except Exception as e:
        log_error(log, "update_todo FAILED", exc=e)
        return {"error": str(e)}


@mcp.tool()
def delete_todo(todo_id: str) -> dict:
    """Delete a todo by its id.

    Args:
        todo_id: the id of the todo to delete.

    Returns:
        A dict with ``success: true`` or ``success: false`` with an error message.
    """
    log.info(f"delete_todo — {todo_id}")
    try:
        ok = td_delete(todo_id)
        if not ok:
            return {"success": False, "error": f"Todo not found: {todo_id}"}
        log.info(f"delete_todo — {todo_id} deleted")
        return {"success": True}
    except Exception as e:
        log_error(log, "delete_todo FAILED", exc=e)
        return {"success": False, "error": str(e)}


@mcp.tool()
def sync_todos() -> dict:
    """Recalculate todo counts in the todos.md frontmatter and rewrite the file."""
    log.info("sync_todos")
    try:
        result = td_sync()
        log.info(f"sync_todos — {result}")
        return result
    except Exception as e:
        log_error(log, "sync_todos FAILED", exc=e)
        return {"error": str(e)}


@mcp.tool()
def get_todo_stats() -> dict:
    """Return aggregated statistics about all todos in the vault.

    Returns:
        Dict with total, completed, pending, overdue counts, per-project breakdown,
        per-priority breakdown, due-date stats, and tag frequency.
    """
    log.info("get_todo_stats")
    try:
        result = td_stats()
        log.info(f"get_todo_stats — {result['total']} todos")
        return result
    except Exception as e:
        log_error(log, "get_todo_stats FAILED", exc=e)
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()

