import os
import re
from datetime import datetime

from fastmcp import FastMCP

from . import chroma_store, config, indexer, keyword_search, llm_client, obsidian_client, pipelines
from .frontmatter import add_tags as fm_add_tags
from .frontmatter import parse as fm_parse
from .frontmatter import remove_tags as fm_remove_tags
from .frontmatter import set_tags as fm_set_tags
from .logger import get_logger, log_error

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


# ── tools ──────────────────────────────────────────────────────────


@mcp.tool()
def get_index_stats() -> str:
    """Return index statistics (total chunks, unique notes, config info, cache stats)."""
    log.info("get_index_stats")
    try:
        stats = chroma_store.get_index_stats()
        cache = llm_client.embed_cache_info()
        lines = [
            f"Unique notes indexed: {stats['unique_notes']}",
            f"Total chunks stored:  {stats['total_chunks']}",
            f"Embedding model:      {config.ollama_embed_model}",
            f"ChromaDB path:        {config.chroma_path}",
            f"Embedding cache:      {cache['currsize']}/{cache['maxsize']} (hits={cache['hits']}, misses={cache['misses']})",
        ]
        result = "\n".join(lines)
        log.info(f"get_index_stats — {result.replace(chr(10), ' | ')}")
        return result
    except Exception as e:
        log_error(log, "get_index_stats FAILED", exc=e)
        return f"Error: {e}"


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
) -> list[dict]:
    """Search notes semantically with optional metadata filters.

    Returns passage-level results — each result is a single matching chunk
    with its snippet, chunk index within the note, and similarity score.
    Multiple passages from the same note may appear.

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

    Returns:
        A list of dicts, each with:
        - ``path`` — vault-relative note path
        - ``title`` — note title (basename without extension)
        - ``matched_chunk_idx`` — index of the matching chunk within the note
        - ``similarity_score`` — 0-to-1 score (higher = more relevant)
        - ``snippet`` — the matching passage text, trimmed to ~400 chars
    """
    log.info(
        "search_notes — query=%s, n=%s, tags=%s, exclude_tags=%s, folder=%s, date_after=%s, date_before=%s, expand=%s, kw_weight=%s, min_sim=%s, div_penalty=%s",
        query, n, tags, exclude_tags, folder, date_after, date_before,
        expand_query, keyword_weight, min_similarity, diversity_penalty,
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
        log.info("search_notes — %s passages returned", len(passages))
        return passages
    except Exception as e:
        log_error(log, "search_notes FAILED", exc=e, query=query, n=n)
        return []


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
        llm_client.clear_embed_cache()
        keyword_search.ensure_index()  # force BM25 rebuild
        log.info("sync_index — caches cleared")
        log.info("sync_index — complete")
        return "Index sync complete. Caches cleared. Check indexer.log for details."
    except Exception as e:
        log_error(log, "sync_index FAILED", exc=e)
        return f"Error: {e}"


@mcp.tool()
def ask_vault(question: str, top_k: int = 3) -> str:
    """Ask a question about your Obsidian vault. Searches relevant notes and uses LLM to answer."""
    log.info(f"ask_vault — {question}")
    try:
        answer = pipelines.query(question, top_k=top_k)
        log.info(f"ask_vault — done, {len(answer)} chars")
        return answer
    except Exception as e:
        log_error(log, "ask_vault FAILED", exc=e, question=question)
        return f"Error: {e}"


@mcp.tool()
def tag_notes(query: str, top_k: int = 5) -> str:
    """Search notes matching a query and auto-suggest tags using LLM."""
    log.info(f"tag_notes — {query}")
    try:
        result = pipelines.tag_notes(query, top_k=top_k)
        log.info("tag_notes — done")
        return result
    except Exception as e:
        log_error(log, "tag_notes FAILED", exc=e, query=query)
        return f"Error: {e}"


@mcp.tool()
def get_subject(subject: str, top_k: int = 10, keyword_weight: float = 0.3) -> list[dict]:
    """Get notes related to a free-form subject.

    Uses LLM to expand the subject with related terms, then performs hybrid
    search (semantic + BM25) across the vault.

    Args:
        subject: the subject or topic to search for.
        top_k: max results to return.
        keyword_weight: blend ratio for BM25 keyword search (0.0 = pure semantic, 1.0 = pure keyword).

    Returns:
        List of dicts with path, title, similarity_score, snippet.
    """
    log.info(f"get_subject — subject={subject}, top_k={top_k}, keyword_weight={keyword_weight}")
    try:
        # Expand the subject using LLM
        expanded = _expand_query(subject)
        queries = [subject] + expanded
        log.info(f"get_subject — expanded to: {queries}")

        passages = _hybrid_search(
            queries=queries,
            n=top_k,
            keyword_weight=keyword_weight,
        )
        log.info("get_subject — %s passages returned", len(passages))
        return passages
    except Exception as e:
        log_error(log, "get_subject FAILED", exc=e, subject=subject)
        return []


if __name__ == "__main__":
    mcp.run()

