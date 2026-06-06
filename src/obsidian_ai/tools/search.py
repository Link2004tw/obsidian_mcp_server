"""Search and retrieval tools for the Obsidian MCP server."""

import re

from .. import (
    chroma_store,
    config,
    entity_store,
    indexer,
    llm_client,
    pipelines,
    ranker,
)
from ..logger import get_logger, log_error
from ._shared import (
    _apply_entity_search,
    _apply_graph_boost,
    _apply_summary_search,
    _build_search_where,
    _expand_query,
    _group_by_note,
    _hybrid_search,
)

log = get_logger("obsidian_ai.tools.search")


def get_index_stats() -> str:
    """Get diagnostic statistics about the note index. Use this when the user asks about how many notes are indexed, what embedding model is used, or wants to check the health/status of the vault index.

    Returns:
        A human-readable string listing unique notes, total chunks, embedding model,
        ChromaDB path, cache hit/miss stats, and entity extraction counts.
    """
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


def find_duplicate_notes(threshold: float = 0.9, n: int = 20) -> list[dict]:
    """Find near-duplicate notes via embedding similarity. Use this when the user wants to clean up their vault by finding notes that are very similar or near-duplicates.

    Args:
        threshold: cosine similarity threshold between 0.0 and 1.0 (default 0.9). Higher values require more similarity to count as a duplicate.
        n: maximum number of duplicate pairs to return (default 20).

    Returns:
        A list of dicts, each with fields like ``path`` and ``similarity_score`` describing a near-duplicate pair.
    """
    log.info(f"find_duplicate_notes — threshold={threshold}, n={n}")
    try:
        results = chroma_store.find_duplicate_notes(threshold=threshold, n=n)
        log.info(f"find_duplicate_notes — {len(results)} pairs found")
        return results
    except Exception as e:
        log_error(log, "find_duplicate_notes FAILED", exc=e)
        return []


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
    expand_entities: bool = False,
    use_summaries: bool = False,
    summary_threshold: float = 0.7,
) -> list[dict]:
    """Search notes by semantic meaning (embedding similarity) with optional metadata filters. Use this when the user wants to find notes about a topic, concept, or idea — it understands meaning, not just keywords.

    By default returns passage-level results — each result is a single matching chunk.
    Multiple passages from the same note may appear. Pass ``group_by_note=True`` to collapse
    chunk-level results into one result per note with the highest similarity score, best
    snippet, and a ``chunk_count``.

    Args:
        query: the natural language search query describing what to find.
        n: maximum number of results to return (default 5).
        tags: optional list of YAML frontmatter tags — only notes having ALL of these tags will be returned.
        exclude_tags: optional list of tags — notes having ANY of these tags will be excluded.
        folder: optional vault-relative folder path — only search notes inside this folder.
        date_after: optional ISO date string (e.g. ``"2024-06-01"``) — only notes modified on or after this date.
        date_before: optional ISO date string (e.g. ``"2024-06-01"``) — only notes modified on or before this date.
        expand_query: if True, use the LLM to generate alternative query phrasings for broader recall (adds ~1-2s).
        keyword_weight: blend ratio for BM25 keyword search between 0.0 and 1.0 (0.0 = pure semantic, 1.0 = pure keyword, default 0.0).
        min_similarity: optional minimum similarity score threshold (0-1). Results below this are filtered out.
        diversity_penalty: diversity penalty factor between 0.0 and 1.0 (0.0 = none, 0.5 = moderate, 1.0 = aggressive). Penalises passages from notes that already have results selected, encouraging diverse sources.
        use_graph: if True, expand results by following [[wiki-links]] from matching notes to find connected notes.
        graph_depth: maximum BFS hops for graph traversal when ``use_graph`` is True (default 1).
        graph_weight: weight for graph proximity boost between 0.0 and 1.0 (default 0.2).
        use_entities: if True, also search the entity index for notes mentioning entities that match the query.
        entity_types: optional list of entity types to filter by (e.g. ``["Person"]``, ``["Project", "Technology"]``).
        group_by_note: if True, collapse chunk-level results into one result per note (default False).
        expand_entities: if True, when entities are auto-detected in the query, also search for related entities via the entity relationship graph.
        use_summaries: if True, also search note-level semantic summaries as an additional retrieval signal.
        summary_threshold: minimum similarity score (0-1) for a summary result to be included (default 0.7).

    Returns:
        A list of dicts. Each dict contains:
        - ``path`` — vault-relative note path
        - ``title`` — note title (filename without extension)
        - ``similarity_score`` — 0-to-1 score (higher = more relevant)
        - ``snippet`` — excerpt of the matching passage
        - ``chunk_count`` — (only when ``group_by_note=True``) how many chunks matched in this note
    """
    log.info(
        "search_notes — query=%s, n=%s, tags=%s, exclude_tags=%s, folder=%s, date_after=%s, date_before=%s, expand=%s, kw_weight=%s, min_sim=%s, div_penalty=%s, use_graph=%s, graph_depth=%s, graph_weight=%s, use_entities=%s, entity_types=%s, group_by_note=%s, expand_entities=%s, use_summaries=%s, summary_threshold=%s",
        query, n, tags, exclude_tags, folder, date_after, date_before,
        expand_query, keyword_weight, min_similarity, diversity_penalty,
        use_graph, graph_depth, graph_weight, use_entities, entity_types, group_by_note, expand_entities, use_summaries, summary_threshold,
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

        # Entity-augmented expansion (auto-detects entities in query)
        passages = _apply_entity_search(
            passages, query,
            entity_types=entity_types,
            force=use_entities,
            expand_entities=expand_entities,
        )

        # Summary-embedding augmentation
        if use_summaries and passages is not None:
            passages = _apply_summary_search(
                passages, query, n=n, summary_threshold=summary_threshold,
            )

        # Graph-augmented expansion
        if use_graph and passages:
            passages = _apply_graph_boost(passages, graph_depth=graph_depth, graph_weight=graph_weight)

        # Group by note (collapse chunks) before final trim
        passages = _group_by_note(passages, n) if group_by_note and passages else passages[:n]

        log.info("search_notes — %s results returned", len(passages))
        return passages
    except Exception as e:
        log_error(log, "search_notes FAILED", exc=e, query=query, n=n)
        return []


def batch_search(
    queries: list[str],
    n: int = 5,
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    folder: str | None = None,
    keyword_weight: float = 0.0,
    min_similarity: float | None = None,
) -> dict[str, list[dict]]:
    """Run multiple independent semantic searches in a single call. Use this when you have several queries to run at once and want to batch them for efficiency. For per-query fine-grained options (graph, entities, expansion), use individual ``search_notes`` calls instead.

    Each query runs a hybrid search (semantic + BM25 keyword) with the same shared metadata filters.

    Args:
        queries: a list of query strings, each searched independently.
        n: maximum results per query (default 5).
        tags: optional list of YAML frontmatter tags — only notes having ALL of these tags.
        exclude_tags: optional list of tags to exclude.
        folder: optional vault-relative folder path to restrict search to.
        keyword_weight: BM25 keyword blend between 0.0 and 1.0 (default 0.0).
        min_similarity: optional minimum similarity threshold (0-1).

    Returns:
        A dict mapping each query string to its list of result dicts. Each result dict has
        ``path``, ``title``, ``similarity_score``, ``snippet``, and ``matched_chunk_idx``.
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


def composite_search(
    query: str,
    n: int = 5,
    retrieval_depth: int = 2,
) -> list[dict]:
    """High-recall composite search combining summary embeddings, entity relationships, and community-aware graph traversal. Use this when you need maximum recall — e.g., the user is looking for everything related to a topic and it's OK to get broad, diverse results.

    Blends three retrieval strategies into a single ranked set (note-level, not chunk-level):

    1. **Summary embeddings** — searches note-level semantic summaries.
    2. **Entity expansion** — auto-detects entities in the query and follows relationship edges (``works_on``, ``part_of``, etc.) to find related notes.
    3. **Community graph** — finds other notes in the same wiki-link communities as top results, surfacing thematically related notes that may not share direct keywords or links.

    Args:
        query: the topic or concept to search for.
        n: maximum number of results to return (default 5).
        retrieval_depth: controls which strategies to use (default 2).
            ``1`` = summary embeddings only (fast).
            ``2`` = summary + entity expansion (balanced).
            ``3`` = full composite with community graph (maximum recall).

    Returns:
        A list of dicts, each with:
        - ``path`` — vault-relative note path
        - ``title`` — note title (filename without extension)
        - ``score`` — 0-to-1 relevance score
        - ``matched_by`` — list of strategy names that found this note (e.g. ``["summary", "entity"]``)
    """
    log.info("composite_search — query=%r, n=%s, depth=%s", query, n, retrieval_depth)
    try:
        results = ranker.composite_search(
            query=query,
            n=n,
            retrieval_depth=retrieval_depth,
        )
        return results
    except Exception as e:
        log_error(log, "composite_search FAILED", exc=e, query=query)
        return []


def ask_agent(query: str) -> str:
    """Route a query to the best vault tool automatically using an LLM-powered agent. Use this when the user is vague about what they want or you're unsure which tool (``search_notes``, ``summarize_topic``, ``search_entities``, ``related_notes``, ``read_note``, ``ask_vault``, etc.) would best answer their request.

    The agent analyses the user's intent and selects the most appropriate tool automatically.

    Args:
        query: the user's natural language request or question about their vault.

    Returns:
        A string containing the result from whichever tool the agent chose to run.
    """
    log.info(f"ask_agent — {query}")
    try:
        result = pipelines.route_query(query)
        log.info(f"ask_agent — done, {len(result)} chars")
        return result
    except Exception as e:
        log_error(log, "ask_agent FAILED", exc=e, query=query)
        return f"Error: {e}"


def ask_vault(
    question: str,
    top_k: int = 3,
    use_graph: bool = False,
    graph_depth: int = 1,
    use_entities: bool = False,
    entity_types: list[str] | None = None,
    keyword_weight: float = 0.0,
    expand_query: bool = False,
    expand_entities: bool = False,
    use_summaries: bool = False,
    summary_threshold: float = 0.7,
    auto_weights: bool = False,
    auto_rewrite: bool = False,
) -> str:
    """Ask a question about the Obsidian vault and get an LLM-generated answer synthesised from relevant notes. Use this when the user asks a direct question about what's in their notes — it searches for relevant information and returns a natural language answer.

    Uses multi-strategy retrieval: semantic search, entity lookup, and optional wiki-link graph traversal. Results are fed to an LLM to produce a concise answer.

    Args:
        question: the natural language question to answer using the vault's contents.
        top_k: number of top notes to retrieve as context for the LLM (default 3).
        use_graph: if True, expand results by following [[wiki-links]] to find connected notes (default False).
        graph_depth: max BFS hops for graph traversal when ``use_graph`` is True (default 1).
        use_entities: if True, also search the entity index for notes mentioning entities matching the query (default False).
        entity_types: optional list of entity types to filter by (e.g. ``["Person"]``).
        keyword_weight: BM25 keyword blend between 0.0 and 1.0 (0.0 = pure semantic, 1.0 = pure keyword, default 0.0).
        expand_query: if True, use the LLM to generate alternative query phrasings for broader recall (default False).
        expand_entities: if True, when entities are auto-detected in the query, also search for related entities via the entity relationship graph (default False).
        use_summaries: if True, include note-level summary embeddings as an additional retrieval signal (default False).
        summary_threshold: minimum similarity score (0-1) for a summary result to be included (default 0.7).
        auto_weights: if True, detect query intent (entity-heavy, keyword-heavy, etc.) and adjust ranking weights dynamically (default False).
        auto_rewrite: if True, rewrite the query using known vault terminology before searching (default False).

    Returns:
        A string containing the LLM-generated answer synthesised from the retrieved notes.
    """
    log.info(
        "ask_vault — question=%s, top_k=%s, use_graph=%s, graph_depth=%s, use_entities=%s, entity_types=%s, keyword_weight=%s, expand_query=%s, expand_entities=%s, use_summaries=%s, summary_threshold=%s, auto_rewrite=%s",
        question, top_k, use_graph, graph_depth, use_entities, entity_types, keyword_weight, expand_query, expand_entities, use_summaries, summary_threshold, auto_rewrite,
    )
    try:
        answer = pipelines.query(
            ask=question, top_k=top_k,
            use_graph=use_graph, graph_depth=graph_depth,
            use_entities=use_entities, entity_types=entity_types,
            keyword_weight=keyword_weight, expand_query=expand_query,
            expand_entities=expand_entities,
            use_summaries=use_summaries, summary_threshold=summary_threshold,
            auto_rewrite=auto_rewrite,
        )
        log.info(f"ask_vault — done, {len(answer)} chars")
        return answer
    except Exception as e:
        log_error(log, "ask_vault FAILED", exc=e, question=question)
        return f"Error: {e}"


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
    expand_entities: bool = False,
    use_summaries: bool = False,
    summary_threshold: float = 0.7,
    auto_weights: bool = False,
    auto_rewrite: bool = False,
) -> list[dict]:
    """Multi-strategy retrieval that returns full note content. Use this when you need both the note content (full text) and metadata about which retrieval strategy found each note. Unlike ``search_notes`` which returns passage snippets, this returns entire note bodies.

    Combines semantic search, entity lookup, and wiki-link graph traversal into a single unified, note-level result set. Each result is tagged with which strategy(s) found it.

    Args:
        query: the search query or topic to find notes about.
        top_k: maximum number of notes to return (default 5).
        use_graph: if True, expand via [[wiki-link]] graph traversal (default False).
        graph_depth: max BFS hops for graph traversal when ``use_graph`` is True (default 1).
        graph_weight: weight for graph proximity boost between 0.0 and 1.0 (default 0.2).
        use_entities: if True, also search the entity index for notes matching entities in the query (default False).
        entity_types: optional list of entity types to filter by (e.g. ``["Person"]``, ``["Project"]``).
        keyword_weight: BM25 keyword blend between 0.0 and 1.0 (0.0 = pure semantic, 1.0 = pure keyword, default 0.0).
        min_similarity: optional minimum similarity score threshold (0-1). Results below this are filtered out.
        expand_query: if True, use the LLM to generate alternative query phrasings for broader recall (default False).
        expand_entities: if True, when entities are auto-detected in the query, also search for related entities via the entity relationship graph (default False).
        use_summaries: if True, include note-level summary embeddings as an additional retrieval signal (default False).
        summary_threshold: minimum similarity score (0-1) for a summary result to be included (default 0.7).
        auto_weights: if True, detect query intent and adjust ranking weights dynamically (default False).
        auto_rewrite: if True, rewrite the query using known vault terminology before searching (default False).

    Returns:
        A list of dicts, each with:
        - ``path`` — vault-relative note path
        - ``title`` — note title (filename without extension)
        - ``content`` — full note content (truncated to context budget)
        - ``similarity_score`` — 0-to-1 blended relevance score (higher = more relevant)
        - ``matched_by`` — list of strategies that found this note (e.g. ``["semantic", "entity"]``)
    """
    log.info(
        "retrieve_notes — query=%s, top_k=%s, use_graph=%s, graph_depth=%s, graph_weight=%s, use_entities=%s, entity_types=%s, keyword_weight=%s, min_similarity=%s, expand_query=%s, expand_entities=%s, use_summaries=%s, summary_threshold=%s, auto_weights=%s, auto_rewrite=%s",
        query, top_k, use_graph, graph_depth, graph_weight, use_entities, entity_types, keyword_weight, min_similarity, expand_query, expand_entities, use_summaries, summary_threshold, auto_weights, auto_rewrite,
    )
    try:
        result = pipelines.retrieve(
            query=query, top_k=top_k,
            use_graph=use_graph, graph_depth=graph_depth, graph_weight=graph_weight,
            use_entities=use_entities, entity_types=entity_types,
            keyword_weight=keyword_weight, min_similarity=min_similarity,
            expand_query=expand_query, expand_entities=expand_entities,
            use_summaries=use_summaries, summary_threshold=summary_threshold,
            auto_weights=auto_weights, auto_rewrite=auto_rewrite,
        )
        notes = result["notes"] if result else []
        log.info("retrieve_notes — %s notes returned", len(notes))
        return notes
    except Exception as e:
        log_error(log, "retrieve_notes FAILED", exc=e, query=query)
        return []


def tag_notes(query: str, top_k: int = 5, sync: bool = True) -> str:
    """Search notes matching a query and automatically suggest YAML frontmatter tags using the LLM. Use this when the user wants to tag or organise their notes — it finds relevant notes by semantic search, then asks an LLM to propose appropriate tags.

    Args:
        query: a semantic search query to find notes to tag.
        top_k: number of top-matching notes to process and suggest tags for (default 5).
        sync: if True (default), re-index each tagged note in ChromaDB so changes are reflected immediately in future searches.

    Returns:
        A string (typically JSON) showing the mapping from note paths to suggested tags.
    """
    log.info(f"tag_notes — query={query!r}, top_k={top_k}")
    try:
        result = pipelines.tag_notes(query, top_k=top_k)
        if sync:
            # Extract note paths from the result JSON and re-index each
            import json
            m = re.search(r"\{.*\}", result, re.DOTALL)
            if m:
                tag_map = json.loads(m.group())
                for path in tag_map:
                    indexer.index_note(path)
        log.info("tag_notes — done")
        return result
    except Exception as e:
        log_error(log, "tag_notes FAILED", exc=e, query=query)
        return f"Error: {e}"


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
    expand_entities: bool = False,
    use_summaries: bool = False,
    summary_threshold: float = 0.7,
    auto_weights: bool = False,
    auto_rewrite: bool = False,
) -> str:
    """Search all notes related to a topic and return an LLM-generated consolidated summary. Use this when the user wants a written overview, briefing, or synthesis of everything the vault contains about a subject — it retrieves the most relevant notes and asks the LLM to synthesise them into a coherent summary.

    Uses multi-strategy retrieval: semantic search, entity lookup, and wiki-link graph traversal. By default, both graph traversal and entity search are enabled for comprehensive coverage.

    Args:
        topic: the topic or subject to summarise.
        top_k: number of notes to retrieve as context for the LLM (default 5).
        use_graph: if True, expand results by following [[wiki-links]] to find connected notes (default True).
        graph_depth: max BFS hops for graph traversal when ``use_graph`` is True (default 1).
        graph_weight: weight for graph proximity boost between 0.0 and 1.0 (default 0.2).
        use_entities: if True, also search the entity index for notes mentioning entities related to the topic (default True).
        entity_types: optional list of entity types to filter by (e.g. ``["Person"]``, ``["Project"]``).
        keyword_weight: BM25 keyword blend between 0.0 and 1.0 (0.0 = pure semantic, 1.0 = pure keyword, default 0.0).
        expand_query: if True, use the LLM to generate alternative query phrasings for broader recall (default False).
        expand_entities: if True, when entities are auto-detected in the query, also search for related entities via the entity relationship graph (default False).
        use_summaries: if True, include note-level summary embeddings as an additional retrieval signal (default False).
        summary_threshold: minimum similarity score (0-1) for a summary result to be included (default 0.7).
        auto_weights: if True, detect query intent (entity-heavy, keyword-heavy, etc.) and adjust ranking weights dynamically (default False).
        auto_rewrite: if True, rewrite the query using known vault terminology before searching (default False).

    Returns:
        A string containing the LLM-generated summary of the topic, synthesised from the retrieved notes.
    """
    log.info(
        "summarize_topic — topic=%s, top_k=%s, use_graph=%s, graph_depth=%s, graph_weight=%s, use_entities=%s, entity_types=%s, keyword_weight=%s, expand_query=%s, expand_entities=%s, use_summaries=%s, summary_threshold=%s, auto_weights=%s, auto_rewrite=%s",
        topic, top_k, use_graph, graph_depth, graph_weight, use_entities, entity_types, keyword_weight, expand_query, expand_entities, use_summaries, summary_threshold, auto_weights, auto_rewrite,
    )
    try:
        result = pipelines.summarize_topic(
            topic=topic, top_k=top_k,
            use_graph=use_graph, graph_depth=graph_depth, graph_weight=graph_weight,
            use_entities=use_entities, entity_types=entity_types,
            keyword_weight=keyword_weight, expand_query=expand_query,
            expand_entities=expand_entities,
            use_summaries=use_summaries, summary_threshold=summary_threshold,
            auto_weights=auto_weights, auto_rewrite=auto_rewrite,
        )
        log.info(f"summarize_topic — done, {len(result)} chars")
        return result
    except Exception as e:
        log_error(log, "summarize_topic FAILED", exc=e, topic=topic)
        return f"Error: {e}"


def get_subject(subject: str, top_k: int = 10, keyword_weight: float = 0.3, group_by_note: bool = False) -> list[dict]:
    """Get notes related to a free-form subject using LLM-powered query expansion. Use this when the user asks about a broad subject, person, or concept — it expands the subject with related terms via the LLM, then performs a hybrid search (semantic + BM25 keyword).

    Unlike ``search_notes``, this tool automatically enriches the query with synonyms and related terms for a broader, more exploratory search.

    Args:
        subject: the free-form subject, person, or topic to search for (e.g. ``"machine learning"``, ``"productivity"``).
        top_k: maximum number of results to return (default 10).
        keyword_weight: BM25 keyword blend between 0.0 and 1.0 (0.0 = pure semantic, 1.0 = pure keyword, default 0.3 — slightly favours keywords by default).
        group_by_note: if True, collapse chunk-level results into one result per note with a ``chunk_count`` field (default False).

    Returns:
        A list of dicts. Each dict contains:
        - ``path`` — vault-relative note path
        - ``title`` — note title (filename without extension)
        - ``similarity_score`` — 0-to-1 relevance score
        - ``snippet`` — excerpt of the matching passage
        - ``chunk_count`` — (only when ``group_by_note=True``) how many chunks matched in this note
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


__all_tools__ = [
    get_index_stats,
    find_duplicate_notes,
    search_notes,
    batch_search,
    composite_search,
    ask_vault,
    retrieve_notes,
    ask_agent,
    tag_notes,
    summarize_topic,
    get_subject,
]
