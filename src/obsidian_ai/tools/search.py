"""Search and retrieval tools for the Obsidian MCP server."""


from .. import (
    chroma_store,
    config,
    entity_store,
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
        - ``expand_entities`` — if True, when entities are auto-detected in the
          query, also search for related entities via the relationship graph
        - ``use_summaries`` — if True, include summary-embedding results as a
            retrieval signal.
        - ``summary_threshold`` — minimum similarity (0–1) for a summary
            result to be included (default 0.7).

    Returns:
        Passage-level results (default) or note-level results (when ``group_by_note=True``).
        Each dict has ``path``, ``title``, ``similarity_score``, ``snippet``,
        and when grouped: ``chunk_count`` (how many chunks matched this note).
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


def composite_search(
    query: str,
    n: int = 5,
    retrieval_depth: int = 2,
) -> list[dict]:
    """High-recall search combining summary embeddings, entity relationships,
    and community-aware graph traversal.

    Blends three retrieval strategies into a single ranked result set:

    * **Summary embeddings** — searches note-level semantic summaries.
    * **Entity expansion** — auto-detects entities in the query and follows
      relationship edges (works_on, part_of, etc.) to find related notes.
    * **Community graph** — finds other notes in the same wiki-link communities
      as top results, surfacing thematically related notes that may not share
      direct keywords or links.

    ``retrieval_depth`` controls which strategies are used:

    * ``1`` — summary embeddings only (fast, broad coverage)
    * ``2`` — summary + entity expansion (default — balanced recall)
    * ``3`` — full composite with community graph (maximum recall)

    Returns note-level results (not chunks), each with ``score`` and
    ``matched_by`` describing which strategies contributed.
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
        expand_entities: if True, when entities are auto-detected in the
            query, also search for related entities via the relationship graph.
        use_summaries: if True, include summary-embedding results as a
            retrieval signal.
        summary_threshold: minimum similarity (0–1) for a summary
            result to be included (default 0.7).
        auto_rewrite: if True, rewrite the query using known vault
            terminology before searching (default False).
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
        expand_entities: if True, when entities are auto-detected in the
            query, also search for related entities via the relationship graph.
        use_summaries: if True, include summary-embedding results as a
            retrieval signal.
        summary_threshold: minimum similarity (0–1) for a summary
            result to be included (default 0.7).
        auto_rewrite: if True, rewrite the query using known vault
            terminology before searching (default False).

    Returns:
        A list of dicts, each with:
        - ``path`` — vault-relative note path
        - ``title`` — note title (basename without extension)
        - ``content`` — full note content (truncated to context budget)
        - ``similarity_score`` — 0-to-1 blended score (higher = more relevant)
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
        expand_entities: if True, when entities are auto-detected in the
            query, also search for related entities via the relationship graph.
        use_summaries: if True, include summary-embedding results as a
            retrieval signal.
        summary_threshold: minimum similarity (0–1) for a summary
            result to be included (default 0.7).
        auto_weights: if True, detect query intent (entity, keyword, graph)
            and adjust ranking weights dynamically (default False).
        auto_rewrite: if True, rewrite the query using known vault
            terminology before searching (default False).

    Returns:
        An LLM-generated summary of the topic across related notes.
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
