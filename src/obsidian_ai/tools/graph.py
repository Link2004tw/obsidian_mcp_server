"""Entity lookup, graph traversal, and community detection tools."""

import contextlib
import os

from .. import (
    chroma_store,
    entity_relations,
    entity_store,
    graph_store,
    obsidian_client,
    ranker,
)
from ..frontmatter import parse as fm_parse
from ..logger import get_logger, log_error
from ._shared import (
    _hybrid_search,
    _normalize_path,
    _truncate_snippet,
)

log = get_logger("obsidian_ai.tools.graph")


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


def get_note_entities(path: str) -> list[dict]:
    """Return all entities found in a specific note during indexing.

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.

    Returns:
        List of dicts with entity_name, entity_type, confidence.
    """
    path = _normalize_path(path)
    log.info(f"get_note_entities — {path}")
    try:
        results = entity_store.get_note_entities(path)
        log.info(f"get_note_entities — {len(results)} entities")
        return results
    except Exception as e:
        log_error(log, "get_note_entities FAILED", exc=e, path=path)
        return []


def get_entity_types() -> list[str]:
    """Return all available entity types used for classification.

    Returns:
        List of entity type strings.
    """
    return entity_store.entity_types()


def get_entity_aliases(name: str) -> str:
    """Return all alias information for a given entity.

    Shows canonical name, entity type, alias list (both LLM-generated
    and manual), and total mention count across the vault.

    Args:
        name: entity name — canonical or alias (e.g. ``"Maria"`` or ``"Her"``).

    Returns:
        A formatted string with canonical name, type, aliases, and mention count,
        or an error message if the entity is not found.
    """
    log.info(f"get_entity_aliases — {name}")
    try:
        result = entity_store.get_aliases(name)
        if result is None:
            return f"Entity not found: {name}"
        aliases = result.get("aliases", [])
        alias_str = ", ".join(aliases) if aliases else "(none)"
        lines = [
            f"Canonical: {result['canonical']}",
            f"Type:      {result['type']}",
            f"Aliases:   {alias_str}",
            f"Mentions:  {result['mention_count']}",
        ]
        return "\n".join(lines)
    except Exception as e:
        log_error(log, "get_entity_aliases FAILED", exc=e, name=name)
        return f"Error: {e}"


def merge_entities(primary: str, secondary: str) -> str:
    """Merge two entity records, keeping *primary* as the canonical name.

    Combines mention lists, merges aliases, and removes the secondary
    entity record. Useful for cleaning up duplicate entities or merging
    variants that weren't auto-detected as aliases.

    Args:
        primary: the entity name to keep (canonical).
        secondary: the entity name to merge into primary and then delete.

    Returns:
        Confirmation message with merged entity details, or an error.
    """
    log.info(f"merge_entities — primary={primary}, secondary={secondary}")
    try:
        result = entity_store.merge(primary, secondary)
        if result is None:
            return f"Could not merge — one or both entities not found: primary=\"{primary}\", secondary=\"{secondary}\""
        alias_str = ", ".join(result["aliases"]) if result["aliases"] else "(none)"
        lines = [
            f"Merged \"{secondary}\" → \"{primary}\"",
            f"Canonical: {result['canonical']}",
            f"Type:      {result['type']}",
            f"Aliases:   {alias_str}",
            f"Mentions:  {result['mention_count']}",
        ]
        return "\n".join(lines)
    except Exception as e:
        log_error(log, "merge_entities FAILED", exc=e, primary=primary, secondary=secondary)
        return f"Error: {e}"


def entity_timeline(name: str, date_from: str | None = None,
                    date_to: str | None = None) -> str:
    """Return a chronological timeline of events for an entity.

    Timeline entries are extracted by the LLM during indexing and include
    dates, event descriptions, and note references.

    Args:
        name: entity name (canonical or alias), e.g. ``"Alice"``.
        date_from: optional lower date bound (inclusive), e.g. ``"2024-01"``.
        date_to: optional upper date bound (inclusive), e.g. ``"2024-12"``.

    Returns:
        A formatted timeline or an error message if no entries are found.
    """
    log.info(f"entity_timeline — name={name}, from={date_from}, to={date_to}")
    try:
        entries = entity_store.get_timeline(name, date_from=date_from, date_to=date_to)
        if entries is None:
            return f"Entity not found: {name}"
        if not entries:
            return f"No timeline entries found for \"{name}\"."
        lines = [f"Timeline for {name} ({len(entries)} entries):"]
        for entry in entries:
            note = entry.get("note", "")
            title = os.path.splitext(os.path.basename(note))[0] if note else "?"
            lines.append(f"  {entry['date']}  {entry['event']}  ({title})")
        return "\n".join(lines)
    except Exception as e:
        log_error(log, "entity_timeline FAILED", exc=e, name=name)
        return f"Error: {e}"


def related_entities(
    name: str,
    relation_type: str | None = None,
    depth: int = 1,
) -> str:
    """Find entities related to a given entity via the relationship graph.

    Traverses the entity-relationship graph between entity records in
    the knowledge index. Relationships are extracted during note indexing
    and include types like ``works_on``, ``uses``, ``part_of``,
    ``located_in``, ``attends``, ``created_by``, and ``related_to``.

    Args:
        name: entity name (e.g. ``"Alice Johnson"``, ``"ESP32"``).
        relation_type: optional filter — only return relationships of
            this type. Supported types: ``works_on``, ``uses``,
            ``part_of``, ``related_to``, ``created_by``, ``located_in``,
            ``attends``. Pass ``null`` or omit for all types.
        depth: maximum traversal depth (default 1, meaning direct
            relationships only). Depth 2 follows relationships one hop
            further. Use depth 3+ for broader exploration.

    Returns:
        A formatted table of related entities with relationship type,
        confidence, and hop depth, or a message if none are found.
    """
    log.info(f"related_entities — name={name}, type={relation_type}, depth={depth}")
    try:
        results = entity_relations.get_related(name, relation_type=relation_type, depth=depth)
        if not results:
            return f"No related entities found for \"{name}\""
        lines = [f"Entities related to \"{name}\" (depth={depth}):"]
        for r in results:
            kind = r.get("relation_type", "?")
            conf = r.get("confidence", 0.0)
            d = r.get("depth", 1)
            lines.append(f"  • {r['entity_name']}  [{kind}]  conf={conf:.2f}  depth={d}")
        return "\n".join(lines)
    except Exception as e:
        log_error(log, "related_entities FAILED", exc=e, name=name)
        return f"Error: {e}"


def get_shortest_path(start: str, end: str) -> str:
    """Find the shortest path between two notes in the wiki-link graph.

    Uses BFS (breadth-first search) on the unweighted wiki-link graph
    to find the shortest path. The graph is built from ``[[wiki-links]]``
    during indexing.

    Args:
        start: vault-relative path of the starting note
            (e.g. ``"Projects/MyProject.md"``).
        end: vault-relative path of the target note
            (e.g. ``"Hardware/ESP32.md"``).

    Returns:
        A formatted path showing each hop from start to end, or an
        error message if no path exists.
    """
    log.info(f"get_shortest_path — start={start}, end={end}")
    try:
        path = graph_store.shortest_path(start, end)
        if path is None:
            return f"No path found between \"{start}\" and \"{end}\""
        lines = [
            f"Shortest path ({len(path) - 1} hops):",
        ]
        for i, p in enumerate(path):
            title = os.path.splitext(os.path.basename(p))[0]
            marker = "\u2192" if i > 0 else " "
            lines.append(f"  {marker} {p}  ({title})")
        return "\n".join(lines)
    except Exception as e:
        log_error(log, "get_shortest_path FAILED", exc=e, start=start, end=end)
        return f"Error: {e}"


def get_backlinks(path: str) -> list[dict]:
    """Return all notes linking TO the given note (incoming wiki-link edges).

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.

    Returns:
        List of dicts with path, title, and trace (the path from source to target).
    """
    path = _normalize_path(path)
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


def get_linked_notes(path: str) -> list[dict]:
    """Return all notes the given note links TO (outgoing wiki-link edges).

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.

    Returns:
        List of dicts with path and title.
    """
    path = _normalize_path(path)
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
            with contextlib.suppress(Exception):
                all_contents[path] = obsidian_client.get_note(path)
        broken = graph_store.get_broken_links(all_contents)
        log.info(f"get_broken_links — {len(broken)} broken links")
        return broken
    except Exception as e:
        log_error(log, "get_broken_links FAILED", exc=e)
        return []


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


def multi_hop_traversal(path: str, max_depth: int = 2) -> list[dict]:
    """Perform BFS graph traversal from a seed note up to N hops.

    Returns all reachable notes with path traces for explainability
    (e.g., A -> B -> C shows the chain of wiki-links).

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.
        max_depth: maximum number of hops to traverse (default 2).

    Returns:
        List of dicts with path, title, depth (hop count), and trace (list of paths from seed).
    """
    path = _normalize_path(path)
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


def related_notes(path: str, k: int = 10, graph_weight: float = 0.3) -> list[dict]:
    """Find notes related to a given note using both semantic similarity and graph proximity.

    Combines embedding-based semantic search with wiki-link graph traversal.
    Notes connected via wiki-links get a proximity boost proportional to graph_weight.

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.
        k: number of results to return.
        graph_weight: how much to weight graph proximity (0.0 = pure semantic, 1.0 = pure graph).

    Returns:
        List of dicts with path, title, similarity_score, and is_graph_connected (bool).
    """
    path = _normalize_path(path)
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
        for _rank, result in enumerate(semantic_results):
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


def get_note_community(path: str, top_k: int = 5) -> str:
    """Return which graph community a note belongs to and top-K other members.

    Communities are detected via label propagation on the wiki-link graph.
    Notes in the same community are densely connected via bidirectional
    wiki-link traversal.

    Args:
        path: vault-relative path of the note, e.g. ``"Projects/MyProject.md"``.
        top_k: max number of other community members to list (default 5).

    Returns:
        A formatted string with community ID and member list, or an error
        message if the note is not in any community.
    """
    log.info(f"get_note_community — path={path}, top_k={top_k}")
    try:
        info = graph_store.get_community_info(path, top_k=top_k)
        if info is None:
            title = os.path.splitext(os.path.basename(path))[0]
            return f"\"{path}\" ({title}) does not belong to any community."
        members = info["members"]
        lines = [
            f"Community {info['community_id']} — {len(members) + 1} members",
            f"  Self:  {path}",
        ]
        if members:
            lines.append(f"  Top-{len(members)} other members:")
            for m in members:
                t = os.path.splitext(os.path.basename(m))[0]
                lines.append(f"    \u2022 {m}  ({t})")
        return "\n".join(lines)
    except Exception as e:
        log_error(log, "get_note_community FAILED", exc=e, path=path)
        return f"Error: {e}"


def get_ranking_weights() -> str:
    """Return the current ranking weights used by the unified Ranker.

    The Ranker blends four signals: semantic, entity, graph, and keyword.
    Weights are normalised to sum to 1.0.

    Returns:
        A formatted string showing each signal's weight.
    """
    log.info("get_ranking_weights")
    w = ranker.weights()
    max_k = max(len(k) for k in w)
    lines = [f"{k.rjust(max_k)}: {v:.2f}" for k, v in sorted(w.items())]
    return "\n".join(lines)


def set_ranking_weights(
    semantic: float | None = None,
    entity: float | None = None,
    graph: float | None = None,
    keyword: float | None = None,
) -> str:
    """Set ranking weights for the unified Ranker at runtime.

    The Ranker blends four signals: semantic, entity, graph, and keyword.
    Only the weights you specify are updated; others keep their current value.
    Weights are automatically normalised to sum to 1.0 internally.

    Args:
        semantic: weight for semantic (embedding) similarity (0.0-1.0).
        entity: weight for entity-index matching (0.0-1.0).
        graph: weight for wiki-link graph proximity (0.0-1.0).
        keyword: weight for BM25 keyword search (0.0-1.0).

    Returns:
        A formatted string showing the updated weights.
    """
    log.info("set_ranking_weights — semantic=%s, entity=%s, graph=%s, keyword=%s",
             semantic, entity, graph, keyword)
    w = ranker.set_weights(semantic=semantic, entity=entity, graph=graph, keyword=keyword)
    max_k = max(len(k) for k in w)
    lines = [f"{k.rjust(max_k)}: {v:.2f}" for k, v in sorted(w.items())]
    return "Updated ranking weights:\n" + "\n".join(lines)


__all_tools__ = [
    "search_entities",
    "get_note_entities",
    "get_entity_types",
    "get_entity_aliases",
    "merge_entities",
    "entity_timeline",
    "related_entities",
    "get_shortest_path",
    "get_backlinks",
    "get_linked_notes",
    "get_broken_links",
    "get_graph_stats",
    "multi_hop_traversal",
    "related_notes",
    "export_graph",
    "get_orphan_notes",
    "get_communities",
    "get_note_community",
    "get_ranking_weights",
    "set_ranking_weights",
]
