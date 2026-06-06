"""Entity lookup, graph traversal, and community detection tools."""

import contextlib
import os

from .. import (
    chroma_store,
    entity_relations,
    entity_resolver,
    entity_store,
    graph_store,
    indexer,
    obsidian_client,
    ranker,
)
from ..frontmatter import parse as fm_parse
from ..logger import get_logger, log_error
from ._shared import (
    _find_notes_mentioning,
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
    """Find notes that mention a specific entity in the vault. Use this when the user asks about a person, project, technology, concept, or any named entity — it performs entity-aware lookup rather than general semantic search.

    Searches the entity index first, then falls back to ChromaDB metadata matching. Optionally expands results by following wiki-links from matching notes.

    Args:
        entity_name: name of the entity to search for (e.g. ``"ESP32"``, ``"Alice"``).
        entity_type: optional filter (e.g. ``"Person"``, ``"Hardware"``). One of:
                     Person, Project, Hardware, Technology, Location, Concept, Event. Default: ``None`` (all types).
        n: maximum number of results to return. Default: 10.
        use_graph: if True, also traverse wiki-links from matching notes to
                   find connected notes that may mention the entity indirectly. Default: False.

    Returns:
        List of dicts, each with keys: path, title, entity_name, entity_type, snippet, confidence.
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
    """List all entities extracted from a specific note during indexing. Use this when the user asks what entities (people, projects, concepts, etc.) were found in a given note.

    Args:
        path: vault-relative path to the note (e.g. ``"Folder/Note.md"`` — not a full filesystem path).

    Returns:
        List of dicts, each with keys: entity_name, entity_type, confidence.
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
    """List all entity type labels used for entity classification in the vault. Use this when the user wants to know what categories (Person, Project, Hardware, etc.) are available for filtering entity searches.

    Returns:
        List of entity type strings (e.g. ``["Person", "Project", "Hardware", "Technology", "Location", "Concept", "Event"]``).
    """
    return entity_store.entity_types()


def get_entity_aliases(name: str) -> str:
    """Show the canonical name, type, aliases, and mention count for an entity. Use this when the user asks about alternative names for a person, project, or concept, or when you need to resolve an ambiguous entity reference.

    Args:
        name: entity name — canonical or alias (e.g. ``"Maria"`` or ``"Her"``).

    Returns:
        A formatted human-readable string showing the canonical name, entity type, list of aliases (both LLM-generated and manually added), and total mention count across the vault. Returns an error message if the entity is not found.
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
    """Merge two entity records into one, keeping *primary* as the canonical name and deleting the secondary. Use this when the user identifies duplicate entities (e.g., the same person indexed under slightly different names) that need to be consolidated.

    Combines mention lists, merges aliases, and removes the secondary entity record.

    Args:
        primary: the entity name to keep as the canonical name.
        secondary: the entity name to merge into primary and then delete.

    Returns:
        A formatted confirmation string with merged entity details (canonical name, type, aliases, mention count), or an error message if one or both entities are not found.
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


def import_entities(data: str, dedup_config: str | None = None) -> str:
    """Import entities and relations from another Obsidian vault and merge them into the current index. Use this when the user wants to transfer entity data between vaults.

    Accepts a JSON string of entities (and optionally relations) exported from another vault. Entities are matched and merged using configurable dedup strategies: exact name match, alias overlap, and fuzzy similarity.

    Args:
        data: JSON string with ``entities`` dict and optional ``relations`` list.
            Format::
                {
                  "entities": {
                    "normalized_key": {
                      "canonical": "EntityName",
                      "type": "Person",
                      "aliases": ["alt_name"],
                      "mentions": [{"path": "...", "chunk_idx": 0, "context": "...", "confidence": 0.95}]
                    }
                  },
                  "relations": [
                    {"source": "EntityName", "type": "works_on", "target": "ProjectX", "confidence": 0.9}
                  ]
                }
        dedup_config: optional JSON string overriding matching thresholds.
            Format::
                {"exact_match": true, "alias_match": true, "fuzzy_threshold": 0.85, "strategy": "auto"}
            Default: ``None`` (uses default thresholds).

    Returns:
        A formatted summary string showing the total incoming entities, how many were merged, added, skipped, and how many relations were added.
    """
    log.info("import_entities")
    try:
        import json
        parsed = json.loads(data)
        config_parsed = json.loads(dedup_config) if dedup_config else None
        resolver = entity_resolver.EntityResolver(dedup_config=config_parsed)
        summary = resolver.resolve(parsed)
        lines = [
            f"Entity import complete — {summary['total_incoming']} incoming entities",
            f"  Merged:   {summary['merged']}",
            f"  Added:    {summary['added']}",
            f"  Skipped:  {summary['skipped']}",
            f"  Relations added: {summary['relations_added']}",
        ]
        return "\n".join(lines)
    except Exception as e:
        log_error(log, "import_entities FAILED", exc=e)
        return f"Error: {e}"


def entity_timeline(name: str, date_from: str | None = None,
                    date_to: str | None = None) -> str:
    """Show a chronological timeline of events linked to an entity. Use this when the user asks "what happened with X" or wants a chronological summary of events involving a person, project, or concept.

    Timeline entries are extracted by the LLM during indexing and include dates, event descriptions, and note references.

    Args:
        name: entity name — canonical or alias (e.g. ``"Alice"``).
        date_from: optional lower date bound (inclusive), e.g. ``"2024-01"``. Default: None (no lower bound).
        date_to: optional upper date bound (inclusive), e.g. ``"2024-12"``. Default: None (no upper bound).

    Returns:
        A formatted human-readable string listing the timeline entries chronologically, each with date, event description, and source note title. Returns an error message if the entity is not found or has no timeline entries.
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
    """Discover entities connected to a given entity via the relationship graph. Use this when the user asks "what is X related to", "who works on X", or wants to explore connections between people, projects, and concepts.

    Traverses the entity-relationship graph in the knowledge index. Relationships are extracted during note indexing and include types like ``works_on``, ``uses``, ``part_of``, ``located_in``, ``attends``, ``created_by``, and ``related_to``.

    Args:
        name: entity name (e.g. ``"Alice Johnson"``, ``"ESP32"``).
        relation_type: optional filter — only return relationships of
            this type. Supported types: ``works_on``, ``uses``,
            ``part_of``, ``related_to``, ``created_by``, ``located_in``,
            ``attends``. Pass ``None`` or omit for all types. Default: None.
        depth: maximum traversal depth. Depth 1 = direct relationships only.
            Depth 2 follows relationships one hop further. Use depth 3+ for
            broader exploration. Default: 1.

    Returns:
        A formatted human-readable string listing each related entity with its relationship type, confidence score, and hop depth. Returns a message if no related entities are found.
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
    """Find the shortest chain of wiki-link connections between two notes in the vault graph. Use this when the user asks how two notes are connected, or "how do I get from X to Y" in the wiki-link network.

    Uses BFS (breadth-first search) on the unweighted wiki-link graph built from ``[[wiki-links]]`` during indexing.

    Args:
        start: vault-relative path of the starting note
            (e.g. ``"Projects/MyProject.md"``).
        end: vault-relative path of the target note
            (e.g. ``"Hardware/ESP32.md"``).

    Returns:
        A formatted human-readable string showing each hop from start to end with note titles, or an error message if no path exists.
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
    """List all notes in the vault that link TO the given note via wiki-links. Use this when the user asks "what notes reference this note" or "which pages link to X".

    Args:
        path: vault-relative path to the note (e.g. ``"Folder/Note.md"`` — not a full filesystem path).

    Returns:
        List of dicts, each with keys: path (source note path), title (source note title).
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
    """List all notes that the given note links TO via outgoing wiki-links. Use this when the user asks "what does this note link to" or wants to explore outward connections from a specific note.

    Args:
        path: vault-relative path to the note (e.g. ``"Folder/Note.md"`` — not a full filesystem path).

    Returns:
        List of dicts, each with keys: path (target note path), title (target note title).
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
    """Scan the entire vault for wiki-links that point to non-existent notes. Use this when the user wants to clean up dead links or verify vault integrity — these are links like ``[[DeletedNote]]`` where the target file no longer exists.

    Returns:
        List of dicts, each with keys: source_path (the note containing the broken link), link_target (the unresolved wiki-link target).
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
    """Return summary statistics about the vault's wiki-link graph structure. Use this when the user wants an overview of the vault's connectivity — total notes, link count, isolated notes, and the most-linked hub notes.

    Returns:
        Dict with keys: nodes (int), edges (int), avg_degree (float), isolated_count (int), isolated (list of note paths), hubs (list of top 5 most-linked note paths with degree).
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
    """Traverse the wiki-link graph outward from a seed note up to N hops, returning all reachable notes. Use this when the user wants to explore the broader network around a note, or see indirectly connected notes.

    Uses BFS (breadth-first search) on the unweighted wiki-link graph. Each result includes a trace showing the full chain of wiki-links from the seed note (e.g., ``["Seed.md", "Intermediary.md", "Target.md"]``).

    Args:
        path: vault-relative path to the starting note (e.g. ``"Folder/Note.md"`` — not a full filesystem path).
        max_depth: maximum number of wiki-link hops to traverse. Default: 2.

    Returns:
        List of dicts, each with keys: path (reachable note), title, depth (hop count from seed), trace (list of paths forming the chain from seed to this note).
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
    """Find notes that are semantically similar or wiki-link-connected to a given note. Use this when the user asks "what notes are related to X", "find similar notes", or wants recommendations based on both content similarity and graph proximity.

    Combines embedding-based semantic search with wiki-link graph traversal. Notes connected via wiki-links get a proximity boost proportional to graph_weight.

    Args:
        path: vault-relative path to the source note (e.g. ``"Folder/Note.md"`` — not a full filesystem path).
        k: number of results to return. Default: 10.
        graph_weight: how much to weight graph proximity over semantic similarity. 0.0 = pure semantic (embeddings only), 1.0 = pure graph (wiki-links only). Default: 0.3.

    Returns:
        List of dicts, each with keys: path, title, similarity_score (float, 0-1), is_graph_connected (bool — whether the note shares a direct wiki-link with the source).
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
    """Export the entire wiki-link graph as a string for external visualization tools. Use this when the user wants to visualize the vault structure in Graphviz, Gephi, or other graph analysis tools.

    Args:
        format: output format. ``"json"`` for a JSON object with nodes and edges, ``"dot"`` for Graphviz DOT format. Default: ``"json"``.

    Returns:
        The full graph representation as a string — either a JSON string (with ``nodes`` and ``edges`` lists) or a DOT format string.
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
    """Find all notes that have no incoming or outgoing wiki-links, making them disconnected from the vault graph. Use this when the user wants to clean up or re-integrate isolated notes — these orphans are not reachable from any other note via wiki-links.

    Returns:
        List of vault-relative note paths for orphan notes (empty list if none found).
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
    """Detect densely connected groups (communities) of notes in the wiki-link graph using label propagation. Use this when the user wants to see how the vault clusters into topical groups or to understand the overall structure of connections.

    Notes in the same community are more densely connected via wiki-links to each other than to notes outside the community.

    Returns:
        Dict mapping community IDs (as strings) to sorted lists of vault-relative note paths in that community.
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
    """Identify which wiki-link community a note belongs to and list its closest neighbours in that community. Use this when the user wants to know the topical cluster a note is part of, or what other notes are closely connected to it via wiki-links.

    Communities are detected via label propagation on the wiki-link graph. Notes in the same community are densely connected via bidirectional wiki-link traversal.

    Args:
        path: vault-relative path of the note (e.g. ``"Projects/MyProject.md"``).
        top_k: maximum number of other community members to list. Default: 5.

    Returns:
        A formatted human-readable string showing the community ID, total member count, and the top-K other members with their titles. Returns a message if the note is not in any community.
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


def list_entities(entity_type: str | None = None, n: int = 50) -> list[dict]:
    """List all entities in the knowledge index, sorted by mention count descending. Use this when the user wants an overview of all known entities, or to browse entities of a specific type.

    Args:
        entity_type: optional filter — one of Person, Project, Hardware,
            Technology, Location, Concept, Event. Pass ``None`` for all types. Default: None.
        n: maximum number of results to return. Default: 50.

    Returns:
        List of dicts, each with keys: entity_name, entity_type, mention_count.
    """
    log.info(f"list_entities — type={entity_type}, n={n}")
    try:
        results = entity_store.list_entities(entity_type=entity_type, n=n)
        log.info(f"list_entities — {len(results)} results")
        return results
    except Exception as e:
        log_error(log, "list_entities FAILED", exc=e, entity_type=entity_type, n=n)
        return []


def add_entity(
    name: str,
    entity_type: str = "Concept",
    aliases: list[str] | None = None,
    relations: list[dict] | None = None,
    reindex_matches: bool = True,
) -> str:
    """Manually create an entity in the knowledge index and relationship graph. Use this when the user wants to register a new person, project, concept, or other entity that wasn't auto-detected during indexing, or to add one explicitly with aliases and relationships.

    Entities are normally auto-extracted during indexing. This tool lets you create one manually so it appears in entity search results and can have relationships tracked.

    If ``relations`` are provided, each must be a dict with ``target`` (entity name) and optionally ``type`` (defaults to ``"related_to"``) and ``confidence`` (default 0.5).

    Args:
        name: canonical entity name (e.g. ``"ESP32"``, ``"Maria"``).
        entity_type: entity type — one of Person, Project, Hardware,
            Technology, Location, Concept, Event. Default: ``"Concept"``.
        aliases: optional list of alternative names for this entity.
        relations: optional list of relationship dicts, each with
            ``target`` (required), ``type`` (default ``"related_to"``),
            and ``confidence`` (default 0.5).
        reindex_matches: if True (default), scan the vault for notes that mention
            this entity name and force-re-index them so they pick up the new
            entity from the LLM.

    Returns:
        A formatted confirmation string showing the created entity's name, type, aliases, mention count, and number of relations created.
    """
    log.info(f"add_entity — name={name}, type={entity_type}, aliases={aliases}, relations={relations}")
    try:
        result = entity_store.add_manual_entity(name, entity_type, aliases=aliases)
        if relations:
            count = 0
            for rel in relations:
                target = rel.get("target", "")
                if not target:
                    continue
                rel_type = rel.get("type", "related_to")
                confidence = rel.get("confidence", 0.5)
                entity_relations.add(
                    source=name,
                    type=rel_type,
                    target=target,
                    confidence=confidence,
                )
                count += 1
            entity_relations.save()

        alias_str = ", ".join(result["aliases"]) if result["aliases"] else "(none)"
        lines = [
            f"Entity added: \"{result['entity_name']}\"",
            f"  Type:       {result['entity_type']}",
            f"  Aliases:    {alias_str}",
            f"  Mentions:   {result['mention_count']}",
        ]
        if relations:
            lines.append(f"  Relations:  {len(relations)} created")

        reindexed = 0
        if reindex_matches:
            for p in _find_notes_mentioning(name):
                if indexer.index_note(p, force=True):
                    reindexed += 1
        if reindexed:
            lines.append(f"  Re-indexed: {reindexed} existing notes mentioning this entity")

        return "\n".join(lines)
    except Exception as e:
        log_error(log, "add_entity FAILED", exc=e, name=name, entity_type=entity_type)
        return f"Error: {e}"


def add_aliases(name: str, new_aliases: list[str], reindex_matches: bool = True) -> str:
    """Register one or more alternative names for an existing entity. Use this when the user mentions that an entity is known by other names (e.g., nicknames, acronyms, translations) so that searching for any alias finds the same entity.

    For example, adding ``"Yahweh"`` as an alias of ``"God"`` means searching for ``"Yahweh"`` will find notes about God.

    Args:
        name: canonical entity name (e.g. ``"God"``, ``"ESP32"``).
        new_aliases: list of alias strings to add (e.g. ``["Yahweh", "Elohim"]``).
        reindex_matches: if True (default), scan the vault for notes that mention
            any of the new aliases and force-re-index them so they pick up this
            entity from the LLM.

    Returns:
        A formatted confirmation string showing the updated entity name, type, full alias list, and mention count.
    """
    log.info(f"add_aliases — name={name}, new_aliases={new_aliases}")
    try:
        result = entity_store.add_aliases(name, new_aliases)
        if result is None:
            return f"Entity '{name}' not found."

        alias_str = ", ".join(result["aliases"]) if result["aliases"] else "(none)"
        lines = [
            f"Aliases added to \"{result['entity_name']}\":",
            f"  Type:       {result['entity_type']}",
            f"  Aliases:    {alias_str}",
            f"  Mentions:   {result['mention_count']}",
        ]

        if reindex_matches:
            all_names = [name] + new_aliases
            seen: set[str] = set()
            reindexed = 0
            for n in all_names:
                for p in _find_notes_mentioning(n):
                    if p not in seen:
                        seen.add(p)
                        if indexer.index_note(p, force=True):
                            reindexed += 1
            if reindexed:
                lines.append(f"  Re-indexed: {reindexed} existing notes mentioning this entity or its aliases")

        return "\n".join(lines)
    except Exception as e:
        log_error(log, "add_aliases FAILED", exc=e, name=name, new_aliases=new_aliases)
        return f"Error: {e}"


def change_entity_type(name: str, new_type: str) -> str:
    """Correct the classification type of an existing entity. Use this when the user notices an entity was misclassified (e.g., a person was auto-detected as ``"Concept"``) and needs to be re-typed.

    For example, if ``"Maria"`` was auto-classified as ``Concept`` but should be ``Person``, use this tool.

    Args:
        name: entity name — canonical or alias (e.g. ``"Maria"``).
        new_type: one of Person, Project, Hardware, Technology, Location,
            Concept, Event.

    Returns:
        A formatted confirmation string showing the updated entity name, new type, and mention count.
    """
    log.info(f"change_entity_type — name={name}, new_type={new_type}")
    try:
        result = entity_store.change_entity_type(name, new_type)
        if result is None:
            return f"Entity '{name}' not found."
        return (
            f"Entity type changed:\n"
            f"  Name:       \"{result['entity_name']}\"\n"
            f"  New type:   {result['entity_type']}\n"
            f"  Mentions:   {result['mention_count']}"
        )
    except Exception as e:
        log_error(log, "change_entity_type FAILED", exc=e, name=name, new_type=new_type)
        return f"Error: {e}"


def get_ranking_weights() -> str:
    """Show the current weight settings for the unified search Ranker. Use this when the user wants to understand how search results are scored — the Ranker blends four signals and you may need to adjust them for better results.

    The Ranker blends four signals: semantic (embedding similarity), entity (entity-index matching), graph (wiki-link proximity), and keyword (BM25 text search). Weights are normalised to sum to 1.0.

    Returns:
        A formatted human-readable string showing the current weight for each signal (semantic, entity, graph, keyword) as values between 0.0 and 1.0.
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
    """Adjust the weight of each signal in the unified search Ranker to tune search result quality. Use this when the user says search results are too heavily biased toward one signal (e.g., too literal/keyword-heavy, or too semantic) and needs to rebalance them.

    The Ranker blends four signals: semantic (embedding similarity), entity (entity-index matching), graph (wiki-link proximity), and keyword (BM25 text search). Only the weights you specify are updated; unspecified weights keep their current value. Weights are automatically normalised to sum to 1.0 internally.

    Args:
        semantic: weight for semantic (embedding) similarity (0.0-1.0). Default: None (keep current).
        entity: weight for entity-index matching (0.0-1.0). Default: None (keep current).
        graph: weight for wiki-link graph proximity (0.0-1.0). Default: None (keep current).
        keyword: weight for BM25 keyword search (0.0-1.0). Default: None (keep current).

    Returns:
        A formatted human-readable string showing all four updated weights after normalisation.
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
    "list_entities",
    "add_entity",
    "add_aliases",
    "change_entity_type",
    "merge_entities",
    "import_entities",
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
