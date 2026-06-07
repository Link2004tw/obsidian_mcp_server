"""Consolidated entities tool — search, manage, and explore named entities."""

import json
import os

from .. import chroma_store, entity_relations, entity_resolver, entity_store, graph_store, indexer
from ..logger import get_logger, log_error
from ._shared import _find_notes_mentioning, _normalize_path
from ._tool_base import build_tool

log = get_logger("obsidian_ai.tools.entities")

_VALID_ACTIONS = {"search", "note_entities", "list", "aliases", "timeline", "related", "add", "link_note", "merge", "change_type", "types", "weights_set", "weights_get", "import"}


def _handle_search(entity_name: str, entity_type: str | None = None, n: int = 10, use_graph: bool = False) -> str:
    from ._shared import _truncate_snippet

    results = entity_store.search(entity_name, type=entity_type, n=n * 2)

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

    results = [r for r in results if not r.get("path", "").startswith("Subjects/")]
    results = results[:n]

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

    results = [r for r in results if not r.get("path", "").startswith("Subjects/")]
    return json.dumps(results, ensure_ascii=False, indent=2) if results else f"No notes found mentioning \"{entity_name}\"."


def _handle_note_entities(path: str) -> str:
    path = _normalize_path(path)
    results = entity_store.get_note_entities(path)
    return json.dumps(results, ensure_ascii=False, indent=2)


def _handle_list(entity_type: str | None = None, n: int = 50) -> str:
    results = entity_store.list_entities(entity_type=entity_type, n=n)
    return json.dumps(results, ensure_ascii=False, indent=2)


def _handle_aliases(name: str) -> str:
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


def _handle_timeline(name: str, date_from: str | None = None, date_to: str | None = None) -> str:
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


def _handle_related(name: str, relation_type: str | None = None, depth: int = 1) -> str:
    results = entity_relations.get_related(name, relation_type=relation_type, depth=depth)
    if not results:
        return f"No related entities found for \"{name}\""
    lines = [f"Entities related to \"{name}\" (depth={depth}):"]
    for r in results:
        kind = r.get("relation_type", "?")
        conf = r.get("confidence", 0.0)
        d = r.get("depth", 1)
        lines.append(f"  \u2022 {r['entity_name']}  [{kind}]  conf={conf:.2f}  depth={d}")
    return "\n".join(lines)


def _handle_add(name: str, entity_type: str = "Concept", aliases: list[str] | None = None, relations: list[dict] | None = None, reindex_matches: bool = True) -> str:
    result = entity_store.add_manual_entity(name, entity_type, aliases=aliases)
    if relations:
        count = 0
        for rel in relations:
            target = rel.get("target", "")
            if not target:
                continue
            rel_type = rel.get("type", "related_to")
            confidence = rel.get("confidence", 0.5)
            entity_relations.add(source=name, type=rel_type, target=target, confidence=confidence)
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


def _handle_link_note(name: str, path: str, entity_type: str | None = None) -> str:
    path = _normalize_path(path)
    etype = entity_type or "Concept"
    entity_store.add_manual_entity(name, etype)
    entity_store.add(name=name, type=etype, confidence=1.0, path=path, chunk_idx=0, context="(manually linked via link_note)")
    if not graph_store.has_entity_edge(etype, name, path):
        graph_store.add_entity_edge(etype, name, path)
        graph_store.flush()
    indexer.index_note(path, force=True)
    return f'Note linked: "{path}" → entity "{name}" ({etype})'


def _handle_merge(primary: str, secondary: str) -> str:
    result = entity_store.merge(primary, secondary)
    if result is None:
        return f"Could not merge — one or both entities not found: primary=\"{primary}\", secondary=\"{secondary}\""
    alias_str = ", ".join(result["aliases"]) if result["aliases"] else "(none)"
    lines = [
        f"Merged \"{secondary}\" \u2192 \"{primary}\"",
        f"Canonical: {result['canonical']}",
        f"Type:      {result['type']}",
        f"Aliases:   {alias_str}",
        f"Mentions:  {result['mention_count']}",
    ]
    return "\n".join(lines)


def _handle_change_type(name: str, new_type: str) -> str:
    result = entity_store.change_entity_type(name, new_type)
    if result is None:
        return f"Entity '{name}' not found."
    return (
        f"Entity type changed:\n"
        f"  Name:       \"{result['entity_name']}\"\n"
        f"  New type:   {result['entity_type']}\n"
        f"  Mentions:   {result['mention_count']}"
    )


def _handle_types() -> str:
    types = entity_store.entity_types()
    return json.dumps(types, ensure_ascii=False, indent=2)


def _handle_weights_get() -> str:
    from .. import ranker
    w = ranker.weights()
    max_k = max(len(k) for k in w)
    lines = [f"{k.rjust(max_k)}: {v:.2f}" for k, v in sorted(w.items())]
    return "\n".join(lines)


def _handle_weights_set(semantic: float | None = None, entity: float | None = None, graph: float | None = None, keyword: float | None = None) -> str:
    from .. import ranker
    w = ranker.set_weights(semantic=semantic, entity=entity, graph=graph, keyword=keyword)
    max_k = max(len(k) for k in w)
    lines = [f"{k.rjust(max_k)}: {v:.2f}" for k, v in sorted(w.items())]
    return "Updated ranking weights:\n" + "\n".join(lines)


def _handle_import(data: str, dedup_config: str | None = None) -> str:
    import json as _json
    parsed = _json.loads(data)
    config_parsed = _json.loads(dedup_config) if dedup_config else None
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


_HANDLERS = {
    "search": _handle_search,
    "note_entities": _handle_note_entities,
    "list": _handle_list,
    "aliases": _handle_aliases,
    "timeline": _handle_timeline,
    "related": _handle_related,
    "add": _handle_add,
    "link_note": _handle_link_note,
    "merge": _handle_merge,
    "change_type": _handle_change_type,
    "types": _handle_types,
    "weights_get": _handle_weights_get,
    "weights_set": _handle_weights_set,
    "import": _handle_import,
}


@build_tool("entities")
def entities(
    action: str,
    name: str = "",
    entity_name: str = "",
    entity_type: str | None = None,
    path: str = "",
    n: int = 50,
    aliases: list[str] | None = None,
    relations: list[dict] | None = None,
    reindex_matches: bool = True,
    primary: str = "",
    secondary: str = "",
    new_type: str = "",
    date_from: str | None = None,
    date_to: str | None = None,
    relation_type: str | None = None,
    depth: int = 1,
    use_graph: bool = False,
    data: str = "",
    dedup_config: str | None = None,
    semantic: float | None = None,
    entity: float | None = None,
    graph: float | None = None,
    keyword: float | None = None,
) -> str:
    """Search, create, and manage named entities (people, projects, concepts, etc.) in the vault.

    Args:
        action: ``search`` — find notes mentioning a specific entity.
                ``note_entities`` — list entities extracted from a specific note.
                ``list`` — list all entities sorted by mention count.
                ``aliases`` — show canonical name, type, aliases for an entity.
                ``timeline`` — show chronological timeline of events for an entity.
                ``related`` — discover entities connected via the relationship graph.
                ``add`` — manually create a new entity with aliases and relations.
                ``link_note`` — associate an existing note with an entity (creates entity if needed).
                ``merge`` — merge two entity records into one.
                ``change_type`` — correct the classification type of an entity.
                ``types`` — list all available entity type labels.
                ``weights_get`` — show current ranking weights for search signals.
                ``weights_set`` — adjust ranking weights (semantic, entity, graph, keyword).
                ``import`` — import entities and relations from another vault.
        name: entity name (for aliases, timeline, related, change_type).
        entity_name: entity name to search for (for search action).
        entity_type: optional type filter (Person, Project, etc.).
        path: vault-relative note path (for note_entities, link_note).
        n: max results (default 50).
        aliases: alias strings (for add).
        relations: relationship dicts (for add).
        reindex_matches: re-index notes mentioning new entity (default True).
        primary: entity name to keep (for merge).
        secondary: entity name to merge in (for merge).
        new_type: target entity type (for change_type).
        date_from: lower date bound for timeline.
        date_to: upper date bound for timeline.
        relation_type: filter for related entities.
        depth: traversal depth for related entities (default 1).
        use_graph: expand via wiki-links for search (default False).
        data: JSON string for import.
        dedup_config: optional JSON dedup config for import.
        semantic/entity/graph/keyword: weight values (0.0-1.0) for weights_set.

    Returns:
        JSON data or formatted text.
    """
    if action not in _VALID_ACTIONS:
        valid = ", ".join(sorted(_VALID_ACTIONS))
        return f"Error: Invalid action '{action}'. Valid actions: {valid}"

    handler = _HANDLERS[action]
    try:
        if action == "search":
            return handler(entity_name=entity_name or name, entity_type=entity_type, n=n, use_graph=use_graph)
        elif action == "note_entities":
            return handler(path=path)
        elif action == "list":
            return handler(entity_type=entity_type, n=n)
        elif action == "aliases":
            return handler(name=name)
        elif action == "timeline":
            return handler(name=name, date_from=date_from, date_to=date_to)
        elif action == "related":
            return handler(name=name, relation_type=relation_type, depth=depth)
        elif action == "add":
            return handler(name=name, entity_type=entity_type or "Concept", aliases=aliases, relations=relations, reindex_matches=reindex_matches)
        elif action == "link_note":
            return handler(name=name, path=path, entity_type=entity_type)
        elif action == "merge":
            return handler(primary=primary, secondary=secondary)
        elif action == "change_type":
            return handler(name=name, new_type=new_type)
        elif action == "types":
            return handler()
        elif action == "weights_get":
            return handler()
        elif action == "weights_set":
            return handler(semantic=semantic, entity=entity, graph=graph, keyword=keyword)
        elif action == "import":
            return handler(data=data, dedup_config=dedup_config)
        else:
            return f"Error: Unhandled action '{action}'"
    except Exception as e:
        log_error(log, f"entities — action={action} FAILED", exc=e)
        return f"Error: {e}"


__all_tools__ = [entities]
