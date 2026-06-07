"""Consolidated graph tool — explore vault connectivity, communities, and structure."""

import json
import os

from .. import chroma_store, graph_store, obsidian_client
from ..frontmatter import parse as fm_parse
from ..logger import get_logger, log_error
from ._shared import _hybrid_search, _normalize_path
from ._tool_base import build_tool

log = get_logger("obsidian_ai.tools.graph")

_VALID_ACTIONS = {"communities", "community_of", "orphans", "path", "stats", "related", "traverse", "export"}


def _handle_communities() -> str:
    communities = graph_store.label_propagation()
    if not communities:
        return "No communities detected."
    grouped: dict[str, list[str]] = {}
    for p, cid in communities.items():
        grouped.setdefault(str(cid), []).append(p)
    for k in grouped:
        grouped[k].sort()
    return json.dumps(grouped, ensure_ascii=False, indent=2)


def _handle_community_of(path: str, top_k: int = 5) -> str:
    path = _normalize_path(path)
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


def _handle_orphans() -> str:
    orphans = graph_store.get_orphans()
    return json.dumps(orphans, ensure_ascii=False, indent=2)


def _handle_path(start: str, end: str) -> str:
    start = _normalize_path(start)
    end = _normalize_path(end)
    result = graph_store.shortest_path(start, end)
    if result is None:
        return f"No path found between \"{start}\" and \"{end}\""
    lines = [f"Shortest path ({len(result) - 1} hops):"]
    for i, p in enumerate(result):
        title = os.path.splitext(os.path.basename(p))[0]
        marker = "\u2192" if i > 0 else " "
        lines.append(f"  {marker} {p}  ({title})")
    return "\n".join(lines)


def _handle_stats() -> str:
    stats = graph_store.stats()
    return json.dumps(stats, ensure_ascii=False, indent=2)


def _handle_related(path: str, k: int = 10, graph_weight: float = 0.3) -> str:
    path = _normalize_path(path)
    graph_neighbors = set(graph_store.get_outgoing(path))
    graph_neighbors.update(graph_store.get_backlinks(path))
    graph_neighbors.discard(path)

    source_content = obsidian_client.get_note(path)
    meta, body = fm_parse(source_content)
    query_text = body[:500]

    semantic_results = _hybrid_search(
        queries=[query_text],
        n=k * 2,
        keyword_weight=0.3,
    )
    semantic_results = [r for r in semantic_results if r["path"] != path]

    seen: dict[str, dict] = {}
    for result in semantic_results:
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

    for neighbor in graph_neighbors:
        if neighbor not in seen:
            seen[neighbor] = {
                "path": neighbor,
                "title": os.path.splitext(os.path.basename(neighbor))[0],
                "similarity_score": round(graph_weight, 4),
                "is_graph_connected": True,
            }

    results = sorted(seen.values(), key=lambda x: x["similarity_score"], reverse=True)[:k]
    return json.dumps(results, ensure_ascii=False, indent=2)


def _handle_traverse(path: str, max_depth: int = 2) -> str:
    path = _normalize_path(path)
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
    return json.dumps(output, ensure_ascii=False, indent=2)


def _handle_export(format: str = "json") -> str:
    if format == "dot":
        return graph_store.to_dot()
    data = graph_store.to_dict()
    return json.dumps(data, indent=2, ensure_ascii=False)


_HANDLERS = {
    "communities": _handle_communities,
    "community_of": _handle_community_of,
    "orphans": _handle_orphans,
    "path": _handle_path,
    "stats": _handle_stats,
    "related": _handle_related,
    "traverse": _handle_traverse,
    "export": _handle_export,
}


@build_tool("graph")
def graph(
    action: str,
    path: str = "",
    start: str = "",
    end: str = "",
    max_depth: int = 2,
    k: int = 10,
    top_k: int = 5,
    graph_weight: float = 0.3,
    format: str = "json",
) -> str:
    """Explore the vault's wiki-link graph structure — communities, paths, orphans, and more.

    Args:
        action: ``communities`` — detect densely connected groups of notes via label propagation.
                ``community_of`` — identify a note's community and its nearest neighbors.
                ``orphans`` — find notes with no incoming or outgoing wiki-links.
                ``path`` — find the shortest chain of wiki-links between two notes.
                ``stats`` — return summary statistics about the vault graph.
                ``related`` — find semantically similar or wiki-link-connected notes.
                ``traverse`` — BFS outward from a seed note up to N hops.
                ``export`` — export the graph as JSON or Graphviz DOT.
        path: vault-relative note path (for community_of, related, traverse).
        start: starting note path (for path).
        end: target note path (for path).
        max_depth: max BFS hops for traverse (default 2).
        k: max results for related (default 10).
        top_k: max community members to list (default 5).
        graph_weight: graph vs semantic weight for related (default 0.3).
        format: export format — ``"json"`` or ``"dot"`` (default ``"json"``).

    Returns:
        JSON data or formatted text.
    """
    if action not in _VALID_ACTIONS:
        valid = ", ".join(sorted(_VALID_ACTIONS))
        return f"Error: Invalid action '{action}'. Valid actions: {valid}"

    handler = _HANDLERS[action]
    try:
        if action == "communities":
            return handler()
        elif action == "community_of":
            return handler(path=path, top_k=top_k)
        elif action == "orphans":
            return handler()
        elif action == "path":
            return handler(start=start, end=end)
        elif action == "stats":
            return handler()
        elif action == "related":
            return handler(path=path, k=k, graph_weight=graph_weight)
        elif action == "traverse":
            return handler(path=path, max_depth=max_depth)
        elif action == "export":
            return handler(format=format)
        else:
            return f"Error: Unhandled action '{action}'"
    except Exception as e:
        log_error(log, f"graph — action={action} FAILED", exc=e)
        return f"Error: {e}"


__all_tools__ = [graph]
