"""Graph store for Obsidian wiki-link relationships."""
import json
import os
from collections import Counter

from . import config
from .wiki_links import extract_wiki_links, normalize_wiki_link_target


class GraphStore:
    """Maintains an adjacency list of wiki-link relationships between notes.

    Persists to JSON and supports incremental updates during indexing.
    """

    def __init__(self, path: str | None = None):
        self._path = path or os.path.join(config.data_dir, "graph.json")
        self._adj: dict[str, set[str]] = {}  # source_path -> set of target_paths
        self._title_map: dict[str, str] = {}  # normalized title -> file path
        self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.isfile(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return
                data = json.loads(content)
            self._adj = {k: set(v) for k, v in data.get("edges", {}).items()}
            self._title_map = dict(data.get("title_map", {}))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        data = {
            "edges": {k: sorted(v) for k, v in self._adj.items()},
            "title_map": self._title_map,
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Title → Path Resolution ─────────────────────────────────────

    @staticmethod
    def _note_title(path: str) -> str:
        """Return the normalized (casefolded) title for a note path."""
        return os.path.splitext(os.path.basename(path))[0].casefold()

    def _build_title_map(self, all_notes: dict[str, str]) -> None:
        """Build title → path mapping. First path wins for duplicate titles."""
        self._title_map = {}
        for path in sorted(all_notes.keys()):
            title = self._note_title(path)
            if title not in self._title_map:
                self._title_map[title] = path

    def resolve_link(self, link_target: str) -> str | None:
        """Resolve a normalized wiki-link target to a file path."""
        return self._title_map.get(link_target)

    # ── Rebuild ──────────────────────────────────────────────────────

    def rebuild(self, all_notes: dict[str, str]) -> None:
        """Rebuild the entire graph from scratch given {path: raw_content}."""
        self._build_title_map(all_notes)
        self._adj = {}

        # Ensure all nodes exist
        for path in all_notes:
            if path not in self._adj:
                self._adj[path] = set()

        # Extract links and build edges
        for path, content in all_notes.items():
            links = extract_wiki_links(content)
            for link in links:
                resolved = self.resolve_link(link)
                if resolved is not None and resolved != path:
                    self._adj.setdefault(path, set()).add(resolved)
                    # Ensure target node exists even if it has no outgoing links
                    if resolved not in self._adj:
                        self._adj[resolved] = set()

    # ── Edge Operations ──────────────────────────────────────────────

    def add_edge(self, source: str, target: str) -> None:
        """Add a directed edge from source to target."""
        self._adj.setdefault(source, set()).add(target)
        if target not in self._adj:
            self._adj[target] = set()

    def remove_node(self, path: str) -> None:
        """Remove a node and all its edges (also cleans up title map)."""
        self._adj.pop(path, None)
        for targets in self._adj.values():
            targets.discard(path)
        title = self._note_title(path)
        if self._title_map.get(title) == path:
            del self._title_map[title]

    def rename_node(self, old: str, new: str) -> None:
        """Rename a node: transfer edges and update references."""
        edges = self._adj.pop(old, set())
        self._adj[new] = edges
        for source, targets in self._adj.items():
            if old in targets:
                targets.discard(old)
                targets.add(new)
        # Update title map
        old_title = self._note_title(old)
        new_title = self._note_title(new)
        if self._title_map.get(old_title) == old:
            del self._title_map[old_title]
            self._title_map[new_title] = new

    def register_title(self, path: str) -> None:
        """Register a note's title in the title map (for incremental indexing)."""
        title = self._note_title(path)
        if title not in self._title_map:
            self._title_map[title] = path

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._adj)

    # ── Queries ──────────────────────────────────────────────────────

    def get_backlinks(self, path: str) -> list[str]:
        """Return all notes linking TO the given note (incoming edges)."""
        return sorted([src for src, targets in self._adj.items() if path in targets])

    def get_outgoing(self, path: str) -> list[str]:
        """Return all notes the given note links TO (outgoing edges)."""
        return sorted(self._adj.get(path, set()))

    def bfs(self, start: str, max_depth: int = 1) -> dict[str, list[str]]:
        """BFS from start up to max_depth hops. Returns {path: [trace], ...}.

        Excludes the start node itself from results.
        """
        if start not in self._adj:
            return {}

        visited: set[str] = {start}
        results: dict[str, list[str]] = {}
        queue: list[tuple[str, list[str]]] = [(start, [start])]

        for _ in range(max_depth):
            next_queue: list[tuple[str, list[str]]] = []
            for node, trace in queue:
                for neighbor in self._adj.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_trace = trace + [neighbor]
                        results[neighbor] = new_trace
                        next_queue.append((neighbor, new_trace))
            queue = next_queue
            if not queue:
                break

        return results

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return graph statistics (excluding entity nodes)."""
        note_adj = self._non_entity_nodes()
        # Filter entity targets from adjacency for stats
        adj = {k: {t for t in v if not self.is_entity_node(t)} for k, v in note_adj.items()}
        nodes = len(adj)
        edge_count = sum(len(t) for t in adj.values())
        avg_degree = edge_count / nodes if nodes > 0 else 0

        # Incoming edge counts
        incoming: Counter = Counter()
        for targets in adj.values():
            for t in targets:
                incoming[t] += 1

        # Total degree (in + out)
        degree: Counter = Counter()
        for src, targets in adj.items():
            degree[src] += len(targets)  # outgoing
            degree[src] += incoming.get(src, 0)  # incoming
            for t in targets:
                degree[t] += 1  # incoming from src

        isolated = [p for p in adj if not adj[p] and p not in incoming]
        hubs = degree.most_common(5)

        return {
            "nodes": nodes,
            "edges": edge_count,
            "avg_degree": round(avg_degree, 2),
            "isolated_count": len(isolated),
            "isolated": sorted(isolated),
            "hubs": [{"path": p, "degree": d} for p, d in hubs],
        }

    # ── Broken Links ─────────────────────────────────────────────────

    def get_broken_links(self, all_notes: dict[str, str]) -> list[dict]:
        """Find wiki-links that don't resolve to any known note.

        Returns list of {source_path, link_target, normalized_target}.
        """
        self._build_title_map(all_notes)
        broken: list[dict] = []
        for path, content in all_notes.items():
            for link in extract_wiki_links(content):
                resolved = self.resolve_link(link)
                if resolved is None:
                    broken.append({
                        "source_path": path,
                        "link_target": link,
                    })
        return broken

    # ── Orphans ──────────────────────────────────────────────────────

    def get_orphans(self) -> list[str]:
        """Return notes with no incoming or outgoing wiki-links (excluding entity nodes)."""
        note_adj = self._non_entity_nodes()
        has_outgoing = {src for src, targets in note_adj.items() if targets}
        has_incoming: set[str] = set()
        for targets in self._adj.values():
            has_incoming.update(targets)
        # Remove entity nodes and their connected notes from incoming counts
        for src, targets in self._adj.items():
            if self.is_entity_node(src):
                for t in targets:
                    has_incoming.discard(t)
        has_incoming = {p for p in has_incoming if not self.is_entity_node(p)}
        connected = has_outgoing | has_incoming
        return sorted([p for p in note_adj if p not in connected])

    # ── Entity Nodes ─────────────────────────────────────────────────

    @staticmethod
    def _entity_node_id(entity_type: str, entity_name: str) -> str:
        """Create a unique node ID for an entity: ``__entity:{type}:{name}``."""
        return f"__entity:{entity_type}:{entity_name}"

    def add_entity_edge(self, entity_type: str, entity_name: str, note_path: str) -> None:
        """Add a directed edge from an entity node to a note."""
        eid = self._entity_node_id(entity_type, entity_name)
        self._adj.setdefault(eid, set()).add(note_path)
        if note_path not in self._adj:
            self._adj[note_path] = set()

    def remove_entity_edges(self, entity_type: str, entity_name: str) -> None:
        """Remove all edges for a specific entity node."""
        eid = self._entity_node_id(entity_type, entity_name)
        self._adj.pop(eid, None)

    def get_entity_notes(self, entity_type: str, entity_name: str) -> list[str]:
        """Return all note paths linked to an entity."""
        eid = self._entity_node_id(entity_type, entity_name)
        return sorted(self._adj.get(eid, set()))

    def get_entity_nodes(self) -> list[dict]:
        """Return all entity nodes in the graph."""
        results: list[dict] = []
        for node in self._adj:
            if node.startswith("__entity:"):
                parts = node.split(":", 2)
                if len(parts) == 3:
                    results.append({
                        "entity_type": parts[1],
                        "entity_name": parts[2],
                        "linked_notes": len(self._adj[node]),
                    })
        return sorted(results, key=lambda r: r["entity_name"])

    def is_entity_node(self, node: str) -> bool:
        """Check if a node ID represents an entity."""
        return node.startswith("__entity:")

    def _non_entity_nodes(self) -> dict[str, set[str]]:
        """Return a view of the adjacency dict excluding entity nodes."""
        return {k: v for k, v in self._adj.items() if not self.is_entity_node(k)}

    # ── Export ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return the graph as a plain dict (for MCP responses, excluding entity nodes)."""
        note_adj = self._non_entity_nodes()
        edges = {}
        for k, v in note_adj.items():
            # Only include note-to-note edges
            edges[k] = sorted(t for t in v if not self.is_entity_node(t))
        return {
            "edges": edges,
            "title_map": self._title_map,
        }

    def from_dict(self, data: dict) -> None:
        """Load graph from a plain dict."""
        self._adj = {k: set(v) for k, v in data.get("edges", {}).items()}
        self._title_map = dict(data.get("title_map", {}))

    def to_dot(self) -> str:
        """Export the graph as DOT format for visualization (Graphviz, etc.)."""
        note_adj = self._non_entity_nodes()
        lines = ["digraph ObsidianVault {"]
        lines.append("  rankdir=LR;")
        lines.append('  node [shape=box, style=rounded];')
        for node in sorted(note_adj):
            label = os.path.splitext(os.path.basename(node))[0]
            label = label.replace('"', '\\"')
            lines.append(f'  "{node}" [label="{label}"];')
        for src, targets in note_adj.items():
            note_targets = sorted(t for t in targets if not self.is_entity_node(t))
            for tgt in note_targets:
                lines.append(f'  "{src}" -> "{tgt}";')
        lines.append("}")
        return "\n".join(lines)

    # ── Community Detection ──────────────────────────────────────────

    def label_propagation(self, max_iter: int = 100) -> dict[str, int]:
        """Detect communities using label propagation algorithm (entity nodes excluded).

        Each node starts with its own unique label. Iteratively, each node
        adopts the most frequent label among its neighbors. Converges when
        no labels change or max_iter is reached.

        Returns:
            {path: community_id} mapping — communities are 0-indexed integers.
        """
        note_adj = self._non_entity_nodes()
        nodes = list(note_adj.keys())
        if not nodes:
            return {}

        labels: dict[str, int] = {node: i for i, node in enumerate(nodes)}

        for _ in range(max_iter):
            changed = False
            for node in nodes:
                neighbors = note_adj[node]
                # Also consider backlinks (incoming edges)
                backlinks = {src for src in note_adj if node in note_adj[src]}
                all_neighbors = neighbors | backlinks

                if not all_neighbors:
                    continue

                neighbor_labels: dict[int, int] = {}
                for nb in all_neighbors:
                    lbl = labels[nb]
                    neighbor_labels[lbl] = neighbor_labels.get(lbl, 0) + 1

                max_count = max(neighbor_labels.values())
                best = min(lbl for lbl, cnt in neighbor_labels.items() if cnt == max_count)

                if labels[node] != best:
                    labels[node] = best
                    changed = True

            if not changed:
                break

        # Renumber communities to 0..n-1
        unique = sorted(set(labels.values()))
        remap = {old: new for new, old in enumerate(unique)}
        return {node: remap[lbl] for node, lbl in labels.items()}


# ── Module-level singleton ──────────────────────────────────────────

_store: GraphStore | None = None


def _get_store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore()
    return _store


def rebuild(all_notes: dict[str, str]) -> None:
    _get_store().rebuild(all_notes)


def save() -> None:
    _get_store().save()


def remove_node(path: str) -> None:
    _get_store().remove_node(path)


def register_title(path: str) -> None:
    _get_store().register_title(path)


def resolve_link(link_target: str) -> str | None:
    return _get_store().resolve_link(link_target)


def add_edge(source: str, target: str) -> None:
    _get_store().add_edge(source, target)


def rename_node(old: str, new: str) -> None:
    _get_store().rename_node(old, new)


def node_count() -> int:
    return _get_store().node_count()


def get_backlinks(path: str) -> list[str]:
    return _get_store().get_backlinks(path)


def get_outgoing(path: str) -> list[str]:
    return _get_store().get_outgoing(path)


def bfs(start: str, max_depth: int = 1) -> dict[str, list[str]]:
    return _get_store().bfs(start, max_depth=max_depth)


def get_broken_links(all_notes: dict[str, str]) -> list[dict]:
    return _get_store().get_broken_links(all_notes)


def stats() -> dict:
    return _get_store().stats()


def get_orphans() -> list[str]:
    return _get_store().get_orphans()


def to_dict() -> dict:
    return _get_store().to_dict()


def to_dot() -> str:
    return _get_store().to_dot()


def label_propagation(max_iter: int = 100) -> dict[str, int]:
    return _get_store().label_propagation(max_iter=max_iter)
