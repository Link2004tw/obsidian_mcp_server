"""Entity relationship store — directed graph of entity-to-entity relationships.

Relationships are stored as triples ``(source, type, target)`` with a
confidence score. An adjacency dict is built in memory for fast traversal
and persisted to JSON for reloads across restarts.
"""

import json
import os
import threading
from collections import defaultdict

from . import config
from .logger import get_logger

log = get_logger(__name__)

RELATIONSHIP_TYPES: set[str] = {
    "works_on",
    "uses",
    "part_of",
    "related_to",
    "created_by",
    "located_in",
    "attends",
}


def _normalize(name: str) -> str:
    return name.casefold().strip()


class RelationshipStore:
    """Persistent directed graph of entity-to-entity relationships.

    Data shape (stored in ``data/entity_relations.json``)::

        {
          "relationships": [
            {
              "source": "Alice",
              "type": "works_on",
              "target": "ProjectX",
              "confidence": 0.95
            },
            ...
          ]
        }
    """

    def __init__(self, path: str | None = None):
        self._path = path or os.path.join(config.data_dir, "entity_relations.json")
        self._relationships: list[dict] = []
        self._adj: dict[str, dict[str, set[str]]] = {}
        self._lock = threading.Lock()
        self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.isfile(self._path):
            try:
                with open(self._path, encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        self._relationships = data.get("relationships", [])
            except (json.JSONDecodeError, OSError):
                self._relationships = []
        self._build_adj()

    def save(self) -> None:
        with self._lock:
            rels_copy = list(self._relationships)
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"relationships": rels_copy}, f, indent=2, ensure_ascii=False)

    # ── Adjacency ────────────────────────────────────────────────────

    def _build_adj(self) -> None:
        self._adj = {}
        for rel in self._relationships:
            source = _normalize(rel["source"])
            target = _normalize(rel["target"])
            rtype = rel["type"]
            self._adj.setdefault(source, {}).setdefault(rtype, set()).add(target)
            self._adj.setdefault(target, {}).setdefault(rtype, set()).add(source)

    # ── Add ──────────────────────────────────────────────────────────

    def add(
        self,
        source: str,
        type: str,
        target: str,
        confidence: float = 0.5,
        source_note: str = "",
    ) -> None:
        """Record a relationship between two entities. Thread-safe.

        Args:
            source: source entity name (e.g. ``"Alice"``).
            type: relationship type (e.g. ``"works_on"``). If not in
                ``RELATIONSHIP_TYPES``, falls back to ``"related_to"``.
            target: target entity name (e.g. ``"ProjectX"``).
            confidence: 0.0–1.0 confidence score.
            source_note: vault-relative path of the note that contained
                this relationship (for provenance).
        """
        if type not in RELATIONSHIP_TYPES:
            type = "related_to"
        confidence = max(0.0, min(1.0, confidence))

        with self._lock:
            # Deduplicate: same (source, type, target) updates confidence
            for existing in self._relationships:
                if (existing["source"] == source
                        and existing["type"] == type
                        and existing["target"] == target):
                    existing["confidence"] = max(existing["confidence"], confidence)
                    if source_note and not existing.get("source_note"):
                        existing["source_note"] = source_note
                    return

            rel = {
                "source": source,
                "type": type,
                "target": target,
                "confidence": round(confidence, 4),
            }
            if source_note:
                rel["source_note"] = source_note
            self._relationships.append(rel)

            # Update adjacency
            s_key = _normalize(source)
            t_key = _normalize(target)
            self._adj.setdefault(s_key, {}).setdefault(type, set()).add(t_key)
            self._adj.setdefault(t_key, {}).setdefault(type, set()).add(s_key)

    # ── Query ────────────────────────────────────────────────────────

    def get_related(
        self,
        entity_name: str,
        relation_type: str | None = None,
        depth: int = 1,
    ) -> list[dict]:
        """Find entities related to *entity_name* via the relationship graph.

        Performs BFS up to *depth* hops. Returns deduplicated results
        sorted by confidence descending.

        Args:
            entity_name: the entity to start from (e.g. ``"Alice"``).
            relation_type: optional filter (e.g. ``"works_on"``).
            depth: max traversal depth (default 1).

        Returns:
            List of dicts with ``entity_name``, ``relation_type``,
            ``confidence``, and ``depth`` (hop count).
        """
        start = _normalize(entity_name)
        if start not in self._adj:
            return []

        visited: set[str] = {start}
        results: list[dict] = []
        queue: list[tuple[str, int]] = [(start, 0)]

        for _ in range(depth):
            next_queue: list[tuple[str, int]] = []
            for node, d in queue:
                edges = self._adj.get(node, {})
                for rtype, targets in edges.items():
                    if relation_type and rtype != relation_type:
                        continue
                    for target in targets:
                        if target not in visited:
                            visited.add(target)
                            res = {
                                "entity_name": target,
                                "relation_type": rtype,
                                "depth": d + 1,
                            }
                            # Find max confidence for this relationship
                            confs = [
                                r["confidence"]
                                for r in self._relationships
                                if (_normalize(r["source"]) == node
                                    and _normalize(r["target"]) == target
                                    and r["type"] == rtype)
                                or (_normalize(r["source"]) == target
                                    and _normalize(r["target"]) == node
                                    and r["type"] == rtype)
                            ]
                            res["confidence"] = round(max(confs), 4) if confs else 0.5
                            results.append(res)
                            next_queue.append((target, d + 1))
            queue = next_queue
            if not queue:
                break

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    # ── Bulk add ─────────────────────────────────────────────────────

    def add_many(self, relationships: list[dict]) -> int:
        """Add multiple relationships at once. Returns count added."""
        count = 0
        for rel in relationships:
            if isinstance(rel, dict) and "source" in rel and "target" in rel and "type" in rel:
                self.add(
                    source=rel["source"],
                    type=rel["type"],
                    target=rel["target"],
                    confidence=rel.get("confidence", 0.5),
                    source_note=rel.get("source_note", ""),
                )
                count += 1
        return count

    # ── Clear / Rebuild ──────────────────────────────────────────────

    def clear(self) -> None:
        """Wipe all relationship data. Thread-safe."""
        with self._lock:
            self._relationships = []
            self._adj = {}

    def rebuild(self) -> None:
        """Rebuild adjacency from stored relationships."""
        with self._lock:
            self._build_adj()

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return relationship index statistics."""
        with self._lock:
            total = len(self._relationships)
            by_type: dict[str, int] = defaultdict(int)
            entities: set[str] = set()
            for rel in self._relationships:
                by_type[rel["type"]] += 1
                entities.add(_normalize(rel["source"]))
                entities.add(_normalize(rel["target"]))
            return {
                "total_relationships": total,
                "unique_entities": len(entities),
                "by_type": dict(by_type),
            }


# ── Module-level singleton ──────────────────────────────────────────

_store: RelationshipStore | None = None


def _get_store() -> RelationshipStore:
    global _store
    if _store is None:
        _store = RelationshipStore()
    return _store


def add(source: str, type: str, target: str, confidence: float = 0.5,
        source_note: str = "") -> None:
    _get_store().add(source, type, target, confidence, source_note)


def add_many(relationships: list[dict]) -> int:
    return _get_store().add_many(relationships)


def get_related(entity_name: str, relation_type: str | None = None,
                depth: int = 1) -> list[dict]:
    return _get_store().get_related(entity_name, relation_type, depth)


def clear() -> None:
    _get_store().clear()


def rebuild() -> None:
    _get_store().rebuild()


def save() -> None:
    _get_store().save()


def stats() -> dict:
    return _get_store().stats()
