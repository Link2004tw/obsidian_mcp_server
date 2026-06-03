"""Entity index store — inverted index mapping entity names to note mentions."""

import json
import os
import threading
from collections import defaultdict

from . import config

_ENTITY_TYPES = {"Person", "Project", "Hardware", "Technology", "Location", "Concept", "Event"}


def _normalize(name: str) -> str:
    """Casefold normalization for entity name lookups."""
    return name.casefold().strip()


class EntityStore:
    """Persistent inverted index of entities → note mentions.

    Data shape (stored in ``data/entities.json``)::

        {
          "entities": {
            "<normalized_name>": {
              "canonical": "<best-cased-name>",
              "type": "<entity_type>",
              "aliases": ["<alt_name>", ...],
              "mentions": [
                {
                  "path": "<vault-relative-path>",
                  "chunk_idx": 0,
                  "context": "<surrounding-text-snippet>",
                  "confidence": 0.95
                },
                ...
              ]
            },
            ...
          }
        }
    """

    def __init__(self, path: str | None = None):
        self._path = path or os.path.join(
            os.path.dirname(config.chroma_path) or "data", "entities.json"
        )
        self._data: dict[str, dict] = {}  # normalized_name -> entity record
        self._lock = threading.Lock()
        self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self._data = json.loads(content).get("entities", {})
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def save(self) -> None:
        with self._lock:
            data_copy = dict(self._data)
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"entities": data_copy}, f, indent=2, ensure_ascii=False)

    # ── Add ──────────────────────────────────────────────────────────

    def add(
        self,
        name: str,
        type: str,
        confidence: float,
        path: str,
        chunk_idx: int = 0,
        context: str = "",
    ) -> None:
        """Record an entity mention found in a note chunk. Thread-safe."""
        if type not in _ENTITY_TYPES:
            type = "Concept"
        key = _normalize(name)
        with self._lock:
            record = self._data.get(key)
            if record is None:
                record = {
                    "canonical": name,
                    "type": type,
                    "aliases": [],
                    "mentions": [],
                }
                self._data[key] = record
            else:
                if record["type"] != type and not _is_broader_type(record["type"], type):
                    record["type"] = type
                _maybe_add_alias(record, name)

            mention: dict = {
                "path": path,
                "chunk_idx": chunk_idx,
                "context": context[:200],
                "confidence": round(confidence, 4),
            }
            # Deduplicate mentions for same (path, chunk_idx)
            for existing in record["mentions"]:
                if existing["path"] == path and existing["chunk_idx"] == chunk_idx:
                    if confidence > existing["confidence"]:
                        existing["confidence"] = round(confidence, 4)
                        existing["context"] = context[:200]
                    return
            record["mentions"].append(mention)
            # Keep mentions sorted by confidence descending
            record["mentions"].sort(key=lambda m: m["confidence"], reverse=True)

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        name: str,
        type: str | None = None,
        n: int = 10,
    ) -> list[dict]:
        """Find notes mentioning an entity. Returns deduplicated results."""
        key = _normalize(name)
        record = self._data.get(key)
        if record is None:
            return []

        if type and record["type"] != type:
            return []

        seen_paths: set[str] = set()
        results: list[dict] = []
        for m in record["mentions"]:
            if m["path"] not in seen_paths:
                seen_paths.add(m["path"])
                results.append({
                    "path": m["path"],
                    "entity_name": record["canonical"],
                    "entity_type": record["type"],
                    "snippet": m.get("context", ""),
                    "confidence": m["confidence"],
                })
                if len(results) >= n:
                    break
        return results

    def search_by_type(self, type: str, n: int = 20) -> list[dict]:
        """Find all entities of a given type."""
        results: list[dict] = []
        for record in self._data.values():
            if record["type"] == type:
                results.append({
                    "entity_name": record["canonical"],
                    "entity_type": record["type"],
                    "mention_count": len(record["mentions"]),
                })
        results.sort(key=lambda r: r["mention_count"], reverse=True)
        return results[:n]

    # ── Note queries ─────────────────────────────────────────────────

    def get_note_entities(self, path: str) -> list[dict]:
        """Return all entities found in a specific note."""
        results: list[dict] = []
        for record in self._data.values():
            for m in record["mentions"]:
                if m["path"] == path:
                    results.append({
                        "entity_name": record["canonical"],
                        "entity_type": record["type"],
                        "confidence": m["confidence"],
                    })
                    break
        return sorted(results, key=lambda r: r["confidence"], reverse=True)

    # ── Rebuild ──────────────────────────────────────────────────────

    def clear(self) -> None:
        """Wipe all entity data (used on full re-index). Thread-safe."""
        with self._lock:
            self._data = {}

    def rebuild(self) -> None:
        """Rebuild the entity store from ChromaDB metadata."""
        from . import chroma_store

        self._data = {}
        _, _, metadatas = chroma_store.get_all_documents()
        for meta in metadatas:
            entities_str = meta.get("entities_str", "")
            if not entities_str:
                continue
            path = meta.get("path", "")
            chunk_idx = meta.get("chunk", 0)
            for token in entities_str.strip(",").split(","):
                token = token.strip()
                if not token:
                    continue
                parts = token.split(":", 1)
                if len(parts) == 2:
                    ent_type, ent_name = parts
                    self.add(
                        name=ent_name,
                        type=ent_type,
                        confidence=1.0,
                        path=path,
                        chunk_idx=chunk_idx,
                        context="",
                    )
        self.save()

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return entity index statistics."""
        if not self._data:
            return {"total_entities": 0, "total_mentions": 0, "by_type": {}}

        total_mentions = sum(len(r["mentions"]) for r in self._data.values())
        by_type: dict[str, int] = defaultdict(int)
        for r in self._data.values():
            by_type[r["type"]] += 1

        return {
            "total_entities": len(self._data),
            "total_mentions": total_mentions,
            "by_type": dict(by_type),
        }


# ── Module-level singleton ──────────────────────────────────────────

_store: EntityStore | None = None


def _get_store() -> EntityStore:
    global _store
    if _store is None:
        _store = EntityStore()
    return _store


# ── Helper functions ────────────────────────────────────────────────


_TYPE_SPECIFICITY = {
    "Hardware": "Technology",
    "Technology": "Concept",
    "Project": "Concept",
    "Person": "Concept",
    "Location": "Concept",
    "Event": "Concept",
}


def _is_broader_type(current: str, new: str) -> bool:
    """Check if *new* is a broader (more general) type than *current*."""
    t = _TYPE_SPECIFICITY.get(current)
    while t is not None:
        if t == new:
            return True
        t = _TYPE_SPECIFICITY.get(t)
    return False


def _maybe_add_alias(record: dict, name: str) -> None:
    """Add *name* as an alias if it differs from the canonical form."""
    if _normalize(name) == _normalize(record["canonical"]):
        return
    if name not in record["aliases"]:
        record["aliases"].append(name)


# ── Public API (module-level convenience functions) ─────────────────


def add(name: str, type: str, confidence: float, path: str,
        chunk_idx: int = 0, context: str = "") -> None:
    _get_store().add(name, type, confidence, path, chunk_idx, context)


def search(name: str, type: str | None = None, n: int = 10) -> list[dict]:
    return _get_store().search(name, type=type, n=n)


def search_by_type(type: str, n: int = 20) -> list[dict]:
    return _get_store().search_by_type(type, n=n)


def get_note_entities(path: str) -> list[dict]:
    return _get_store().get_note_entities(path)


def clear() -> None:
    _get_store().clear()


def rebuild() -> None:
    _get_store().rebuild()


def save() -> None:
    _get_store().save()


def stats() -> dict:
    return _get_store().stats()


def entity_types() -> list[str]:
    return sorted(_ENTITY_TYPES)
