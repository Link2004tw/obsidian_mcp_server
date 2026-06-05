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
        self._alias_map: dict[str, str] = {}  # normalized_alias -> normalized_canonical
        self._lock = threading.Lock()
        self._load()
        self._load_manual_aliases()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.isfile(self._path):
            try:
                with open(self._path, encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        self._data = data.get("entities", {})
                        self._alias_map = {
                            k: v for d in data.get("alias_map", {})
                            for k, v in d.items()
                        } if isinstance(data.get("alias_map"), list) else dict(data.get("alias_map", {}))
            except (json.JSONDecodeError, OSError):
                self._data = {}
                self._alias_map = {}
        self._rebuild_alias_map()

    def save(self) -> None:
        with self._lock:
            data_copy = dict(self._data)
            alias_copy = dict(self._alias_map)
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"entities": data_copy, "alias_map": alias_copy}, f, indent=2, ensure_ascii=False)

    # ── Alias Map ────────────────────────────────────────────────────

    def _rebuild_alias_map(self) -> None:
        """Rebuild the alias_map from all entity records."""
        self._alias_map = {}
        for key, record in self._data.items():
            for alias in record.get("aliases", []):
                alias_key = _normalize(alias)
                if alias_key != key:
                    self._alias_map[alias_key] = key

    def _register_alias(self, record_key: str, alias: str) -> None:
        """Register a single alias in the alias map (thread-safe, caller must hold lock)."""
        alias_key = _normalize(alias)
        if alias_key != record_key:
            self._alias_map[alias_key] = record_key

    # ── Manual Aliases ───────────────────────────────────────────────

    def _manual_aliases_path(self) -> str:
        return getattr(config, "entity_aliases_file", "") or os.path.join(
            os.path.dirname(self._path) or "data", "entity_aliases.json"
        )

    def _load_manual_aliases(self) -> None:
        """Load user-defined aliases from ``entity_aliases.json`` and apply them.

        Manual aliases take precedence over LLM-generated aliases.
        """
        path = self._manual_aliases_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        aliases_by_entity: dict[str, list[str]] = (
            raw if isinstance(raw, dict) else {}
        )
        if not aliases_by_entity:
            return

        with self._lock:
            for canonical_name, alias_list in aliases_by_entity.items():
                key = _normalize(canonical_name)
                record = self._data.get(key)
                if record is None:
                    # Create a minimal record — type will be inferred on next index
                    record = {
                        "canonical": canonical_name,
                        "type": "Concept",
                        "aliases": [],
                        "mentions": [],
                    }
                    self._data[key] = record
                for alias in alias_list:
                    if alias and alias not in record["aliases"]:
                        record["aliases"].append(alias)
                        self._register_alias(key, alias)

    # ── Add ──────────────────────────────────────────────────────────

    def add(
        self,
        name: str,
        type: str,
        confidence: float,
        path: str,
        chunk_idx: int = 0,
        context: str = "",
        aliases: list[str] | None = None,
    ) -> None:
        """Record an entity mention found in a note chunk. Thread-safe.

        Args:
            name: canonical entity name (e.g. ``"ESP32"``).
            type: entity type from ``_ENTITY_TYPES``.
            confidence: 0.0–1.0 confidence score.
            path: vault-relative note path.
            chunk_idx: chunk index within the note.
            context: surrounding text snippet.
            aliases: optional list of alternative names for this entity
                (e.g. ``["ESP-32", "esp32 dev board"]``).
        """
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

            # Register LLM-generated / user-provided aliases
            if aliases:
                for alias in aliases:
                    alias_str = str(alias).strip()
                    if alias_str and alias_str not in record["aliases"]:
                        record["aliases"].append(alias_str)
                        self._register_alias(key, alias_str)

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
        """Find notes mentioning an entity. Returns deduplicated results.

        Looks up the name directly first, then falls back to the alias map
        so queries like ``"Her"`` can find entity ``"Maria"``.
        """
        key = _normalize(name)
        record = self._data.get(key)
        if record is None:
            canonical_key = self._alias_map.get(key)
            if canonical_key:
                record = self._data.get(canonical_key)

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

    # ── Aliases ──────────────────────────────────────────────────────

    def get_aliases(self, name: str) -> dict | None:
        """Return all alias information for a given entity.

        Args:
            name: entity name (canonical or alias).

        Returns:
            Dict with ``canonical``, ``type``, ``aliases`` (list), and
            ``mention_count``, or ``None`` if the entity is not found.
        """
        key = _normalize(name)
        record = self._data.get(key)
        if record is None:
            canonical_key = self._alias_map.get(key)
            if canonical_key:
                record = self._data.get(canonical_key)
        if record is None:
            return None
        return {
            "canonical": record["canonical"],
            "type": record["type"],
            "aliases": list(record.get("aliases", [])),
            "mention_count": len(record["mentions"]),
        }

    def merge(self, primary: str, secondary: str) -> dict | None:
        """Merge *secondary* entity into *primary*.

        Combines mention lists, merges aliases, keeps the primary canonical
        name, and removes the secondary record. The alias map is rebuilt
        afterward.

        Args:
            primary: the entity name to keep (canonical).
            secondary: the entity name to merge and then delete.

        Returns:
            The updated primary record dict, or ``None`` if either entity
            is not found.
        """
        primary_key = _normalize(primary)
        secondary_key = _normalize(secondary)

        if primary_key == secondary_key:
            return None

        with self._lock:
            primary_rec = self._data.get(primary_key)
            secondary_rec = self._data.get(secondary_key)

            if primary_rec is None or secondary_rec is None:
                return None

            # Merge mentions (deduplicate by (path, chunk_idx))
            seen = {(m["path"], m["chunk_idx"]) for m in primary_rec["mentions"]}
            for m in secondary_rec["mentions"]:
                key_pair = (m["path"], m["chunk_idx"])
                if key_pair not in seen:
                    seen.add(key_pair)
                    primary_rec["mentions"].append(m)

            primary_rec["mentions"].sort(key=lambda x: x["confidence"], reverse=True)

            # Merge aliases
            for alias in secondary_rec.get("aliases", []):
                if alias not in primary_rec["aliases"]:
                    primary_rec["aliases"].append(alias)

            # Add the secondary name itself as an alias
            secondary_canonical = secondary_rec["canonical"]
            if secondary_canonical not in primary_rec["aliases"]:
                primary_rec["aliases"].append(secondary_canonical)

            # Remove secondary record
            del self._data[secondary_key]

            # Rebuild alias map
            self._rebuild_alias_map()

        self.save()
        return {
            "canonical": primary_rec["canonical"],
            "type": primary_rec["type"],
            "aliases": list(primary_rec["aliases"]),
            "mention_count": len(primary_rec["mentions"]),
        }

    def list_entities(self, entity_type: str | None = None, n: int = 1000) -> list[dict]:
        """List all entities, optionally filtered by entity type.

        Returns sorted by mention count descending.
        """
        results = []
        for record in self._data.values():
            if entity_type and record["type"] != entity_type:
                continue
            results.append({
                "entity_name": record["canonical"],
                "entity_type": record["type"],
                "mention_count": len(record["mentions"]),
            })
        results.sort(key=lambda r: r["mention_count"], reverse=True)
        return results[:n]

    def add_manual_entity(self, name: str, entity_type: str, aliases: list[str] | None = None) -> dict:
        """Manually add an entity to the index without requiring a note mention.

        Thread-safe. Persists immediately.

        Args:
            name: canonical entity name (e.g. ``"ESP32"``).
            entity_type: type from ``_ENTITY_TYPES`` (defaults to ``"Concept"``).
            aliases: optional list of alternative names.

        Returns:
            The created/updated entity record dict.
        """
        if entity_type not in _ENTITY_TYPES:
            entity_type = "Concept"
        key = _normalize(name)
        with self._lock:
            record = self._data.get(key)
            if record is None:
                record = {
                    "canonical": name,
                    "type": entity_type,
                    "aliases": [],
                    "mentions": [],
                }
                self._data[key] = record
            else:
                record["type"] = entity_type
            if aliases:
                for alias in aliases:
                    alias_str = str(alias).strip()
                    if alias_str and alias_str not in record["aliases"]:
                        record["aliases"].append(alias_str)
                        self._register_alias(key, alias_str)
        self.save()
        return {
            "entity_name": record["canonical"],
            "entity_type": record["type"],
            "aliases": list(record["aliases"]),
            "mention_count": len(record["mentions"]),
        }

    # ── Rebuild ──────────────────────────────────────────────────────

    def clear(self) -> None:
        """Wipe all entity data (used on full re-index). Thread-safe."""
        with self._lock:
            self._data = {}
            self._alias_map = {}

    def rebuild(self) -> None:
        """Rebuild the entity store from ChromaDB metadata."""
        from . import chroma_store

        self._data = {}
        self._alias_map = {}
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
        self._load_manual_aliases()
        self.save()

    # ── Timeline ─────────────────────────────────────────────────────

    def add_timeline_entry(
        self,
        entity_name: str,
        date: str,
        event: str,
        note: str = "",
        confidence: float = 0.5,
    ) -> None:
        """Record a timeline event for an entity. Thread-safe.

        Args:
            entity_name: canonical entity name (e.g. ``"Alice"``).
            date: temporal reference (e.g. ``"2024-01"``, ``"2024"``,
                ``"early 2024"``).
            event: description of what happened.
            note: vault-relative note path that mentions this event.
            confidence: 0.0–1.0 confidence score.
        """
        key = _normalize(entity_name)
        with self._lock:
            record = self._data.get(key)
            if record is None:
                return
            timeline = record.setdefault("timeline", [])
            entry = {
                "date": date,
                "event": event,
                "note": note,
                "confidence": round(confidence, 4),
            }
            # Deduplicate by (date, event, note)
            for existing in timeline:
                if (existing["date"] == date
                        and existing["event"] == event
                        and existing["note"] == note):
                    if confidence > existing["confidence"]:
                        existing["confidence"] = round(confidence, 4)
                    return
            timeline.append(entry)
            timeline.sort(key=lambda e: e["date"])

    def get_timeline(
        self,
        name: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict] | None:
        """Return timeline entries for an entity, sorted by date.

        Args:
            name: entity name (canonical or alias).
            date_from: optional lower bound for date filtering (inclusive).
            date_to: optional upper bound for date filtering (inclusive).

        Returns:
            List of timeline entries sorted by date, or ``None`` if the
            entity is not found.
        """
        key = _normalize(name)
        record = self._data.get(key)
        if record is None:
            canonical_key = self._alias_map.get(key)
            if canonical_key:
                record = self._data.get(canonical_key)
        if record is None:
            return None

        timeline = list(record.get("timeline", []))
        if date_from:
            timeline = [e for e in timeline if e["date"] >= date_from]
        if date_to:
            timeline = [e for e in timeline if e["date"] <= date_to]
        timeline.sort(key=lambda e: e["date"])
        return timeline

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
        chunk_idx: int = 0, context: str = "", aliases: list[str] | None = None) -> None:
    _get_store().add(name, type, confidence, path, chunk_idx, context, aliases=aliases)


def search(name: str, type: str | None = None, n: int = 10) -> list[dict[str, str | float]]:
    return _get_store().search(name, type=type, n=n)


def search_by_type(type: str, n: int = 20) -> list[dict[str, str | int]]:
    return _get_store().search_by_type(type, n=n)


def get_note_entities(path: str) -> list[dict[str, str | float]]:
    return _get_store().get_note_entities(path)


def get_aliases(name: str) -> dict | None:
    return _get_store().get_aliases(name)


def merge(primary: str, secondary: str) -> dict | None:
    return _get_store().merge(primary, secondary)


def clear() -> None:
    _get_store().clear()


def rebuild() -> None:
    _get_store().rebuild()


def save() -> None:
    _get_store().save()


def stats() -> dict[str, int | dict[str, int]]:
    return _get_store().stats()


def entity_types() -> list[str]:
    return sorted(_ENTITY_TYPES)


def add_timeline_entry(entity_name: str, date: str, event: str,
                       note: str = "", confidence: float = 0.5) -> None:
    _get_store().add_timeline_entry(entity_name, date, event, note=note, confidence=confidence)


def get_timeline(name: str, date_from: str | None = None,
                 date_to: str | None = None) -> list[dict] | None:
    return _get_store().get_timeline(name, date_from=date_from, date_to=date_to)


def list_entities(entity_type: str | None = None, n: int = 1000) -> list[dict]:
    return _get_store().list_entities(entity_type=entity_type, n=n)


def add_manual_entity(name: str, entity_type: str, aliases: list[str] | None = None) -> dict:
    return _get_store().add_manual_entity(name, entity_type, aliases=aliases)
