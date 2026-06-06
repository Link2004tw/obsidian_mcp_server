"""Cross-vault entity resolution — import and merge entities from another vault."""

from difflib import SequenceMatcher
from typing import cast

from . import entity_relations, entity_store
from .logger import get_logger

log = get_logger(__name__)

DEFAULT_CONFIG = {
    "exact_match": True,
    "alias_match": True,
    "fuzzy_threshold": 0.85,
    "strategy": "auto",
}


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.casefold(), b.casefold()).ratio()


class EntityResolver:
    def __init__(self, dedup_config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(dedup_config or {})}

    def resolve(self, data: dict) -> dict:
        """Import entities + relations from external JSON, merging into current store.

        Args:
            data: dict with ``entities`` and optionally ``relations`` keys.

        Returns:
            Summary dict with counts of merged, added, skipped, and relation counts.
        """
        incoming_entities = data.get("entities", {})
        incoming_relations = data.get("relations", [])

        merged_count = 0
        added_count = 0
        skipped_count = 0
        added_relations = 0

        # Phase 1: resolve entities
        for norm_key, record in incoming_entities.items():
            canonical = record.get("canonical", norm_key)
            ent_type = record.get("type", "Concept")
            aliases = record.get("aliases", [])

            matched = self._find_match(norm_key, canonical, aliases)

            if matched:
                existing = entity_store.get_aliases(matched)
                if existing:
                    entity_store.merge(matched, canonical)
                    merged_count += 1
                else:
                    skipped_count += 1
            else:
                entity_store.add_manual_entity(canonical, ent_type, aliases=aliases)
                added_count += 1

        # Phase 2: resolve relations
        for rel in incoming_relations:
            source = rel.get("source", "")
            rel_type = rel.get("type", "related_to")
            target = rel.get("target", "")
            confidence = rel.get("confidence", 0.5)
            if source and target:
                entity_relations.add(
                    source=source,
                    type=rel_type,
                    target=target,
                    confidence=confidence,
                )
                added_relations += 1

        if added_relations:
            entity_relations.save()
        entity_store.save()

        summary = {
            "merged": merged_count,
            "added": added_count,
            "skipped": skipped_count,
            "relations_added": added_relations,
            "total_incoming": len(incoming_entities),
        }
        log.info(
            "Entity import complete — merged=%d, added=%d, skipped=%d, relations=%d",
            merged_count, added_count, skipped_count, added_relations,
        )
        return summary

    def _find_match(self, norm_key: str, canonical: str, aliases: list[str]) -> str | None:
        """Try each matching strategy in order. Returns matched canonical name or None."""

        # 1. Exact match on normalized canonical name
        if self.config.get("exact_match", True):
            existing = entity_store.get_aliases(canonical)
            if existing is not None:
                return cast(str, existing["canonical"])

        # 2. Alias overlap — check if any incoming alias matches an existing entity
        if self.config.get("alias_match", True):
            for alias in [canonical] + aliases:
                existing = entity_store.get_aliases(alias)
                if existing is not None:
                    return cast(str, existing["canonical"])

        # 3. Fuzzy match against all canonical names
        fuzzy_threshold = self.config.get("fuzzy_threshold", 0.85)
        if fuzzy_threshold > 0:
            for existing in entity_store.list_entities(n=10000):
                ratio = _fuzzy_ratio(canonical, existing["entity_name"])
                if ratio >= fuzzy_threshold:
                    return cast(str, existing["entity_name"])

        return None
