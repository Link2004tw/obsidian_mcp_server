"""Tests for entity_store.py."""

import os

from obsidian_ai import entity_store


def _make_store():
    """Create a temporary EntityStore for testing."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        fname = f.name
    store = entity_store.EntityStore(path=fname)
    store.clear()
    return store, fname


def test_add_and_search():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.95, "note1.md", chunk_idx=0, context="Alice did X")
        store.add("Alice", "Person", 0.9, "note2.md", chunk_idx=1, context="Alice did Y")
        store.save()

        results = store.search("Alice")
        assert len(results) == 2
        assert results[0]["path"] == "note1.md"
        assert results[0]["entity_name"] == "Alice"
        assert results[0]["entity_type"] == "Person"
    finally:
        os.unlink(path)


def test_search_no_results():
    store, path = _make_store()
    try:
        results = store.search("Nonexistent")
        assert results == []
    finally:
        os.unlink(path)


def test_search_by_type():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.95, "note1.md")
        store.add("Bob", "Person", 0.9, "note2.md")
        store.add("ESP32", "Hardware", 0.98, "note3.md")

        people = store.search_by_type("Person")
        assert len(people) == 2
        assert {p["entity_name"] for p in people} == {"Alice", "Bob"}

        hw = store.search_by_type("Hardware")
        assert len(hw) == 1
        assert hw[0]["entity_name"] == "ESP32"
    finally:
        os.unlink(path)


def test_get_note_entities():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.95, "note1.md")
        store.add("ProjectX", "Project", 0.9, "note1.md")
        store.add("Bob", "Person", 0.8, "note2.md")

        entities = store.get_note_entities("note1.md")
        assert len(entities) == 2
        names = {e["entity_name"] for e in entities}
        assert names == {"Alice", "ProjectX"}
    finally:
        os.unlink(path)


def test_deduplicate_same_chunk():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.95, "note1.md", chunk_idx=0)
        store.add("Alice", "Person", 0.8, "note1.md", chunk_idx=0)

        results = store.search("Alice")
        assert len(results) == 1
        assert results[0]["confidence"] == 0.95
    finally:
        os.unlink(path)


def test_persistence():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.95, "note1.md")
        store.save()

        store2 = entity_store.EntityStore(path=path)
        results = store2.search("Alice")
        assert len(results) == 1
        assert results[0]["entity_name"] == "Alice"
    finally:
        os.unlink(path)


def test_clear():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.95, "note1.md")
        assert len(store.search("Alice")) == 1
        store.clear()
        assert store.search("Alice") == []
    finally:
        os.unlink(path)


def test_invalid_type_defaults():
    store, path = _make_store()
    try:
        store.add("Foo", "InvalidType", 0.9, "note1.md")
        results = store.search("Foo")
        assert len(results) == 1
        assert results[0]["entity_type"] == "Concept"
    finally:
        os.unlink(path)


def test_aliases():
    store, path = _make_store()
    try:
        store.add("ESP32", "Hardware", 0.95, "note1.md")
        # "esp32" is casefold-equal to "ESP32" → merged into same record, added as alias
        store.add("esp32", "Hardware", 0.9, "note1.md")
        # "ESP-32" is a different normalized key → separate record
        store.add("ESP-32", "Hardware", 0.85, "note2.md")

        # Search should find only the ESP32 record (one path)
        results = store.search("ESP32")
        assert len(results) == 1
        assert results[0]["path"] == "note1.md"

        # Check aliases were recorded
        record = store._data["esp32"]
        assert record["canonical"] == "ESP32"

        # Check ESP-32 has its own record
        results2 = store.search("ESP-32")
        assert len(results2) == 1
        assert results2[0]["path"] == "note2.md"
    finally:
        os.unlink(path)


def test_stats():
    store, path = _make_store()
    try:
        assert store.stats()["total_entities"] == 0

        store.add("Alice", "Person", 0.95, "note1.md")
        store.add("Bob", "Person", 0.9, "note2.md")
        store.add("ESP32", "Hardware", 0.98, "note3.md")

        s = store.stats()
        assert s["total_entities"] == 3
        assert s["total_mentions"] == 3
        assert s["by_type"]["Person"] == 2
        assert s["by_type"]["Hardware"] == 1
    finally:
        os.unlink(path)


def test_case_insensitive_search():
    store, path = _make_store()
    try:
        store.add("ESP32", "Hardware", 0.95, "note1.md")

        results = store.search("esp32")
        assert len(results) == 1
        assert results[0]["entity_name"] == "ESP32"

        results = store.search("Esp32")
        assert len(results) == 1
    finally:
        os.unlink(path)


def test_add_with_aliases_param():
    store, path = _make_store()
    try:
        store.add("Maria", "Person", 0.95, "note1.md",
                  aliases=["Her", "The girl from church", "M"])
        result = store.search("Maria")
        assert len(result) == 1

        record = store._data["maria"]
        assert "Her" in record["aliases"]
        assert "The girl from church" in record["aliases"]
    finally:
        os.unlink(path)


def test_search_by_alias():
    store, path = _make_store()
    try:
        store.add("ESP32", "Hardware", 0.95, "note1.md",
                  aliases=["ESP-32", "esp32 chip"])
        # Searching by alias should find the canonical entity
        results = store.search("ESP-32")
        assert len(results) == 1
        assert results[0]["entity_name"] == "ESP32"
        assert results[0]["path"] == "note1.md"

        # Searching by another alias
        results = store.search("esp32 chip")
        assert len(results) == 1
        assert results[0]["entity_name"] == "ESP32"
    finally:
        os.unlink(path)


def test_get_aliases():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.95, "note1.md",
                  aliases=["Ali", "A. Johnson"])
        result = store.get_aliases("Alice")
        assert result is not None
        assert result["canonical"] == "Alice"
        assert result["type"] == "Person"
        assert "Ali" in result["aliases"]
        assert result["mention_count"] == 1

        # Lookup by alias should also work
        result2 = store.get_aliases("Ali")
        assert result2 is not None
        assert result2["canonical"] == "Alice"

        # Nonexistent entity
        assert store.get_aliases("Nobody") is None
    finally:
        os.unlink(path)


def test_merge_entities():
    store, path = _make_store()
    try:
        store.add("ESP32", "Hardware", 0.95, "note1.md")
        store.add("ESP-32", "Hardware", 0.85, "note2.md",
                  aliases=["esp32 module"])

        # Before merge: separate entities
        assert len(store.search("ESP32")) == 1
        assert len(store.search("ESP-32")) == 1

        result = store.merge("ESP32", "ESP-32")
        assert result is not None
        assert result["canonical"] == "ESP32"
        assert result["mention_count"] == 2
        assert "ESP-32" in result["aliases"]

        # After merge: both paths found via canonical name
        merged = store.search("ESP32")
        assert len(merged) == 2
        paths = {r["path"] for r in merged}
        assert paths == {"note1.md", "note2.md"}

        # Old secondary key should now redirect to primary via alias map
        redirected = store.search("ESP-32")
        assert len(redirected) == 2

        # Merging same entity should return None
        assert store.merge("ESP32", "ESP32") is None

        # Merging nonexistent entity should return None
        assert store.merge("ESP32", "Nonexistent") is None
    finally:
        os.unlink(path)


# ── Timeline ────────────────────────────────────────────────────────


def test_timeline_add_entry():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.9, "note1.md")
        store.add_timeline_entry("Alice", "2024-01", "Started ProjectX", note="note1.md")
        result = store.get_timeline("Alice")
        assert result is not None
        assert len(result) == 1
        assert result[0]["date"] == "2024-01"
        assert result[0]["event"] == "Started ProjectX"
    finally:
        os.unlink(path)


def test_timeline_multiple_entries_sorted():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.9, "note1.md")
        store.add_timeline_entry("Alice", "2024-06", "Became team lead", note="note2.md")
        store.add_timeline_entry("Alice", "2024-01", "Started ProjectX", note="note1.md")
        store.add_timeline_entry("Alice", "2024-03", "First prototype", note="note3.md")
        result = store.get_timeline("Alice")
        assert result is not None
        assert len(result) == 3
        assert result[0]["date"] == "2024-01"
        assert result[1]["date"] == "2024-03"
        assert result[2]["date"] == "2024-06"
    finally:
        os.unlink(path)


def test_timeline_nonexistent_entity():
    store, path = _make_store()
    try:
        result = store.get_timeline("Nonexistent")
        assert result is None
    finally:
        os.unlink(path)


def test_timeline_date_filter():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.9, "note1.md")
        store.add_timeline_entry("Alice", "2024-01", "Started", note="n1.md")
        store.add_timeline_entry("Alice", "2024-06", "Milestone", note="n2.md")
        store.add_timeline_entry("Alice", "2024-12", "Finished", note="n3.md")
        result = store.get_timeline("Alice", date_from="2024-03", date_to="2024-09")
        assert result is not None
        assert len(result) == 1
        assert result[0]["event"] == "Milestone"
    finally:
        os.unlink(path)


def test_timeline_search_by_alias():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.9, "note1.md", aliases=["Ali", "A"])
        store.add_timeline_entry("Alice", "2024-01", "Started", note="n1.md")
        result = store.get_timeline("Ali")
        assert result is not None
        assert len(result) == 1
    finally:
        os.unlink(path)


def test_timeline_deduplicate():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.9, "note1.md")
        store.add_timeline_entry("Alice", "2024-01", "Started", note="n1.md", confidence=0.8)
        store.add_timeline_entry("Alice", "2024-01", "Started", note="n1.md", confidence=0.9)
        result = store.get_timeline("Alice")
        assert result is not None
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9
    finally:
        os.unlink(path)


def test_timeline_empty_for_entity():
    store, path = _make_store()
    try:
        store.add("Alice", "Person", 0.9, "note1.md")
        result = store.get_timeline("Alice")
        assert result is not None
        assert result == []
    finally:
        os.unlink(path)
