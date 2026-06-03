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
