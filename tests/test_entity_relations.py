"""Tests for entity_relations.py."""

import contextlib
import os

from obsidian_ai import entity_relations


def _make_store():
    """Create a temporary RelationshipStore for testing."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        fname = f.name
    store = entity_relations.RelationshipStore(path=fname)
    store.clear()
    return store, fname


def _cleanup(store, path):
    store.clear()
    with contextlib.suppress(OSError):
        os.unlink(path)


def test_add_and_get_direct():
    store, path = _make_store()
    try:
        store.add("Alice", "works_on", "ProjectX", confidence=0.95)
        store.add("Alice", "uses", "ESP32", confidence=0.9)
        store.save()

        results = store.get_related("Alice", depth=1)
        assert len(results) == 2
        names = {r["entity_name"] for r in results}
        assert names == {"projectx", "esp32"}
    finally:
        _cleanup(store, path)


def test_get_related_by_type():
    store, path = _make_store()
    try:
        store.add("Alice", "works_on", "ProjectX", confidence=0.95)
        store.add("Alice", "uses", "ESP32", confidence=0.9)
        store.save()

        results = store.get_related("Alice", relation_type="works_on")
        assert len(results) == 1
        assert results[0]["entity_name"] == "projectx"
        assert results[0]["relation_type"] == "works_on"
    finally:
        _cleanup(store, path)


def test_get_related_no_results():
    store, path = _make_store()
    try:
        results = store.get_related("Nobody")
        assert results == []
    finally:
        _cleanup(store, path)


def test_bfs_depth_2():
    store, path = _make_store()
    try:
        store.add("Alice", "works_on", "ProjectX", confidence=0.95)
        store.add("ProjectX", "uses", "ESP32", confidence=0.9)
        store.add("ESP32", "part_of", "IoTKit", confidence=0.8)

        results_depth1 = store.get_related("Alice", depth=1)
        assert len(results_depth1) == 1
        assert results_depth1[0]["entity_name"] == "projectx"

        results_depth2 = store.get_related("Alice", depth=2)
        assert len(results_depth2) == 2
        names = {r["entity_name"] for r in results_depth2}
        assert names == {"projectx", "esp32"}
    finally:
        _cleanup(store, path)


def test_bfs_depth_3():
    store, path = _make_store()
    try:
        store.add("Alice", "works_on", "ProjectX", confidence=0.95)
        store.add("ProjectX", "uses", "ESP32", confidence=0.9)
        store.add("ESP32", "part_of", "IoTKit", confidence=0.8)

        results = store.get_related("Alice", depth=3)
        assert len(results) == 3
        names = {r["entity_name"] for r in results}
        assert names == {"projectx", "esp32", "iotkit"}
    finally:
        _cleanup(store, path)


def test_deduplication():
    store, path = _make_store()
    try:
        store.add("Alice", "works_on", "ProjectX", confidence=0.8)
        store.add("Alice", "works_on", "ProjectX", confidence=0.95)
        assert len(store._relationships) == 1
        assert store._relationships[0]["confidence"] == 0.95
    finally:
        _cleanup(store, path)


def test_add_many():
    store, path = _make_store()
    try:
        count = store.add_many([
            {"source": "Alice", "type": "works_on", "target": "ProjectX", "confidence": 0.95},
            {"source": "Alice", "type": "uses", "target": "ESP32", "confidence": 0.9},
            {"source": "Bob", "type": "leads", "target": "ProjectX", "confidence": 0.8},
        ])
        assert count == 3

        results = store.get_related("Alice")
        assert len(results) == 2
    finally:
        _cleanup(store, path)


def test_add_many_invalid_skipped():
    store, path = _make_store()
    try:
        count = store.add_many([
            {"source": "Alice", "type": "works_on", "target": "ProjectX"},
            {"bad": "entry"},
            "not_a_dict",
        ])
        assert count == 1
    finally:
        _cleanup(store, path)


def test_invalid_relationship_type_falls_back():
    store, path = _make_store()
    try:
        store.add("Alice", "custom_type", "Bob", confidence=0.8)
        results = store.get_related("Alice")
        assert len(results) == 1
        assert results[0]["relation_type"] == "related_to"
    finally:
        _cleanup(store, path)


def test_confidence_clamping():
    store, path = _make_store()
    try:
        store.add("Alice", "works_on", "X", confidence=2.0)
        store.add("Bob", "works_on", "Y", confidence=-0.5)
        assert 0.0 <= store._relationships[0]["confidence"] <= 1.0
        assert 0.0 <= store._relationships[1]["confidence"] <= 1.0
    finally:
        _cleanup(store, path)


def test_save_and_reload():
    store, path = _make_store()
    try:
        store.add("Alice", "works_on", "ProjectX", confidence=0.95)
        store.save()

        store2 = entity_relations.RelationshipStore(path=path)
        results = store2.get_related("Alice")
        assert len(results) == 1
        assert results[0]["entity_name"] == "projectx"
        assert results[0]["relation_type"] == "works_on"
    finally:
        _cleanup(store, path)


def test_stats():
    store, path = _make_store()
    try:
        store.add("Alice", "works_on", "ProjectX", confidence=0.95)
        store.add("Bob", "works_on", "ProjectX", confidence=0.9)
        s = store.stats()
        assert s["total_relationships"] == 2
        assert s["unique_entities"] == 3
        assert s["by_type"]["works_on"] == 2
    finally:
        _cleanup(store, path)


def test_clear():
    store, path = _make_store()
    try:
        store.add("Alice", "works_on", "ProjectX", confidence=0.95)
        assert store.stats()["total_relationships"] == 1
        store.clear()
        assert store.stats()["total_relationships"] == 0
    finally:
        _cleanup(store, path)


def test_module_level_functions():
    rel_store = entity_relations._get_store()
    original_path = rel_store._path
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        test_path = f.name
    try:
        orig_clear = rel_store.clear
        rel_store._path = test_path
        rel_store.clear()
        entity_relations.add("Alice", "works_on", "ProjectX")
        entity_relations.add("Alice", "uses", "ESP32")
        results = entity_relations.get_related("Alice")
        assert len(results) == 2
        names = {r["entity_name"] for r in results}
        assert names == {"projectx", "esp32"}
        entity_relations.clear()
        assert entity_relations.stats()["total_relationships"] == 0
        rel_store._path = original_path
        rel_store.clear = orig_clear
    finally:
        with contextlib.suppress(OSError):
            os.unlink(test_path)
