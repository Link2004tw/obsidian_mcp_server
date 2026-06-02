"""Tests for graph_store.py — uses temporary JSON files."""
import json
import os
import tempfile

from obsidian_ai.graph_store import GraphStore


def _make_graph(path: str) -> GraphStore:
    """Create a GraphStore pointing to a temp JSON file."""
    return GraphStore(path=path)


# ── Persistence ─────────────────────────────────────────────────────


def test_save_and_load():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name

    try:
        g = _make_graph(tmp)
        g._adj = {"A.md": {"B.md"}, "B.md": {"C.md"}, "C.md": set()}
        g._title_map = {"a": "A.md", "b": "B.md", "c": "C.md"}
        g.save()

        g2 = _make_graph(tmp)
        assert g2._adj == {"A.md": {"B.md"}, "B.md": {"C.md"}, "C.md": set()}
        assert g2._title_map == {"a": "A.md", "b": "B.md", "c": "C.md"}
    finally:
        os.unlink(tmp)


def test_load_nonexistent():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    os.unlink(tmp)

    g = _make_graph(tmp)
    assert g._adj == {}
    assert g._title_map == {}


# ── Title → Path Resolution ─────────────────────────────────────────


def test_note_title():
    assert GraphStore._note_title("Folder/My Note.md") == "my note"
    assert GraphStore._note_title("Simple.md") == "simple"
    assert GraphStore._note_title("DEEP/nested/path/Title Here.md") == "title here"


def test_build_title_map():
    notes = {
        "A/Note One.md": "content",
        "B/Note Two.md": "content",
        "A/Note One (copy).md": "content",  # different title
    }
    g = _make_graph("/dev/null")
    g._build_title_map(notes)
    assert g._title_map["note one"] == "A/Note One.md"
    assert g._title_map["note two"] == "B/Note Two.md"
    assert g._title_map["note one (copy)"] == "A/Note One (copy).md"


def test_build_title_map_duplicate_titles():
    """First path wins for duplicate titles."""
    notes = {
        "B/Duplicate.md": "content",
        "A/Duplicate.md": "content",  # would be first alphabetically
    }
    g = _make_graph("/dev/null")
    g._build_title_map(notes)
    # sorted keys: A/Duplicate.md comes first
    assert g._title_map["duplicate"] == "A/Duplicate.md"


# ── Rebuild ─────────────────────────────────────────────────────────


def test_rebuild_basic():
    notes = {
        "Note A.md": "See [[Note B]] and [[Note C]].",
        "Note B.md": "References [[Note C]].",
        "Note C.md": "No links here.",
    }
    g = _make_graph("/dev/null")
    g.rebuild(notes)

    assert "Note A.md" in g._adj
    assert "Note B.md" in g._adj["Note A.md"]
    assert "Note C.md" in g._adj["Note A.md"]
    assert "Note C.md" in g._adj["Note B.md"]
    assert g._adj["Note C.md"] == set()


def test_rebuild_broken_links_ignored():
    """Links to non-existent notes are not added as edges."""
    notes = {
        "Note A.md": "See [[Note B]] and [[Nonexistent]].",
        "Note B.md": "Content.",
    }
    g = _make_graph("/dev/null")
    g.rebuild(notes)

    assert "Note B.md" in g._adj["Note A.md"]
    # Nonexistent is not in adj as a node (it's not resolved)
    assert "Nonexistent" not in g._adj


def test_rebuild_self_links_ignored():
    notes = {
        "Note A.md": "See [[Note A]].",
    }
    g = _make_graph("/dev/null")
    g.rebuild(notes)
    assert g._adj["Note A.md"] == set()


# ── Edge Operations ─────────────────────────────────────────────────


def test_add_edge():
    g = _make_graph("/dev/null")
    g.add_edge("A.md", "B.md")
    assert "B.md" in g._adj["A.md"]
    assert "B.md" in g._adj  # target node exists


def test_remove_node():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md"}, "B.md": {"C.md"}, "C.md": set()}
    g.remove_node("B.md")
    assert "B.md" not in g._adj
    assert "B.md" not in g._adj["A.md"]


def test_rename_node():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md"}, "B.md": {"C.md"}, "C.md": set()}
    g._title_map = {"a": "A.md", "b": "B.md", "c": "C.md"}
    g.rename_node("B.md", "B Renamed.md")

    assert "B.md" not in g._adj
    assert "B Renamed.md" in g._adj
    assert "C.md" in g._adj["B Renamed.md"]
    # A.md should now point to B Renamed.md
    assert "B Renamed.md" in g._adj["A.md"]
    assert g._title_map["b renamed"] == "B Renamed.md"
    assert "b" not in g._title_map  # old title key removed


def test_register_title():
    g = _make_graph("/dev/null")
    g.register_title("New/Note.md")
    assert g._title_map["note"] == "New/Note.md"


# ── Queries ─────────────────────────────────────────────────────────


def test_get_backlinks():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"C.md"}, "B.md": {"C.md"}, "C.md": set()}
    assert g.get_backlinks("C.md") == ["A.md", "B.md"]
    assert g.get_backlinks("A.md") == []


def test_get_outgoing():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md", "C.md"}, "B.md": set(), "C.md": set()}
    assert g.get_outgoing("A.md") == ["B.md", "C.md"]
    assert g.get_outgoing("B.md") == []


# ── BFS ─────────────────────────────────────────────────────────────


def test_bfs_depth_1():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md", "C.md"}, "B.md": {"D.md"}, "C.md": set(), "D.md": set()}
    results = g.bfs("A.md", max_depth=1)
    assert "B.md" in results
    assert "C.md" in results
    assert results["B.md"] == ["A.md", "B.md"]
    assert "D.md" not in results  # depth 2


def test_bfs_depth_2():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md"}, "B.md": {"C.md"}, "C.md": {"D.md"}, "D.md": set()}
    results = g.bfs("A.md", max_depth=2)
    assert "B.md" in results
    assert "C.md" in results
    assert "D.md" not in results  # depth 3


def test_bfs_start_not_in_results():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md"}, "B.md": set()}
    results = g.bfs("A.md", max_depth=1)
    assert "A.md" not in results


def test_bfs_nonexistent_start():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md"}, "B.md": set()}
    assert g.bfs("Z.md", max_depth=1) == {}


# ── Stats ───────────────────────────────────────────────────────────


def test_stats_empty():
    g = _make_graph("/dev/null")
    stats = g.stats()
    assert stats["nodes"] == 0
    assert stats["edges"] == 0
    assert stats["avg_degree"] == 0


def test_stats_basic():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md", "C.md"}, "B.md": {"C.md"}, "C.md": set()}
    stats = g.stats()
    assert stats["nodes"] == 3
    assert stats["edges"] == 3  # A->B, A->C, B->C
    assert stats["avg_degree"] > 0


def test_stats_isolated():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md"}, "B.md": set(), "C.md": set()}
    stats = g.stats()
    assert "C.md" in stats["isolated"]


def test_stats_hubs():
    g = _make_graph("/dev/null")
    g._adj = {
        "A.md": {"B.md", "C.md", "D.md", "E.md"},
        "B.md": set(),
        "C.md": set(),
        "D.md": set(),
        "E.md": set(),
    }
    stats = g.stats()
    assert stats["hubs"][0]["path"] == "A.md"


# ── Broken Links ────────────────────────────────────────────────────


def test_get_broken_links():
    notes = {
        "A.md": "See [[Note B]] and [[Missing]].",
        "Note B.md": "Content.",
    }
    g = _make_graph("/dev/null")
    broken = g.get_broken_links(notes)
    assert len(broken) == 1
    assert broken[0]["source_path"] == "A.md"
    assert broken[0]["link_target"] == "missing"


def test_get_broken_links_none():
    notes = {
        "A.md": "See [[Note B]].",
        "Note B.md": "Content.",
    }
    g = _make_graph("/dev/null")
    assert g.get_broken_links(notes) == []


# ── Orphans ─────────────────────────────────────────────────────────


def test_get_orphans():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md"}, "B.md": set(), "C.md": set()}
    orphans = g.get_orphans()
    assert "C.md" in orphans
    assert "A.md" not in orphans  # has outgoing
    assert "B.md" not in orphans  # has incoming


# ── Export/Import ───────────────────────────────────────────────────


def test_to_dict_and_from_dict():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md"}, "B.md": set()}
    g._title_map = {"a": "A.md", "b": "B.md"}

    data = g.to_dict()
    assert data["edges"]["A.md"] == ["B.md"]
    assert data["title_map"]["a"] == "A.md"

    g2 = _make_graph("/dev/null")
    g2.from_dict(data)
    assert g2._adj == g._adj
    assert g2._title_map == g._title_map


# ── Entity Nodes ─────────────────────────────────────────────────────


def test_entity_node_id():
    eid = GraphStore._entity_node_id("Person", "Alice")
    assert eid == "__entity:Person:Alice"


def test_add_entity_edge():
    g = _make_graph("/dev/null")
    g.add_entity_edge("Person", "Alice", "note1.md")
    g.add_entity_edge("Person", "Alice", "note2.md")

    assert g.is_entity_node("__entity:Person:Alice")
    assert not g.is_entity_node("note1.md")

    notes = g.get_entity_notes("Person", "Alice")
    assert notes == ["note1.md", "note2.md"]


def test_remove_entity_edges():
    g = _make_graph("/dev/null")
    g.add_entity_edge("Person", "Alice", "note1.md")
    assert g.is_entity_node("__entity:Person:Alice")
    assert g.get_entity_notes("Person", "Alice") == ["note1.md"]

    g.remove_entity_edges("Person", "Alice")
    # Node is removed from _adj entirely
    assert "__entity:Person:Alice" not in g._adj
    assert g.get_entity_notes("Person", "Alice") == []


def test_get_entity_nodes():
    g = _make_graph("/dev/null")
    g.add_entity_edge("Person", "Alice", "note1.md")
    g.add_entity_edge("Hardware", "ESP32", "note2.md")

    nodes = g.get_entity_nodes()
    assert len(nodes) == 2
    names = {n["entity_name"] for n in nodes}
    types = {n["entity_type"] for n in nodes}
    assert names == {"Alice", "ESP32"}
    assert types == {"Person", "Hardware"}


def test_entity_nodes_excluded_from_stats():
    g = _make_graph("/dev/null")
    g.add_edge("A.md", "B.md")
    g.add_entity_edge("Person", "Alice", "A.md")

    s = g.stats()
    assert s["nodes"] == 2  # A.md, B.md — entity node excluded
    assert s["edges"] == 1


def test_entity_nodes_excluded_from_orphans():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": set(), "__entity:Person:Alice": {"A.md"}}

    orphans = g.get_orphans()
    assert "A.md" in orphans
    assert "__entity:Person:Alice" not in orphans


def test_entity_nodes_excluded_from_to_dict():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"__entity:Person:Alice"}, "__entity:Person:Alice": {"A.md"}}
    g._title_map = {"a": "A.md"}

    data = g.to_dict()
    assert "A.md" in data["edges"]
    assert "__entity:Person:Alice" not in data["edges"]
    # A.md's edge to entity should also be filtered
    assert data["edges"]["A.md"] == []


def test_non_entity_nodes():
    g = _make_graph("/dev/null")
    g._adj = {"A.md": {"B.md"}, "__entity:Person:Alice": {"A.md"}}

    filtered = g._non_entity_nodes()
    assert "A.md" in filtered
    assert "__entity:Person:Alice" not in filtered
