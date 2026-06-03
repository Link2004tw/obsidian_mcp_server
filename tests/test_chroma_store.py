"""Tests for chroma_store.py — uses a temporary ChromaDB instance."""
import os

import pytest

from obsidian_ai import chroma_store


@pytest.fixture(autouse=True)
def _tmp_chroma(tmp_path):
    """Initialize chroma_store with a temporary directory for each test."""
    chroma_store.init(path=str(tmp_path))
    yield
    # Reset after test
    chroma_store._client = None
    chroma_store._collection = None


# ── upsert and query ───────────────────────────────────────────────


def test_upsert_and_query():
    embedding = [0.1] * 10
    chroma_store.upsert("note.md", 0, embedding, {"path": "note.md", "title": "Note", "chunk": 0})
    results = chroma_store.query(embedding, n=1)
    assert len(results) == 1
    assert results[0]["metadata"]["path"] == "note.md"


def test_query_with_where_filter():
    emb = [0.2] * 10
    chroma_store.upsert("a.md", 0, emb, {"path": "a.md", "title": "A", "chunk": 0, "tags": "python"})
    chroma_store.upsert("b.md", 0, emb, {"path": "b.md", "title": "B", "chunk": 0, "tags": "java"})
    results = chroma_store.query(emb, n=10, where={"tags": "python"})
    assert len(results) == 1
    assert results[0]["metadata"]["path"] == "a.md"


def test_query_returns_document():
    emb = [0.3] * 10
    chroma_store.upsert("doc.md", 0, emb, {"path": "doc.md", "title": "Doc", "chunk": 0}, document="Hello world")
    results = chroma_store.query(emb, n=1)
    assert results[0]["document"] == "Hello world"


# ── delete_by_path ─────────────────────────────────────────────────


def test_delete_by_path():
    emb = [0.4] * 10
    chroma_store.upsert("del.md", 0, emb, {"path": "del.md", "title": "Del", "chunk": 0})
    chroma_store.upsert("del.md", 1, emb, {"path": "del.md", "title": "Del", "chunk": 1})
    assert chroma_store.count() == 2
    chroma_store.delete_by_path("del.md")
    assert chroma_store.count() == 0


def test_delete_nonexistent():
    chroma_store.delete_by_path("nope.md")
    # Should not raise


# ── get_by_path ────────────────────────────────────────────────────


def test_get_by_path():
    emb = [0.5] * 10
    chroma_store.upsert("find.md", 0, emb, {"path": "find.md", "title": "Find", "chunk": 0})
    metas = chroma_store.get_by_path("find.md")
    assert len(metas) == 1
    assert metas[0]["path"] == "find.md"


def test_get_by_path_empty():
    assert chroma_store.get_by_path("missing.md") == []


# ── get_by_title ───────────────────────────────────────────────────


def test_get_by_title():
    emb = [0.6] * 10
    chroma_store.upsert("folder/title.md", 0, emb, {"path": "folder/title.md", "title": "Title", "chunk": 0})
    metas = chroma_store.get_by_title("Title")
    assert len(metas) == 1
    assert metas[0]["path"] == "folder/title.md"


def test_get_by_title_dedup():
    """Multiple chunks with same title should be deduplicated."""
    emb = [0.7] * 10
    chroma_store.upsert("a.md", 0, emb, {"path": "a.md", "title": "Same", "chunk": 0})
    chroma_store.upsert("a.md", 1, emb, {"path": "a.md", "title": "Same", "chunk": 1})
    metas = chroma_store.get_by_title("Same")
    assert len(metas) == 1


# ── count ──────────────────────────────────────────────────────────


def test_count_empty():
    assert chroma_store.count() == 0


def test_count_after_upsert():
    emb = [0.8] * 10
    chroma_store.upsert("a.md", 0, emb, {"path": "a.md", "title": "A", "chunk": 0})
    chroma_store.upsert("b.md", 0, emb, {"path": "b.md", "title": "B", "chunk": 0})
    assert chroma_store.count() == 2


# ── get_all_documents ──────────────────────────────────────────────


def test_get_all_documents():
    emb = [0.9] * 10
    chroma_store.upsert("x.md", 0, emb, {"path": "x.md", "title": "X", "chunk": 0}, document="content")
    ids, docs, metas = chroma_store.get_all_documents()
    assert len(ids) == 1
    assert docs[0] == "content"
    assert metas[0]["path"] == "x.md"


# ── get_index_stats ────────────────────────────────────────────────


def test_get_index_stats():
    emb = [1.0] * 10
    chroma_store.upsert("a.md", 0, emb, {"path": "a.md", "title": "A", "chunk": 0})
    chroma_store.upsert("a.md", 1, emb, {"path": "a.md", "title": "A", "chunk": 1})
    chroma_store.upsert("b.md", 0, emb, {"path": "b.md", "title": "B", "chunk": 0})
    stats = chroma_store.get_index_stats()
    assert stats["total_chunks"] == 3
    assert stats["unique_notes"] == 2


# ── _dedup_paths ───────────────────────────────────────────────────


def test_dedup_paths():
    results = [
        {"metadata": {"path": "a.md", "title": "A"}},
        {"metadata": {"path": "b.md", "title": "B"}},
        {"metadata": {"path": "a.md", "title": "A"}},
    ]
    deduped = chroma_store._dedup_paths(results)
    assert len(deduped) == 2
    paths = [p for p, _ in deduped]
    assert "a.md" in paths
    assert "b.md" in paths


# ── search_by_tags ─────────────────────────────────────────────────


def test_search_by_tags():
    emb = [0.5] * 10
    meta_a = {"path": "a.md", "title": "A", "chunk": 0, "tags_str": ",python,ai,"}
    meta_b = {"path": "b.md", "title": "B", "chunk": 0, "tags_str": ",java,spring,"}
    chroma_store.upsert("a.md", 0, emb, meta_a)
    chroma_store.upsert("b.md", 0, emb, meta_b)
    results = chroma_store.search_by_tags(["python"])
    assert len(results) == 1
    assert results[0]["path"] == "a.md"


def test_search_by_tags_multiple():
    emb = [0.5] * 10
    meta = {"path": "a.md", "title": "A", "chunk": 0, "tags_str": ",python,ai,"}
    chroma_store.upsert("a.md", 0, emb, meta)
    results = chroma_store.search_by_tags(["python", "ai"])
    assert len(results) == 1


def test_search_by_tags_no_match():
    emb = [0.5] * 10
    meta = {"path": "a.md", "title": "A", "chunk": 0, "tags_str": ",python,"}
    chroma_store.upsert("a.md", 0, emb, meta)
    results = chroma_store.search_by_tags(["java"])
    assert len(results) == 0


def test_search_by_tags_empty():
    assert chroma_store.search_by_tags([]) == []


# ── init reconfiguration ───────────────────────────────────────────


def test_init_reconfigures(tmp_path):
    """init() can be called to switch to a different directory."""
    path1 = str(tmp_path / "db1")
    path2 = str(tmp_path / "db2")
    os.makedirs(path1, exist_ok=True)
    os.makedirs(path2, exist_ok=True)

    chroma_store.init(path=path1)
    emb = [0.1] * 10
    chroma_store.upsert("a.md", 0, emb, {"path": "a.md", "title": "A", "chunk": 0})
    assert chroma_store.count() == 1

    chroma_store.init(path=path2)
    assert chroma_store.count() == 0  # fresh DB
