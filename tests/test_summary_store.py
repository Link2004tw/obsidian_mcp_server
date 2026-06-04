"""Tests for summary_store.py — uses a temporary ChromaDB instance."""
from unittest.mock import patch

import pytest

from obsidian_ai import summary_store


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    """Point config.chroma_path to a temporary dir so ChromaDB uses it."""
    import obsidian_ai.config as cfg
    monkeypatch.setattr(cfg, "chroma_path", str(tmp_path))
    summary_store._client = None
    summary_store._collection = None
    yield
    summary_store._client = None
    summary_store._collection = None


@patch("obsidian_ai.summary_store.llm_client")
def test_add_and_count(mock_llm):
    mock_llm.embed.return_value = [0.1] * 768
    summary_store.add("note1.md", "Note1", "Summary of note 1")
    summary_store.add("note2.md", "Note2", "Summary of note 2")
    assert summary_store.count() == 2


@patch("obsidian_ai.summary_store.llm_client")
def test_add_empty_summary_skipped(mock_llm):
    summary_store.add("note1.md", "Note1", "")
    assert summary_store.count() == 0
    mock_llm.embed.assert_not_called()


@patch("obsidian_ai.summary_store.llm_client")
def test_upsert_replaces(mock_llm):
    mock_llm.embed.return_value = [0.1] * 768
    summary_store.add("note1.md", "Note1", "First summary")
    summary_store.add("note1.md", "Note1", "Updated summary")
    assert summary_store.count() == 1


@patch("obsidian_ai.summary_store.llm_client")
def test_delete_by_path(mock_llm):
    mock_llm.embed.return_value = [0.1] * 768
    summary_store.add("note1.md", "Note1", "Summary 1")
    summary_store.add("note2.md", "Note2", "Summary 2")
    assert summary_store.count() == 2
    summary_store.delete_by_path("note1.md")
    assert summary_store.count() == 1


@patch("obsidian_ai.summary_store.llm_client")
def test_clear(mock_llm):
    mock_llm.embed.return_value = [0.1] * 768
    summary_store.add("note1.md", "Note1", "Summary 1")
    summary_store.clear()
    assert summary_store.count() == 0


@patch("obsidian_ai.summary_store.llm_client")
def test_query_returns_results(mock_llm):
    # embed returns a fixed vector; all summaries get the same vector,
    # so distance is ~0 for all and they all come back
    mock_llm.embed.return_value = [0.1] * 768
    summary_store.add("note1.md", "Note1", "Summary about Alice")
    summary_store.add("note2.md", "Note2", "Summary about ProjectX")

    results = summary_store.query("Alice", n=5)
    assert len(results) == 2
    paths = {r["path"] for r in results}
    assert paths == {"note1.md", "note2.md"}
    assert all("similarity" in r for r in results)
    assert all("summary" in r for r in results)


@patch("obsidian_ai.summary_store.llm_client")
def test_query_empty_store(mock_llm):
    mock_llm.embed.return_value = [0.1] * 768
    results = summary_store.query("nothing", n=5)
    assert results == []


@patch("obsidian_ai.summary_store.llm_client")
def test_query_sorts_by_similarity_desc(mock_llm):
    """Results should be sorted by similarity descending."""
    mock_llm.embed.return_value = [0.1] * 768
    summary_store.add("note1.md", "Note1", "Some summary")
    summary_store.add("note2.md", "Note2", "Another summary")

    results = summary_store.query("test", n=5)
    assert len(results) >= 2
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)
