"""Tests for keyword_search.py — BM25 tokenization and search logic."""
from unittest.mock import patch

from obsidian_ai import keyword_search

# ── _tokenize ──────────────────────────────────────────────────────


def test_tokenize_basic():
    tokens = keyword_search._tokenize("Hello World")
    assert tokens == ["hello", "world"]


def test_tokenize_removes_short():
    tokens = keyword_search._tokenize("I am a test")
    # "I" and "am" are <= 1 char, "a" is <= 1 char
    assert "i" not in tokens
    assert "a" not in tokens
    assert "test" in tokens


def test_tokenize_alphanumeric():
    tokens = keyword_search._tokenize("v2.0 release notes")
    assert "v2" in tokens
    assert "0" not in tokens  # single char filtered
    assert "release" in tokens
    assert "notes" in tokens


def test_tokenize_empty():
    assert keyword_search._tokenize("") == []


def test_tokenize_special_chars():
    tokens = keyword_search._tokenize("hello-world foo_bar")
    assert "hello" in tokens
    assert "world" in tokens
    assert "foo" in tokens
    assert "bar" in tokens


# ── normalise_scores ───────────────────────────────────────────────


def test_normalise_scores_basic():
    results = [
        {"bm25_score": 10.0},
        {"bm25_score": 5.0},
        {"bm25_score": 0.0},
    ]
    keyword_search.normalise_scores(results)
    assert results[0]["bm25_score"] == 1.0  # max
    assert results[1]["bm25_score"] == 0.5  # 5/10
    assert results[2]["bm25_score"] == 0.0  # 0


def test_normalise_scores_all_same():
    results = [{"bm25_score": 3.0}, {"bm25_score": 3.0}]
    keyword_search.normalise_scores(results)
    # All same → mx == mn, implementation sets all to 1.0
    assert all(r["bm25_score"] == 1.0 for r in results)


def test_normalise_scores_empty():
    keyword_search.normalise_scores([])
    # Should not raise


def test_normalise_scores_single():
    results = [{"bm25_score": 7.0}]
    keyword_search.normalise_scores(results)
    # Single item → mx == mn → set to 1.0
    assert results[0]["bm25_score"] == 1.0


# ── search with mocked ChromaDB ────────────────────────────────────


@patch("obsidian_ai.keyword_search.chroma_store")
def test_search_empty_corpus(mock_chroma):
    """search returns empty list when BM25 index is empty."""
    mock_chroma.count.return_value = 0
    mock_chroma.get_all_documents.return_value = ([], [], [])
    keyword_search._bm25 = None
    keyword_search._bm25_corpus_ids = None
    keyword_search._bm25_doc_count = 0
    result = keyword_search.search("test query")
    assert result == []


@patch("obsidian_ai.keyword_search.chroma_store")
def test_search_with_results(mock_chroma):
    """search returns scored results from BM25 index."""
    mock_chroma.count.return_value = 2
    mock_chroma.get_all_documents.return_value = (
        ["doc1::chunk_0", "doc2::chunk_0"],
        ["python programming tutorial", "java programming guide"],
        [{"path": "note1.md", "title": "Note 1", "chunk": 0},
         {"path": "note2.md", "title": "Note 2", "chunk": 0}],
    )
    mock_chroma.get_metadata_by_ids.return_value = (
        [{"path": "note1.md", "title": "Note 1", "chunk": 0},
         {"path": "note2.md", "title": "Note 2", "chunk": 0}],
        ["python programming tutorial", "java programming guide"],
    )
    # Force rebuild
    keyword_search._bm25 = None
    keyword_search._bm25_doc_count = 0

    results = keyword_search.search("python tutorial", n=10)
    assert len(results) > 0
    # All results should have bm25_score
    for r in results:
        assert "bm25_score" in r
