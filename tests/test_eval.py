"""Tests for eval.py — retrieval evaluation benchmark."""
import json
import os
import tempfile

from obsidian_ai.eval import (
    _extract_titles,
    _mrr,
    _precision_at_k,
    _recall_at_k,
    format_results,
    load_benchmark,
    run_eval,
)


def _make_benchmark(queries: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(queries, f)
        return f.name


def test_extract_titles():
    paths = ["Folder/Note.md", "Root.md", "Deep/Nested/File.md"]
    titles = _extract_titles(paths)
    assert "note" in titles
    assert "root" in titles
    assert "file" in titles


def test_precision_at_k_perfect():
    retrieved = ["A.md", "B.md", "C.md"]
    expected = {"a", "b"}
    assert _precision_at_k(retrieved, expected, 2) == 1.0


def test_precision_at_k_partial():
    retrieved = ["A.md", "B.md", "C.md"]
    expected = {"a"}
    assert _precision_at_k(retrieved, expected, 3) == 1.0 / 3.0


def test_precision_at_k_empty():
    retrieved = []
    expected = {"a"}
    assert _precision_at_k(retrieved, expected, 5) == 0.0


def test_recall_at_k_perfect():
    retrieved = ["A.md", "B.md", "C.md"]
    expected = {"a", "b"}
    assert _recall_at_k(retrieved, expected, 2) == 1.0


def test_recall_at_k_partial():
    retrieved = ["A.md", "B.md"]
    expected = {"a", "b", "c"}
    assert round(_recall_at_k(retrieved, expected, 2), 4) == round(2.0 / 3.0, 4)


def test_recall_at_k_empty_expected():
    assert _recall_at_k(["A.md"], set(), 5) == 0.0


def test_mrr_first():
    retrieved = ["X.md", "A.md", "B.md"]
    expected = {"a"}
    assert _mrr(retrieved, expected) == 0.5  # 1/2 (A is at rank 2)


def test_mrr_not_found():
    retrieved = ["X.md", "Y.md"]
    expected = {"a"}
    assert _mrr(retrieved, expected) == 0.0


def test_mrr_first_rank():
    retrieved = ["A.md", "B.md", "C.md"]
    expected = {"a"}
    assert _mrr(retrieved, expected) == 1.0


def test_load_benchmark():
    path = _make_benchmark([
        {"query": "test", "expected": ["A"], "description": "desc"},
    ])
    try:
        queries = load_benchmark(path)
        assert len(queries) == 1
        assert queries[0]["query"] == "test"
    finally:
        os.unlink(path)


def test_load_benchmark_missing():
    assert load_benchmark("/nonexistent/path.json") == []


def test_run_eval_empty():
    result = run_eval(queries=[])
    assert result["total_queries"] == 0
    assert result["precision_at_k"] == 0.0


def test_run_eval_basic():
    queries = [
        {"query": "test", "expected": ["A"], "description": ""},
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name

    try:
        # We can't easily mock ranker.search from here, but we can
        # verify the eval runs and returns expected structure
        # with no results (since there's no vault)
        result = run_eval(queries=queries, top_k=5)
        assert "precision_at_k" in result
        assert "recall_at_k" in result
        assert "mrr" in result
        assert "per_query" in result
        assert result["total_queries"] == 1
    finally:
        os.unlink(tmp)


def test_format_results():
    results = {
        "precision_at_k": 0.5,
        "recall_at_k": 0.3,
        "mrr": 0.4,
        "total_queries": 2,
        "per_query": [
            {"query": "q1", "description": "d1", "expected": ["A"], "retrieved": ["A.md"],
             "precision_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0},
        ],
    }
    text = format_results(results)
    assert "Total queries" in text
    assert "Precision" in text
    assert "Recall" in text
    assert "MRR" in text
