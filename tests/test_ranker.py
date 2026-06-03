"""Tests for ranker.py — unified ranking pipeline."""
from unittest.mock import MagicMock, patch

from obsidian_ai.ranker import Ranker, _expand_entity_names, _truncate_snippet


def test_default_weights():
    r = Ranker()
    w = r.weights
    assert abs(w["semantic"] - 0.40) < 0.01
    assert abs(w["entity"] - 0.30) < 0.01
    assert abs(w["graph"] - 0.20) < 0.01
    assert abs(w["keyword"] - 0.10) < 0.01


def test_set_weights():
    r = Ranker()
    r.set_weights(semantic=0.5, keyword=0.5)
    w = r.weights
    assert abs(w["semantic"] - 0.50) < 0.01
    assert abs(w["keyword"] - 0.50) < 0.01
    # Unchanged
    assert abs(w["entity"] - 0.30) < 0.01
    assert abs(w["graph"] - 0.20) < 0.01


def test_set_weights_partial():
    r = Ranker()
    r.set_weights(entity=0.0)
    assert r.weights["entity"] == 0.0


def test_weights_immutable():
    r = Ranker()
    w = r.weights
    w["semantic"] = 999.0
    assert r.weights["semantic"] == 0.40


def test_truncate_snippet_short():
    assert _truncate_snippet("hello", 10) == "hello"


def test_truncate_snippet_long():
    text = "a" * 300
    result = _truncate_snippet(text, 10)
    assert result == "a" * 10 + "..."
    assert len(result) == 13


def test_truncate_snippet_default_max():
    text = "x" * 400
    result = _truncate_snippet(text)
    assert len(result) == 303
    assert result.endswith("...")


def test_normalize_removes_zero_weights():
    r = Ranker({"semantic": 0.0, "entity": 1.0, "graph": 0.0, "keyword": 0.0})
    n = r._normalize()
    assert "semantic" not in n
    assert "graph" not in n
    assert "keyword" not in n
    assert abs(n["entity"] - 1.0) < 0.01


def test_normalize_sums_to_one():
    r = Ranker()
    n = r._normalize()
    assert abs(sum(n.values()) - 1.0) < 0.01


def test_normalize_with_override():
    r = Ranker()
    n = r._normalize({"semantic": 0.6, "keyword": 0.4})
    assert abs(sum(n.values()) - 1.0) < 0.01
    assert n["semantic"] > n["keyword"]


def test_empty_weights_fallback():
    r = Ranker({"semantic": 0.0, "entity": 0.0, "graph": 0.0, "keyword": 0.0})
    n = r._normalize()
    assert n == {"semantic": 1.0}


def test_search_empty_results():
    r = Ranker({"semantic": 1.0, "entity": 0.0, "graph": 0.0, "keyword": 0.0})
    mock_embed = MagicMock(return_value=[0.1] * 768)
    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma:
        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = []
        result = r.search("no matching query")
    assert result == []


def test_search_semantic_only():
    r = Ranker({"semantic": 1.0, "entity": 0.0, "graph": 0.0, "keyword": 0.0})

    mock_embed = MagicMock(return_value=[0.1] * 768)
    mock_chroma_results = [
        {
            "id": "note1::chunk_0",
            "metadata": {"path": "test/note1.md", "title": "Note1", "chunk": 0},
            "distance": 0.5,
            "document": "This is the content of note1.",
        },
        {
            "id": "note2::chunk_0",
            "metadata": {"path": "test/note2.md", "title": "Note2", "chunk": 0},
            "distance": 0.8,
            "document": "This is the content of note2.",
        },
    ]

    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma:
        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = mock_chroma_results

        results = r.search("test query", n=5)

    assert len(results) == 2
    assert results[0]["path"] == "test/note1.md"
    assert results[1]["path"] == "test/note2.md"
    assert results[0]["similarity_score"] > results[1]["similarity_score"]
    assert "semantic" in results[0]["matched_by"]
    assert len(results[0]["matched_by"]) == 1


def test_search_with_exclude_tags():
    r = Ranker({"semantic": 1.0, "entity": 0.0, "graph": 0.0, "keyword": 0.0})
    mock_embed = MagicMock(return_value=[0.1] * 768)
    mock_chroma_results = [
        {
            "id": "note1::chunk_0",
            "metadata": {"path": "test/note1.md", "title": "Note1", "chunk": 0, "tags_str": ",important,"},
            "distance": 0.5,
            "document": "Content.",
        },
    ]

    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma:
        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = mock_chroma_results

        results = r.search("test", exclude_tags=["important"])

    assert len(results) == 0


# ── Entity expansion tests ────────────────────────────────────────────


def test_expand_entity_names_empty_input():
    assert _expand_entity_names([]) == []


def test_expand_entity_names_no_relations():
    with patch("obsidian_ai.entity_relations") as mock_er:
        mock_er.get_related.return_value = []
        results = _expand_entity_names([
            {"entity_name": "Alice", "path": "note1.md", "entity_type": "Person", "confidence": 0.95},
        ])
    assert results == []


def test_expand_entity_names_with_relations():
    with patch("obsidian_ai.entity_relations") as mock_er:
        mock_er.get_related.return_value = [
            {"entity_name": "projectx", "relation_type": "works_on", "confidence": 0.95, "depth": 1},
            {"entity_name": "esp32", "relation_type": "uses", "confidence": 0.9, "depth": 1},
        ]
        results = _expand_entity_names([
            {"entity_name": "Alice", "path": "note1.md", "entity_type": "Person", "confidence": 0.95},
        ])
    assert len(results) == 2
    assert "projectx" in results
    assert "esp32" in results


def test_expand_entity_names_deduplicates():
    """Related entities that are already in the original set should be excluded."""
    with patch("obsidian_ai.entity_relations") as mock_er:
        mock_er.get_related.return_value = [
            {"entity_name": "projectx", "relation_type": "works_on", "confidence": 0.95, "depth": 1},
        ]
        results = _expand_entity_names([
            {"entity_name": "Alice", "path": "note1.md", "entity_type": "Person", "confidence": 0.95},
            {"entity_name": "ProjectX", "path": "note2.md", "entity_type": "Project", "confidence": 0.9},
        ])
    assert results == []


def test_expand_entity_names_case_insensitive_dedup():
    with patch("obsidian_ai.entity_relations") as mock_er:
        mock_er.get_related.return_value = [
            {"entity_name": "PROJECTX", "relation_type": "works_on", "confidence": 0.95, "depth": 1},
        ]
        results = _expand_entity_names([
            {"entity_name": "Alice", "path": "note1.md", "entity_type": "Person", "confidence": 0.95},
            {"entity_name": "ProjectX", "path": "note2.md", "entity_type": "Project", "confidence": 0.9},
        ])
    assert results == []


def test_search_with_expand_entities_uses_entity_store():
    """When expand_entities=True and entities are auto-detected, the ranker
    should call entity_store.search for expanded entity names."""
    r = Ranker({"semantic": 0.0, "entity": 1.0, "graph": 0.0, "keyword": 0.0})
    mock_embed = MagicMock(return_value=[0.1] * 768)
    alice_result = [{"path": "note_alice.md", "entity_name": "Alice", "entity_type": "Person", "confidence": 0.95}]
    projectx_result = [{"path": "note_projectx.md", "entity_name": "ProjectX", "entity_type": "Project", "confidence": 0.9}]

    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma, \
         patch("obsidian_ai.entity_store") as mock_es, \
         patch("obsidian_ai.entity_relations") as mock_er:

        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = []

        # _auto_detect_entities calls entity_store.search("Alice") twice
        # (full query + word match), both return alice_result
        mock_es.search.side_effect = [
            alice_result,  # full query "Alice"
            alice_result,  # word "Alice" (>2 chars)
            projectx_result,  # expansion: entity_store.search("projectx")
        ]
        mock_er.get_related.return_value = [
            {"entity_name": "projectx", "relation_type": "works_on", "confidence": 0.95, "depth": 1},
        ]

        results = r.search("Alice", n=5, expand_entities=True)

    assert len(results) == 2
    paths = {r["path"] for r in results}
    assert "note_alice.md" in paths
    assert "note_projectx.md" in paths


def test_search_without_expand_entities_does_not_expand():
    """When expand_entities=False (default), no expansion should occur."""
    r = Ranker({"semantic": 0.0, "entity": 1.0, "graph": 0.0, "keyword": 0.0})
    mock_embed = MagicMock(return_value=[0.1] * 768)
    alice_result = [{"path": "note_alice.md", "entity_name": "Alice", "entity_type": "Person", "confidence": 0.95}]

    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma, \
         patch("obsidian_ai.entity_store") as mock_es, \
         patch("obsidian_ai.entity_relations") as mock_er:

        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = []

        mock_es.search.side_effect = [
            alice_result,  # full query "Alice"
            alice_result,  # word "Alice"
        ]

        results = r.search("Alice", n=5, expand_entities=False)

    assert len(results) == 1
    assert results[0]["path"] == "note_alice.md"
    # entity_relations should NOT be called when expand_entities is False
    mock_er.get_related.assert_not_called()
