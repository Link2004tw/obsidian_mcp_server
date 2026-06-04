"""Tests for ranker.py — unified ranking pipeline."""
from unittest.mock import MagicMock, patch

from obsidian_ai.ranker import (
    INTENT_WEIGHTS,
    Ranker,
    _expand_entity_names,
    _truncate_snippet,
    detect_intent,
)


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


# ── Summary-first retrieval tests ────────────────────────────────────


def test_search_with_summaries_enabled():
    r = Ranker({"semantic": 0.5, "entity": 0.0, "graph": 0.0, "keyword": 0.0})
    mock_embed = MagicMock(return_value=[0.1] * 768)

    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma, \
         patch("obsidian_ai.summary_store") as mock_ss:

        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = []
        mock_ss.query.return_value = [
            {"path": "note1.md", "title": "Note1", "similarity": 0.85, "summary": "About AI."},
        ]

        results = r.search("AI", n=5, use_summaries=True, summary_threshold=0.5)

    assert len(results) == 1
    assert results[0]["path"] == "note1.md"
    assert "summary" in results[0]["matched_by"]


def test_search_with_summaries_below_threshold():
    """Summary results below threshold should not be included."""
    r = Ranker({"semantic": 0.5, "entity": 0.0, "graph": 0.0, "keyword": 0.0})
    mock_embed = MagicMock(return_value=[0.1] * 768)

    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma, \
         patch("obsidian_ai.summary_store") as mock_ss:

        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = []
        mock_ss.query.return_value = [
            {"path": "note1.md", "title": "Note1", "similarity": 0.3, "summary": "Weak match."},
        ]

        results = r.search("AI", n=5, use_summaries=True, summary_threshold=0.7)

    assert len(results) == 0


def test_search_with_summaries_disabled():
    """When use_summaries is False, summary_store should not be queried."""
    r = Ranker({"semantic": 0.5, "entity": 0.0, "graph": 0.0, "keyword": 0.0})
    mock_embed = MagicMock(return_value=[0.1] * 768)

    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma, \
         patch("obsidian_ai.summary_store") as mock_ss:

        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = []

        r.search("AI", n=5, use_summaries=False)

    mock_ss.query.assert_not_called()


# ── Composite Search ─────────────────────────────────────────────────


def test_composite_search_depth1_summary_only():
    """Depth=1 should only use summary search."""
    with patch("obsidian_ai.summary_store") as mock_ss, \
         patch("obsidian_ai.entity_store") as mock_es, \
         patch("obsidian_ai.graph_store") as mock_gs:

        mock_ss.query.return_value = [
            {"path": "A.md", "title": "A", "similarity": 0.85, "summary": "Note A"},
            {"path": "B.md", "title": "B", "similarity": 0.72, "summary": "Note B"},
        ]

        from obsidian_ai.ranker import composite_search
        results = composite_search("test", n=5, retrieval_depth=1)

    assert len(results) == 2
    assert results[0]["path"] == "A.md"
    assert results[1]["path"] == "B.md"
    assert all("summary" in r["matched_by"] for r in results)
    mock_es.search.assert_not_called()
    mock_gs.community_neighbors.assert_not_called()


def test_composite_search_depth2_includes_entity():
    """Depth=2 should include entity expansion."""
    with patch("obsidian_ai.summary_store") as mock_ss, \
         patch("obsidian_ai.entity_store") as mock_es, \
         patch("obsidian_ai.entity_relations") as mock_er, \
         patch("obsidian_ai.graph_store") as mock_gs:

        mock_ss.query.return_value = [
            {"path": "A.md", "title": "A", "similarity": 0.85, "summary": "Note A"},
        ]

        mock_es.search.side_effect = [
            [{"path": "B.md", "entity_name": "Alice", "confidence": 0.9}],  # auto-detect full query
            [{"path": "B.md", "entity_name": "Alice", "confidence": 0.9}],  # auto-detect word
            [],  # auto-detect bigram
        ]
        mock_er.get_related.return_value = []  # no relationship expansion

        from obsidian_ai.ranker import composite_search
        results = composite_search("Alice", n=5, retrieval_depth=2)

    assert len(results) >= 2  # A.md from summary, B.md from entity
    paths = {r["path"] for r in results}
    assert "A.md" in paths
    assert "B.md" in paths
    # graph store should NOT be called at depth=2
    mock_gs.community_neighbors.assert_not_called()


def test_composite_search_depth3_includes_community():
    """Depth=3 should include community-aware graph traversal."""
    with patch("obsidian_ai.summary_store") as mock_ss, \
         patch("obsidian_ai.entity_store") as mock_es, \
         patch("obsidian_ai.entity_relations") as mock_er, \
         patch("obsidian_ai.graph_store") as mock_gs:

        mock_ss.query.return_value = [
            {"path": "A.md", "title": "A", "similarity": 0.85, "summary": "Note A"},
        ]

        mock_es.search.side_effect = [
            [],
            [],
            [],
        ]
        mock_er.get_related.return_value = []

        # Community neighbors: A.md is in same community as C.md
        mock_gs.community_neighbors.return_value = {
            "A.md": ["C.md", "D.md"],
        }

        from obsidian_ai.ranker import composite_search
        results = composite_search("test", n=5, retrieval_depth=3)

    assert len(results) == 3  # A.md + C.md + D.md
    paths = {r["path"] for r in results}
    assert "A.md" in paths
    assert "C.md" in paths
    assert "D.md" in paths
    mock_gs.community_neighbors.assert_called_once()


def test_composite_search_no_results():
    """When all signals return empty, should return empty list."""
    with patch("obsidian_ai.summary_store") as mock_ss, \
         patch("obsidian_ai.entity_store") as mock_es, \
         patch("obsidian_ai.entity_relations") as mock_er:

        mock_ss.query.return_value = []  # no summary results
        mock_es.search.side_effect = [[], [], []]
        mock_er.get_related.return_value = []

        from obsidian_ai.ranker import composite_search
        results = composite_search("nonexistent", n=5, retrieval_depth=3)

    assert results == []


def test_composite_search_respects_min_similarity():
    with patch("obsidian_ai.summary_store") as mock_ss, \
         patch("obsidian_ai.entity_store") as mock_es, \
         patch("obsidian_ai.entity_relations") as mock_er:

        mock_ss.query.return_value = [
            {"path": "A.md", "title": "A", "similarity": 0.85, "summary": "Note A"},
            {"path": "B.md", "title": "B", "similarity": 0.30, "summary": "Note B"},
        ]

        mock_es.search.side_effect = [[], [], []]
        mock_er.get_related.return_value = []

        from obsidian_ai.ranker import composite_search
        results = composite_search("test", n=5, retrieval_depth=2, min_similarity=0.5)

    assert len(results) == 1
    assert results[0]["path"] == "A.md"


# ── Community Boost ──────────────────────────────────────────────────


def test_community_boost_boosts_semantic_results():
    r = Ranker({"semantic": 1.0, "entity": 0.0, "graph": 0.0, "keyword": 0.0})
    mock_embed = MagicMock(return_value=[0.1] * 768)

    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma, \
         patch("obsidian_ai.graph_store") as mock_gs:

        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = [
            {"id": "A::chunk_0", "metadata": {"path": "A.md"}, "distance": 0.2, "document": "text a"},
            {"id": "B::chunk_0", "metadata": {"path": "B.md"}, "distance": 0.3, "document": "text b"},
            {"id": "C::chunk_0", "metadata": {"path": "C.md"}, "distance": 0.5, "document": "text c"},
        ]
        # A and B share community 0; C is in community 1
        mock_gs.label_propagation.return_value = {"A.md": 0, "B.md": 0, "C.md": 1}

        results = r.search(
            "test query", n=5,
            use_community_boost=True, community_boost_weight=0.5,
        )

    assert len(results) >= 2
    scores = {r["path"]: r["similarity_score"] for r in results}
    # B should have a higher score than C (same community as top semantic A)
    assert scores["B.md"] > scores["C.md"]
    # A and B should have "community" in matched_by
    a_entry = next(r for r in results if r["path"] == "A.md")
    b_entry = next(r for r in results if r["path"] == "B.md")
    assert "community" in a_entry["matched_by"]
    assert "community" in b_entry["matched_by"]


def test_community_boost_noop_when_disabled():
    r = Ranker({"semantic": 1.0, "entity": 0.0, "graph": 0.0, "keyword": 0.0})
    mock_embed = MagicMock(return_value=[0.1] * 768)

    with patch("obsidian_ai.ranker.llm_client") as mock_llm, \
         patch("obsidian_ai.ranker.chroma_store") as mock_chroma, \
         patch("obsidian_ai.graph_store") as mock_gs:

        mock_llm.embed = mock_embed
        mock_chroma.query.return_value = [
            {"id": "A::chunk_0", "metadata": {"path": "A.md"}, "distance": 0.2, "document": "text a"},
            {"id": "B::chunk_0", "metadata": {"path": "B.md"}, "distance": 0.3, "document": "text b"},
        ]

        r.search("test query", n=5, use_community_boost=False)

    # No community boost applied — scores should be pure semantic
    mock_gs.label_propagation.assert_not_called()


# ── Intent detection ────────────────────────────────────────────────


def test_detect_intent_general():
    assert detect_intent("what is the meaning of life") == "general"


@patch("obsidian_ai.entity_store.search", return_value=[])
def test_detect_intent_entity_no_match(patched):
    assert detect_intent("who is alice") == "entity"


@patch("obsidian_ai.entity_store.search", return_value=[])
def test_detect_intent_entity_pattern_what_is(patched):
    assert detect_intent("what is python") == "entity"


@patch("obsidian_ai.entity_store.search", return_value=[])
def test_detect_intent_tell_me_about(patched):
    assert detect_intent("tell me about ESP32") == "entity"


def test_detect_intent_keyword_how_to():
    assert detect_intent("how to configure ollama") == "keyword"


def test_detect_intent_keyword_file_ext():
    assert detect_intent("pip install setup.py") == "keyword"


def test_detect_intent_graph_relationship():
    assert detect_intent("how does X relate to Y") == "graph"


def test_detect_intent_graph_connection():
    assert detect_intent("connection between A and B") == "graph"


def test_detect_intent_graph_related_to():
    assert detect_intent("notes related to python") == "graph"


def test_intent_weights_have_all_keys():
    for _intent, weights in INTENT_WEIGHTS.items():
        assert "semantic" in weights
        assert "entity" in weights
        assert "graph" in weights
        assert "keyword" in weights


def test_intent_weights_sum_to_one():
    for intent, weights in INTENT_WEIGHTS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, f"{intent} weights sum to {total}"


def test_auto_weights_with_ranker():
    r = Ranker()
    results = r.search(query="python", n=5, auto_weights=True)
    assert isinstance(results, list)
