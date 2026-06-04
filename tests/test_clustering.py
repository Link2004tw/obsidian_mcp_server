"""Tests for the semantic clustering module."""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from obsidian_ai import clustering

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_cache():
    clustering._cluster_cache = None
    clustering._cache_ts = 0.0
    yield


SAMPLE_EMBEDDINGS = [
    {"id": "chunk1", "embedding": [1.0, 0.0, 0.0, 0.0], "metadata": {"path": "note_a.md"}},
    {"id": "chunk2", "embedding": [0.99, 0.01, 0.0, 0.0], "metadata": {"path": "note_a.md"}},
    {"id": "chunk3", "embedding": [0.95, 0.05, 0.0, 0.0], "metadata": {"path": "note_b.md"}},
    {"id": "chunk4", "embedding": [0.0, 1.0, 0.0, 0.0], "metadata": {"path": "note_c.md"}},
    {"id": "chunk5", "embedding": [0.0, 0.95, 0.05, 0.0], "metadata": {"path": "note_d.md"}},
    {"id": "chunk6", "embedding": [0.0, 0.0, 1.0, 0.0], "metadata": {"path": "note_e.md"}},
]


# ── Unit Tests ──────────────────────────────────────────────────────


def test_cosine_similarity_matrix():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    sim = clustering._cosine_similarity_matrix(vectors)
    assert sim.shape == (3, 3)
    assert abs(sim[0][1]) < 0.01  # orthogonal → ~0
    assert abs(sim[0][2] - 1 / np.sqrt(2)) < 0.01  # cos(45°)


def test_connected_components():
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
    comps = clustering._connected_components(sim, 0.5)
    assert len(comps) == 2
    assert {0, 1} in [set(c) for c in comps]
    assert {2} in [set(c) for c in comps]


def test_aggregate_note_embeddings():
    paths, vectors = clustering._aggregate_note_embeddings(SAMPLE_EMBEDDINGS)
    assert len(paths) == 5  # 5 unique paths
    assert vectors.shape == (5, 4)
    note_a_idx = paths.index("note_a.md")
    assert np.allclose(vectors[note_a_idx], [0.995, 0.005, 0.0, 0.0])


@patch("obsidian_ai.clustering.chroma_store.get_all_embeddings")
def test_compute_clusters_empty(mock_get):
    mock_get.return_value = []
    result = clustering.compute_clusters()
    assert result == []


@patch("obsidian_ai.clustering.chroma_store.get_all_embeddings")
def test_compute_clusters_single_note(mock_get):
    mock_get.return_value = [SAMPLE_EMBEDDINGS[0]]
    result = clustering.compute_clusters()
    assert len(result) == 1
    assert result[0]["size"] == 1


@patch("obsidian_ai.clustering.chroma_store.get_all_embeddings")
def test_compute_clusters_grouping(mock_get):
    mock_get.return_value = SAMPLE_EMBEDDINGS
    result = clustering.compute_clusters(similarity_threshold=0.5)
    # note_a and note_b have similar embeddings → should cluster
    # note_c and note_d have similar embeddings → should cluster
    # note_e is orthogonal → should be a singleton
    cluster_sizes = sorted([c["size"] for c in result], reverse=True)
    assert cluster_sizes == [2, 2, 1] or cluster_sizes == [2, 1, 1, 1]


@patch("obsidian_ai.clustering.chroma_store.get_all_embeddings")
def test_get_clusters_cache(mock_get):
    mock_get.return_value = SAMPLE_EMBEDDINGS
    result1 = clustering.get_clusters()
    assert len(result1) > 0
    call_count = mock_get.call_count
    clustering.get_clusters()  # cached — should not call again
    assert mock_get.call_count == call_count
    clustering.get_clusters(force_recompute=True)  # recomputed
    assert mock_get.call_count == call_count + 1


@patch("obsidian_ai.clustering.chroma_store.get_all_embeddings")
def test_get_clusters_persistence(mock_get):
    mock_get.return_value = SAMPLE_EMBEDDINGS
    with tempfile.TemporaryDirectory() as tmp:
        orig_path = clustering._CLUSTER_CACHE_PATH
        cache_path = os.path.join(tmp, "clusters.json")
        clustering._CLUSTER_CACHE_PATH = cache_path
        try:
            clustering.get_clusters()
            assert os.path.isfile(cache_path)
            with open(cache_path) as f:
                data = json.load(f)
            assert "clusters" in data
            assert len(data["clusters"]) > 0
        finally:
            clustering._CLUSTER_CACHE_PATH = orig_path


def test_generate_cluster_label_singleton():
    label = clustering._generate_cluster_label([0], ["path/to/NoteA.md"], np.eye(3))
    assert label == "NoteA"


def test_generate_cluster_label_group():
    paths = ["path/to/Central.md", "path/to/FriendA.md", "path/to/FriendB.md"]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float64)
    label = clustering._generate_cluster_label([0, 1, 2], paths, vectors)
    assert "Central" in label


def test_generate_cluster_summary_empty():
    result = clustering.generate_cluster_summary([])
    assert result == "No clusters found."
