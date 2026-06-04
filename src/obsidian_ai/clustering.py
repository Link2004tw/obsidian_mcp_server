"""Semantic clustering of notes by embedding similarity."""

import json
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np

from . import chroma_store, config, llm_client
from .logger import get_logger

log = get_logger(__name__)

_CLUSTER_CACHE_PATH = os.path.join(config.data_dir, "clusters.json")
_CLUSTER_CACHE_TTL = 3600  # 1 hour
_cluster_cache: dict[str, Any] | None = None
_cache_ts: float = 0.0


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = vectors / norms
    return normalized @ normalized.T


def _connected_components(sim_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    n = len(sim_matrix)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                adj[i].append(j)
                adj[j].append(i)
    visited = [False] * n
    components: list[list[int]] = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            comp: list[int] = []
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    comp.append(node)
                    stack.extend(adj[node])
            if comp:
                components.append(comp)
    return components


def _aggregate_note_embeddings(
    all_data: list[dict],
) -> tuple[list[str], np.ndarray]:
    note_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for item in all_data:
        emb = item["embedding"]
        path = item["metadata"].get("path", "")
        if emb is not None and path:
            note_chunks[path].append(np.array(emb, dtype=np.float64))
    paths: list[str] = []
    vectors: list[np.ndarray] = []
    for path, chunk_vecs in note_chunks.items():
        paths.append(path)
        vectors.append(np.mean(chunk_vecs, axis=0))
    return paths, np.array(vectors)


def _generate_cluster_label(indices: list[int], paths: list[str], vectors: np.ndarray) -> str:
    if len(indices) == 1:
        path = paths[indices[0]]
        title = os.path.splitext(os.path.basename(path))[0]
        return title
    sub_vecs = vectors[indices]
    sim_matrix = _cosine_similarity_matrix(sub_vecs)
    centrality = sim_matrix.sum(axis=1)
    central_idx = indices[int(np.argmax(centrality))]
    central_path = paths[central_idx]
    central_title = os.path.splitext(os.path.basename(central_path))[0]
    titles = []
    for idx in indices[:5]:
        t = os.path.splitext(os.path.basename(paths[idx]))[0]
        if t != central_title:
            titles.append(t)
    if titles:
        return f"{central_title} (+{', '.join(titles)})"
    return central_title


def compute_clusters(
    similarity_threshold: float = 0.6,
    min_cluster_size: int = 1,
) -> list[dict]:
    """Run semantic clustering on all note embeddings in the vault.

    Args:
        similarity_threshold: minimum cosine similarity (0-1) for two
            notes to be considered connected (default 0.6).
        min_cluster_size: minimum notes per cluster; smaller groups
            are returned as singletons (default 1).

    Returns:
        List of dicts, each with:
        - ``label`` — descriptive label for the cluster
        - ``notes`` — list of note paths in the cluster
        - ``size`` — number of notes
        - ``central_note`` — the most central note path
    """
    all_data = chroma_store.get_all_embeddings()
    if not all_data:
        return []

    paths, vectors = _aggregate_note_embeddings(all_data)
    if len(paths) < 2:
        return [{"label": os.path.splitext(os.path.basename(paths[0]))[0],
                 "notes": list(paths), "size": 1, "central_note": paths[0]}]

    sim_matrix = _cosine_similarity_matrix(vectors)
    components = _connected_components(sim_matrix, similarity_threshold)

    clusters: list[dict] = []
    for comp in components:
        if len(comp) < min_cluster_size:
            for idx in comp:
                path = paths[idx]
                title = os.path.splitext(os.path.basename(path))[0]
                clusters.append({"label": title, "notes": [path], "size": 1, "central_note": path})
        else:
            label = _generate_cluster_label(comp, paths, vectors)
            central_idx = comp[int(np.argmax(sim_matrix[comp].sum(axis=1)))]
            clusters.append({
                "label": label,
                "notes": [paths[i] for i in comp],
                "size": len(comp),
                "central_note": paths[central_idx],
            })

    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


def _save_cluster_cache(clusters: list[dict]) -> None:
    global _cluster_cache, _cache_ts
    try:
        data = {"timestamp": time.time(), "clusters": clusters}
        os.makedirs(os.path.dirname(_CLUSTER_CACHE_PATH), exist_ok=True)
        with open(_CLUSTER_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _cluster_cache = data
        _cache_ts = data["timestamp"]
    except Exception as e:
        log.warning(f"Failed to save cluster cache: {e}")


def _load_cluster_cache() -> dict | None:
    global _cluster_cache, _cache_ts
    if _cluster_cache is not None and time.time() - _cache_ts < _CLUSTER_CACHE_TTL:
        return _cluster_cache
    try:
        if os.path.isfile(_CLUSTER_CACHE_PATH):
            with open(_CLUSTER_CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("timestamp", 0) < _CLUSTER_CACHE_TTL:
                _cluster_cache = data
                _cache_ts = data["timestamp"]
                return data
    except Exception as e:
        log.warning(f"Failed to load cluster cache: {e}")
    return None


def get_clusters(force_recompute: bool = False, similarity_threshold: float = 0.6) -> list[dict]:
    """Return cached or freshly-computed semantic clusters.

    Args:
        force_recompute: if True, ignore cache and re-run clustering.
        similarity_threshold: passed to ``compute_clusters`` on recompute.

    Returns:
        Same structure as ``compute_clusters``.
    """
    if not force_recompute:
        cached = _load_cluster_cache()
        if cached is not None:
            return cached["clusters"]
    clusters = compute_clusters(similarity_threshold=similarity_threshold)
    _save_cluster_cache(clusters)
    return clusters


def generate_cluster_summary(clusters: list[dict]) -> str:
    """Generate a short LLM summary of the cluster landscape."""
    if not clusters:
        return "No clusters found."
    overview = []
    for c in clusters[:10]:
        overview.append(f"- {c['label']} ({c['size']} notes)")
    prompt = (
        "Here are the top semantic clusters found in a knowledge vault:\n"
        + "\n".join(overview)
        + "\n\nWrite a 2-3 sentence summary of what these clusters reveal about the vault's content."
    )
    try:
        return llm_client.chat([{"role": "user", "content": prompt}])
    except Exception as e:
        log.warning(f"Cluster summary failed: {e}")
        return "\n".join(overview)
