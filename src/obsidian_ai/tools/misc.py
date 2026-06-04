"""Miscellaneous tools: clustering, health check."""

from .. import chroma_store, config, llm_client, obsidian_client
from ..logger import get_logger, log_error

log = get_logger("obsidian_ai.tools.misc")


def get_clusters(
    force_recompute: bool = False,
    similarity_threshold: float = 0.6,
) -> list[dict]:
    """Return semantic clusters of notes based on embedding similarity.

    Groups notes by meaning across the vault. Each cluster has a
    descriptive label, central note, and list of member notes.

    Args:
        force_recompute: if True, ignore cache and re-run clustering.
        similarity_threshold: minimum cosine similarity (0-1) for two
            notes to be considered connected (default 0.6).

    Returns:
        List of clusters, each with:
        - ``label`` — descriptive cluster label
        - ``notes`` — list of note paths in the cluster
        - ``size`` — number of notes in the cluster
        - ``central_note`` — path of the most central note
    """
    log.info(f"get_clusters — force_recompute={force_recompute}, similarity_threshold={similarity_threshold}")
    try:
        from ..clustering import get_clusters as _get_clusters
        result = _get_clusters(force_recompute=force_recompute, similarity_threshold=similarity_threshold)
        log.info(f"get_clusters — {len(result)} clusters returned")
        return result
    except Exception as e:
        log_error(log, "get_clusters FAILED", exc=e)
        return []


def health_check() -> dict:
    """Check the health of all backend services (Ollama, config, caches).

    Returns a dict with per-service status and an overall health indicator.

    Returns:
        Dict with ``ollama`` status, ``overall`` (``"healthy"`` or ``"degraded"``).
    """
    log.info("health_check")
    try:
        health = llm_client.check_health()
        note_count = len(obsidian_client.list_all_notes())
        chunk_count = chroma_store.count()
        overall = "healthy"
        if not health.get("ollama", {}).get("available", False):
            overall = "degraded"
        return {
            "ollama": health.get("ollama", {}),
            "embed_model": config.ollama_embed_model,
            "chat_model": config.ollama_chat_model,
            "note_count": note_count,
            "chunk_count": chunk_count,
            "overall": overall,
        }
    except Exception as e:
        log_error(log, "health_check FAILED", exc=e)
        return {
            "ollama": {"available": False},
            "overall": "degraded",
            "error": str(e),
        }


__all_tools__ = [
    get_clusters,
    health_check,
]
