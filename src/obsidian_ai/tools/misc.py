"""Miscellaneous tools: clustering, health check."""

from .. import chroma_store, config, llm_client, obsidian_client
from ..logger import get_logger, log_error

log = get_logger("obsidian_ai.tools.misc")


def get_clusters(
    force_recompute: bool = False,
    similarity_threshold: float = 0.6,
) -> list[dict]:
    """Group notes into semantically similar clusters using embedding cosine similarity. Use this when the user wants to discover thematic groupings, identify topic areas, or get an overview of how their notes relate by meaning across the vault.

    Notes are clustered via community detection on a similarity graph built from embedding vectors. Each returned cluster includes a human-readable label, the member notes, and the most representative (central) note. Results are cached; pass ``force_recompute=True`` to regenerate.

    Args:
        force_recompute: Whether to ignore the cached clustering result and re-run from scratch. Defaults to ``False``. Use ``True`` after adding or editing many notes to get fresh groupings.
        similarity_threshold: Minimum cosine similarity (0.0 to 1.0) required for two notes to be connected in the similarity graph. Higher values produce tighter, smaller clusters; lower values produce broader, fewer clusters. Defaults to ``0.6``.

    Returns:
        A list of cluster dicts, each containing:
        - ``label`` (str) — a descriptive label generated from the cluster's content
        - ``notes`` (list[str]) — vault-relative paths of all notes in the cluster
        - ``size`` (int) — number of notes in the cluster
        - ``central_note`` (str) — the vault-relative path of the most semantically central note in the cluster

        Returns an empty list if clustering fails.
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
    """Check whether all backend services are running and responsive. Use this to verify that Ollama (or the configured OpenAI-compatible API), ChromaDB (the embedding vector store), and the Obsidian vault are accessible before calling other tools.

    Pings the LLM provider, counts notes from the Obsidian vault, and counts chunks in ChromaDB. Returns per-service status and an overall health indicator.

    Returns:
        A dict with:
        - ``ollama`` (dict) — status response from the LLM provider, includes keys like ``available`` (bool) and possibly ``models`` (list).
        - ``embed_model`` (str) — name of the embedding model configured (e.g. ``"nomic-embed-text"``).
        - ``chat_model`` (str) — name of the chat / LLM model configured (e.g. ``"qwen2.5"`` or ``"gpt-4"``).
        - ``note_count`` (int) — total number of notes found in the Obsidian vault.
        - ``chunk_count`` (int) — total number of indexed chunks in ChromaDB.
        - ``overall`` (str) — ``"healthy"`` if the LLM provider is available, ``"degraded"`` otherwise.
        - ``error`` (str) — present only if an exception was raised, containing the error message.
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
