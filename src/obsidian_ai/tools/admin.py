"""Consolidated admin tool — health check, reindex, model switching."""

import json

import requests

from .. import chroma_store, config, entity_store, indexer, keyword_search, llm_client, obsidian_client
from .. import todos as _impl
from ..logger import get_logger, log_error
from ._tool_base import build_tool

log = get_logger("obsidian_ai.tools.admin")

_VALID_ACTIONS = {"health", "reindex", "stats", "switch_model", "sync_todos"}


def _handle_health() -> str:
    health = llm_client.check_health()
    note_count = len(obsidian_client.list_all_notes())
    chunk_count = chroma_store.count()
    overall = "healthy"
    if not health.get("ollama", {}).get("available", False):
        overall = "degraded"
    result = {
        "ollama": health.get("ollama", {}),
        "embed_model": config.ollama_embed_model,
        "chat_model": config.ollama_chat_model,
        "note_count": note_count,
        "chunk_count": chunk_count,
        "overall": overall,
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


def _handle_reindex(folder: str | None = None, subject: str | None = None) -> str:
    if subject:
        found = entity_store.search(subject)
        paths = sorted(set(n["path"] for n in found))
        if not paths:
            return f"No notes found mentioning '{subject}'."
        count = 0
        for path in paths:
            if indexer.index_note(path, force=True):
                count += 1
        return f"Re-indexed {count} notes mentioning '{subject}'."

    indexer.run_index(folder=folder)
    entity_store.rebuild()
    llm_client.clear_embed_cache()
    keyword_search.ensure_index()
    return f"Index sync complete{' for folder: ' + folder if folder else ''}."


def _handle_stats() -> str:
    stats = chroma_store.get_index_stats()
    cache = llm_client.embed_cache_info()
    ent_stats = entity_store.stats()
    lines = [
        f"Unique notes indexed: {stats['unique_notes']}",
        f"Total chunks stored:  {stats['total_chunks']}",
        f"Embedding model:      {config.ollama_embed_model}",
        f"ChromaDB path:        {config.chroma_path}",
        f"Embedding cache:      {cache['currsize']}/{cache['maxsize']} (hits={cache['hits']}, misses={cache['misses']})",
        f"Entities extracted:   {ent_stats['total_entities']} ({ent_stats['total_mentions']} mentions)",
    ]
    return "\n".join(lines)


def _handle_switch_model(model_name: str) -> str:
    resp = requests.post(
        f"{config.ollama_base_url}/api/embeddings",
        json={"model": model_name, "prompt": "test"},
        timeout=30,
    )
    if resp.status_code != 200:
        return f"Model '{model_name}' not found in Ollama. Pull it first: ollama pull {model_name}"

    old_model = config.ollama_embed_model
    llm_client.switch_embed_model(model_name)
    chroma_store.reset_collection()
    keyword_search.ensure_index()
    indexer.run_index()
    entity_store.rebuild()
    keyword_search.ensure_index()

    return f"Switched embedding model: {old_model} \u2192 {model_name}. Vault re-indexed."


def _handle_sync_todos(sync: bool = True) -> str:
    result = _impl.sync_todos()
    if sync:
        todo_path = config.todo_file
        indexer.index_note(todo_path)
    return json.dumps(result, indent=2, ensure_ascii=False)


_HANDLERS = {
    "health": _handle_health,
    "reindex": _handle_reindex,
    "stats": _handle_stats,
    "switch_model": _handle_switch_model,
    "sync_todos": _handle_sync_todos,
}


@build_tool("admin")
def admin(
    action: str,
    folder: str | None = None,
    subject: str | None = None,
    model_name: str = "",
    sync: bool = True,
) -> str:
    """System administration — health checks, re-indexing, model configuration.

    Args:
        action: ``health`` — verify LLM provider, ChromaDB, and vault are accessible.
                ``reindex`` — re-run the full indexing pipeline (or a subset).
                ``stats`` — show diagnostic index statistics.
                ``switch_model`` — switch the embedding model at runtime (requires re-index).
                ``sync_todos`` — recalculate todo statistics in frontmatter.
        folder: optional folder to re-index (for reindex action).
        subject: optional entity name — re-index notes mentioning it (for reindex action).
        model_name: Ollama embedding model name (for switch_model).
        sync: if True (default), re-index after sync_todos.

    Returns:
        A formatted string with the result.
    """
    if action not in _VALID_ACTIONS:
        valid = ", ".join(sorted(_VALID_ACTIONS))
        return f"Error: Invalid action '{action}'. Valid actions: {valid}"

    handler = _HANDLERS[action]
    try:
        if action == "health":
            return handler()
        elif action == "reindex":
            return handler(folder=folder, subject=subject)
        elif action == "stats":
            return handler()
        elif action == "switch_model":
            return handler(model_name=model_name)
        elif action == "sync_todos":
            return handler(sync=sync)
        else:
            return f"Error: Unhandled action '{action}'"
    except Exception as e:
        log_error(log, f"admin — action={action} FAILED", exc=e)
        return f"Error: {e}"


__all_tools__ = [admin]
