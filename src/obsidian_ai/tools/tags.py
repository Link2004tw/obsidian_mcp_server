"""Consolidated tags tool — add, remove, set, batch, auto-suggest frontmatter tags."""

from .. import indexer, obsidian_client
from ..frontmatter import add_tags as fm_add_tags
from ..frontmatter import parse as fm_parse
from ..frontmatter import remove_tags as fm_remove_tags
from ..frontmatter import set_tags as fm_set_tags
from ..logger import get_logger, log_error
from ._shared import _normalize_path
from ._tool_base import build_tool

log = get_logger("obsidian_ai.tools.tags")

_VALID_ACTIONS = {"add", "remove", "set", "batch_add", "auto_suggest"}


def _handle_add(path: str, tags: list[str], sync: bool = True) -> str:
    path = _normalize_path(path)
    content = obsidian_client.get_note(path)
    new_content = fm_add_tags(content, tags)
    obsidian_client.put_note(path, new_content)
    if sync:
        indexer.index_note(path)
    meta, _ = fm_parse(new_content)
    return f"Tags added to {path}: {meta.get('tags', [])}"


def _handle_remove(path: str, tags: list[str], sync: bool = True) -> str:
    path = _normalize_path(path)
    content = obsidian_client.get_note(path)
    new_content = fm_remove_tags(content, tags)
    obsidian_client.put_note(path, new_content)
    if sync:
        indexer.index_note(path)
    meta, _ = fm_parse(new_content)
    return f"Tags removed from {path}. Remaining: {meta.get('tags', [])}"


def _handle_set(path: str, tags: list[str], sync: bool = True) -> str:
    path = _normalize_path(path)
    content = obsidian_client.get_note(path)
    new_content = fm_set_tags(content, tags)
    obsidian_client.put_note(path, new_content)
    if sync:
        indexer.index_note(path)
    meta, _ = fm_parse(new_content)
    return f"Tags set on {path}: {meta.get('tags', [])}"


def _handle_batch_add(note_paths: list[str], tags: list[str], sync: bool = True) -> str:
    results: dict[str, str] = {}
    for note_path in note_paths:
        try:
            content = obsidian_client.get_note(note_path)
            new_content = fm_add_tags(content, tags)
            obsidian_client.put_note(note_path, new_content)
            if sync:
                indexer.index_note(note_path)
            meta, _ = fm_parse(new_content)
            results[note_path] = f"Tags added: {meta.get('tags', [])}"
        except Exception as e:
            log_error(log, f"batch_add — {note_path} FAILED", exc=e)
            results[note_path] = f"Error: {e}"
    import json
    return json.dumps(results, ensure_ascii=False, indent=2)


def _handle_auto_suggest(query: str, top_k: int = 5, sync: bool = True) -> str:
    from .. import pipelines
    result = pipelines.tag_notes(ask=query, top_k=top_k)
    if sync:
        import json as _json
        import re
        m = re.search(r"\{.*\}", result, re.DOTALL)
        if m:
            tag_map = _json.loads(m.group())
            for path in tag_map:
                indexer.index_note(path)
    return result


_HANDLERS = {
    "add": _handle_add,
    "remove": _handle_remove,
    "set": _handle_set,
    "batch_add": _handle_batch_add,
    "auto_suggest": _handle_auto_suggest,
}


@build_tool("tags")
def tags(
    action: str,
    path: str = "",
    tags: list[str] | None = None,
    note_paths: list[str] | None = None,
    query: str = "",
    top_k: int = 5,
    sync: bool = True,
) -> str:
    """Manage YAML frontmatter tags on notes.

    Args:
        action: ``add`` — add tags to a note without affecting existing tags.
                ``remove`` — remove specific tags from a note.
                ``set`` — replace all tags on a note with a new list.
                ``batch_add`` — add the same tags to multiple notes at once.
                ``auto_suggest`` — search notes matching a query and auto-suggest tags via LLM.
        path: vault-relative note path (for add/remove/set).
        tags: list of tag strings to add/remove/set.
        note_paths: list of note paths (for batch_add).
        query: semantic search query (for auto_suggest).
        top_k: number of notes to process for auto_suggest (default 5).
        sync: if True (default), re-index affected notes.

    Returns:
        A confirmation string or JSON mapping.
    """
    if action not in _VALID_ACTIONS:
        valid = ", ".join(sorted(_VALID_ACTIONS))
        return f"Error: Invalid action '{action}'. Valid actions: {valid}"

    handler = _HANDLERS[action]
    try:
        if action == "add":
            return handler(path=path, tags=tags or [], sync=sync)
        elif action == "remove":
            return handler(path=path, tags=tags or [], sync=sync)
        elif action == "set":
            return handler(path=path, tags=tags or [], sync=sync)
        elif action == "batch_add":
            return handler(note_paths=note_paths or [], tags=tags or [], sync=sync)
        elif action == "auto_suggest":
            return handler(query=query, top_k=top_k, sync=sync)
        else:
            return f"Error: Unhandled action '{action}'"
    except Exception as e:
        log_error(log, f"tags — action={action} FAILED", exc=e)
        return f"Error: {e}"


__all_tools__ = [tags]
