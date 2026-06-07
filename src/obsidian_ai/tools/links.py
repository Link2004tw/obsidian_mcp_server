"""Consolidated links tool — create, query, and find wiki-link connections."""

import json
import os
import re

from .. import graph_store, indexer, obsidian_client
from ..logger import get_logger, log_error
from ._shared import _normalize_path
from ._tool_base import build_tool

log = get_logger("obsidian_ai.tools.links")

_VALID_ACTIONS = {"create", "backlinks", "outgoing", "broken"}


def _handle_create(path_a: str, path_b: str, sync: bool = True) -> str:
    path_a = _normalize_path(path_a)
    path_b = _normalize_path(path_b)

    name_a = os.path.splitext(os.path.basename(path_a))[0]
    name_b = os.path.splitext(os.path.basename(path_b))[0]
    link_to_b = f"[[{name_b}]]"
    link_to_a = f"[[{name_a}]]"

    content_a = obsidian_client.get_note(path_a)
    if not re.search(re.escape(link_to_b), content_a):
        content_a = content_a.rstrip() + f"\n\n{link_to_b}"
        obsidian_client.put_note(path_a, content_a)

    content_b = obsidian_client.get_note(path_b)
    if not re.search(re.escape(link_to_a), content_b):
        content_b = content_b.rstrip() + f"\n\n{link_to_a}"
        obsidian_client.put_note(path_b, content_b)

    if sync:
        indexer.index_note(path_a)
        indexer.index_note(path_b)

    return f"Linked: {path_a} <-> {path_b}"


def _handle_backlinks(path: str) -> str:
    path = _normalize_path(path)
    sources = graph_store.get_backlinks(path)
    results = []
    for src in sources:
        title = os.path.splitext(os.path.basename(src))[0]
        results.append({"path": src, "title": title})
    return json.dumps(results, ensure_ascii=False, indent=2)


def _handle_outgoing(path: str) -> str:
    path = _normalize_path(path)
    targets = graph_store.get_outgoing(path)
    results = []
    for tgt in targets:
        title = os.path.splitext(os.path.basename(tgt))[0]
        results.append({"path": tgt, "title": title})
    return json.dumps(results, ensure_ascii=False, indent=2)


def _handle_broken() -> str:
    import contextlib
    notes = obsidian_client.list_all_notes()
    all_contents: dict[str, str] = {}
    for note_path in notes:
        with contextlib.suppress(Exception):
            all_contents[note_path] = obsidian_client.get_note(note_path)
    broken = graph_store.get_broken_links(all_contents)
    return json.dumps(broken, ensure_ascii=False, indent=2)


_HANDLERS = {
    "create": _handle_create,
    "backlinks": _handle_backlinks,
    "outgoing": _handle_outgoing,
    "broken": _handle_broken,
}


@build_tool("links")
def links(
    action: str,
    path: str = "",
    path_a: str = "",
    path_b: str = "",
    sync: bool = True,
) -> str:
    """Create and explore [[wiki-link]] connections between notes.

    Args:
        action: ``create`` — create bidirectional wiki-links between two notes.
                ``backlinks`` — list all notes that link TO a given note.
                ``outgoing`` — list all notes a given note links TO.
                ``broken`` — find wiki-links that point to non-existent notes.
        path: vault-relative note path (for backlinks/outgoing).
        path_a: first note path (for create).
        path_b: second note path (for create).
        sync: if True (default), re-index after creating a backlink.

    Returns:
        JSON data or a confirmation string.
    """
    if action not in _VALID_ACTIONS:
        valid = ", ".join(sorted(_VALID_ACTIONS))
        return f"Error: Invalid action '{action}'. Valid actions: {valid}"

    handler = _HANDLERS[action]
    try:
        if action == "create":
            return handler(path_a=path_a, path_b=path_b, sync=sync)
        elif action == "backlinks":
            return handler(path=path)
        elif action == "outgoing":
            return handler(path=path)
        elif action == "broken":
            return handler()
        else:
            return f"Error: Unhandled action '{action}'"
    except Exception as e:
        log_error(log, f"links — action={action} FAILED", exc=e)
        return f"Error: {e}"


__all_tools__ = [links]
