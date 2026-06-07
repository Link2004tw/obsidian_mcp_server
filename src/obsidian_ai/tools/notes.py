"""Consolidated notes tool — read, write, list, search notes."""

import json
import os
import re

from .. import indexer, obsidian_client
from ..frontmatter import validate as fm_validate
from ..logger import get_logger, log_error
from ._shared import _normalize_path
from ._tool_base import build_tool

log = get_logger("obsidian_ai.tools.notes")

_VALID_ACTIONS = {"read", "write", "list", "list_folder", "search_by_tags", "read_by_title", "add_note_to_subject"}


def _handle_read(path: str) -> str:
    path = _normalize_path(path)
    return obsidian_client.get_note(path)


def _handle_write(path: str, content: str, sync: bool = True) -> str:
    path = _normalize_path(path)
    warnings = fm_validate(content)
    if warnings:
        log.warning(f"write — frontmatter warnings for {path}: {warnings}")
    obsidian_client.put_note(path, content)
    if sync:
        indexer.index_note(path)
    msg = f"Written: {path}"
    if warnings:
        msg += "\nFrontmatter warnings:\n" + "\n".join(f"  \u2022 {w}" for w in warnings)
    return msg


def _handle_list() -> str:
    notes = obsidian_client.list_all_notes()
    return json.dumps(notes, ensure_ascii=False, indent=2)


def _handle_list_folder(folder: str) -> str:
    folder = _normalize_path(folder) if folder else ""
    entries = obsidian_client.list_folder(folder)
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _handle_search_by_tags(tags: list[str], n: int = 10) -> str:
    from .. import chroma_store
    results = chroma_store.search_by_tags(tags, n=n)
    out = []
    for r in results:
        raw = r.get("tags_str", "")
        note_tags = [t for t in raw.strip(",").split(",") if t] if raw else []
        out.append({
            "path": r["path"],
            "title": r.get("title", ""),
            "tags": note_tags,
            "snippet": r.get("snippet", ""),
        })
    return json.dumps(out, ensure_ascii=False, indent=2)


def _handle_read_by_title(title: str, folder_path: str = "") -> str:
    from .. import chroma_store
    folder_path = _normalize_path(folder_path) if folder_path else ""
    matches = chroma_store.get_by_title(title)
    if not matches:
        return f"Error: No note found with title: {title}"

    paths = [m["path"] for m in matches]
    if folder_path:
        prefix = folder_path.replace("\\", "/").rstrip("/") + "/"
        filtered = [p for p in paths if p.startswith(prefix)]
        if not filtered:
            return f'Error: No note with title "{title}" found in folder: {folder_path}'
        paths = filtered

    if len(paths) == 1:
        return obsidian_client.get_note(paths[0])

    parts = []
    for p in paths:
        content = obsidian_client.get_note(p)
        parts.append(f"\u2500\u2500\u2500 {p} \u2500\u2500\u2500\n{content}")
    return f'Found {len(paths)} notes with title "{title}":\n\n' + "\n\n".join(parts)


def _handle_add_note_to_subject(
    subject: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    sync: bool = True,
    reindex_matches: bool = True,
) -> str:
    from .. import entity_store, graph_store
    from ._shared import _find_notes_mentioning

    safe_subject = re.sub(r'[<>:"/\\|?*]', "", subject).strip() or "untitled"
    safe_title = re.sub(r'[<>:"/\\|?*]', "", title).strip() or "untitled"
    safe_subject_lower = safe_subject.casefold().replace(" ", "-")

    folder = f"Subjects/{safe_subject}"
    hub_path = f"{folder}/{safe_subject}.md"
    note_path = f"{folder}/{safe_title}.md"

    all_tags = [safe_subject_lower]
    if tags:
        for t in tags:
            t_clean = str(t).strip().lower().replace(" ", "-")
            if t_clean and t_clean not in all_tags:
                all_tags.append(t_clean)

    hub_exists = True
    try:
        hub_content = obsidian_client.get_note(hub_path)
    except Exception:
        hub_exists = False

    if not hub_exists:
        hub_lines = [
            "---",
            "tags:",
            f"  - {safe_subject_lower}",
            "---",
            "",
            f"# {safe_subject}",
            "",
            f"Notes about {safe_subject}:",
            "",
        ]
        hub_content = "\n".join(hub_lines)
        obsidian_client.put_note(hub_path, hub_content)
        graph_store.register_title(hub_path)

    tag_lines = "\n".join(f"  - {t}" for t in all_tags)
    body = content.strip()
    backlink = f"[[{safe_subject}]]"
    if backlink not in body:
        body = body + f"\n\n{backlink}"
    new_note_content = f"---\ntags:\n{tag_lines}\n---\n\n{body}"
    obsidian_client.put_note(note_path, new_note_content)
    graph_store.register_title(note_path)

    hub_link = f"[[{safe_title}]]"
    if hub_link not in hub_content:
        hub_content = hub_content.rstrip() + f"\n\n{hub_link}"
        obsidian_client.put_note(hub_path, hub_content)

    graph_store.add_edge(note_path, hub_path)
    graph_store.add_edge(hub_path, note_path)
    graph_store.flush()

    entity_store.add_manual_entity(safe_subject, "Concept", aliases=[safe_subject])
    if not graph_store.has_entity_edge("Concept", safe_subject, note_path):
        graph_store.add_entity_edge("Concept", safe_subject, note_path)
        graph_store.flush()

    if sync:
        indexer.index_note(note_path)
        if not hub_exists:
            indexer.index_note(hub_path)

    reindexed = 0
    if reindex_matches:
        for p in _find_notes_mentioning(safe_subject):
            if p == note_path or p == hub_path:
                continue
            if indexer.index_note(p, force=True):
                reindexed += 1

    tag_summary = ", ".join(all_tags)
    lines = [
        f"Note created: {note_path}",
        f"  Subject:    {safe_subject}",
        f"  Tags:       {tag_summary}",
        f"  Graph:      linked to hub ({hub_path}) via [[wiki-links]]",
        f"  Entity:     \"{safe_subject}\" registered as Concept",
    ]
    if reindexed:
        lines.append(f"  Re-indexed: {reindexed} existing notes mentioning this subject")
    return "\n".join(lines)


_HANDLERS = {
    "read": _handle_read,
    "write": _handle_write,
    "list": _handle_list,
    "list_folder": _handle_list_folder,
    "search_by_tags": _handle_search_by_tags,
    "read_by_title": _handle_read_by_title,
    "add_note_to_subject": _handle_add_note_to_subject,
}


@build_tool("notes")
def notes(
    action: str,
    path: str = "",
    content: str = "",
    folder: str = "",
    title: str = "",
    tags: list[str] | None = None,
    n: int = 10,
    subject: str = "",
    sync: bool = True,
    reindex_matches: bool = True,
) -> str:
    """Create, read, update, list, and organize notes in the vault.

    Args:
        action: ``read`` — read full note content by path.
                ``write`` — create or overwrite a note with Markdown content.
                ``list`` — list every note in the vault.
                ``list_folder`` — list notes/subfolders in a folder.
                ``search_by_tags`` — find notes with all given YAML frontmatter tags (AND logic).
                ``read_by_title`` — look up a note by filename (without ``.md``) and read it.
                ``add_note_to_subject`` — create a note under Subjects/ with auto hub and backlinks.
        path: vault-relative path for read/write actions.
        content: Markdown body content (for write and add_note_to_subject).
        folder: vault-relative folder path for list_folder and read_by_title.
        title: note title (filename without ``.md``) for read_by_title and add_note_to_subject.
        tags: list of tag strings for search_by_tags.
        n: max results for search_by_tags (default 10).
        subject: subject name for add_note_to_subject (e.g. ``"Maria"``).
        sync: if True (default), re-index after write/add_note_to_subject.
        reindex_matches: if True (default), re-index existing notes mentioning the subject.

    Returns:
        A string — either note content, JSON data, or a confirmation message.
    """
    if action not in _VALID_ACTIONS:
        valid = ", ".join(sorted(_VALID_ACTIONS))
        return f"Error: Invalid action '{action}'. Valid actions: {valid}"

    handler = _HANDLERS[action]
    try:
        if action == "read":
            return handler(path=path)
        elif action == "write":
            return handler(path=path, content=content, sync=sync)
        elif action == "list":
            return handler()
        elif action == "list_folder":
            return handler(folder=folder)
        elif action == "search_by_tags":
            return handler(tags=tags or [], n=n)
        elif action == "read_by_title":
            return handler(title=title, folder_path=folder)
        elif action == "add_note_to_subject":
            return handler(subject=subject, title=title, content=content, tags=tags, sync=sync, reindex_matches=reindex_matches)
        else:
            return f"Error: Unhandled action '{action}'"
    except Exception as e:
        log_error(log, f"notes — action={action} FAILED", exc=e)
        return f"Error: {e}"


__all_tools__ = [notes]
