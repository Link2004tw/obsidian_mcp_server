"""Note creation, reading, updating, and deletion tools."""

import os
import re

import requests

from .. import (
    chroma_store,
    config,
    entity_store,
    graph_store,
    indexer,
    keyword_search,
    llm_client,
    obsidian_client,
)
from ..frontmatter import add_tags as fm_add_tags
from ..frontmatter import parse as fm_parse
from ..frontmatter import remove_tags as fm_remove_tags
from ..frontmatter import set_tags as fm_set_tags
from ..frontmatter import validate as fm_validate
from ..logger import get_logger, log_error
from ._shared import _find_notes_mentioning, _normalize_path

log = get_logger("obsidian_ai.tools.notes")


def read_note(path: str) -> str:
    """Read the full content of a note by path.

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.
    """
    path = _normalize_path(path)
    log.info(f"read_note — {path}")
    try:
        content = obsidian_client.get_note(path)
        log.info(f"read_note — {path} — {len(content)} chars")
        return content
    except Exception as e:
        log_error(log, f"read_note FAILED: {path}", exc=e)
        return f"Error: {e}"


def write_note(path: str, content: str, sync: bool = True) -> str:
    """Create or overwrite a note with the given content.

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.
        content: Markdown content to write.
        sync: if True (default), re-index the note so it's immediately searchable.
    """
    path = _normalize_path(path)
    log.info(f"write_note — {path} — {len(content)} chars")
    warnings = fm_validate(content)
    if warnings:
        log.warning(f"write_note — frontmatter warnings for {path}: {warnings}")
    try:
        obsidian_client.put_note(path, content)
        if sync:
            indexer.index_note(path)
        if warnings:
            return f"Written: {path}\nFrontmatter warnings:\n" + "\n".join(f"  • {w}" for w in warnings)
        return f"Written: {path}"
    except Exception as e:
        log_error(log, f"write_note FAILED: {path}", exc=e, content_len=len(content))
        return f"Error: {e}"


def list_all_notes() -> list[str]:
    """Return a list of all note paths in the vault."""
    log.info("list_all_notes")
    try:
        notes = obsidian_client.list_all_notes()
        log.info(f"list_all_notes — {len(notes)} notes returned")
        return notes
    except Exception as e:
        log_error(log, "list_all_notes FAILED", exc=e)
        return []


def list_folder(folder_path: str) -> list[str]:
    """Return entries directly inside a specific folder (non-recursive).

    Args:
        folder_path: vault-relative folder path, e.g. ``"Folder"`` or ``""`` for root — not a full filesystem path.
    """
    folder_path = _normalize_path(folder_path)
    log.info(f"list_folder — {folder_path}")
    try:
        notes = obsidian_client.list_folder(folder_path)
        log.info(f"list_folder — {len(notes)} notes returned")
        return notes
    except Exception as e:
        log_error(log, f"list_folder FAILED: {folder_path}", exc=e)
        return []


def list_folder_deep(folder_path: str) -> list[str]:
    """Return a list of note paths within a specific folder (recursive).

    Args:
        folder_path: vault-relative folder path, e.g. ``"Folder"`` or ``""`` for root — not a full filesystem path.
    """
    folder_path = _normalize_path(folder_path)
    log.info(f"list_folder_deep — {folder_path}")
    try:
        notes = obsidian_client.list_folder_deep(folder_path)
        log.info(f"list_folder_deep — {len(notes)} notes returned")
        return notes
    except Exception as e:
        log_error(log, f"list_folder_deep FAILED: {folder_path}", exc=e)
        return []


def search_by_tags(tags: list[str], n: int = 10) -> list[dict]:
    """Find notes that have ALL of the given YAML frontmatter tags."""
    log.info(f"search_by_tags — tags={tags}, n={n}")
    try:
        results = chroma_store.search_by_tags(tags, n=n)
        out = []
        for r in results:
            raw = r.get("tags_str", "")
            note_tags = [t for t in raw.strip(",").split(",") if t] if raw else []
            out.append(
                {
                    "path": r["path"],
                    "title": r.get("title", ""),
                    "tags": note_tags,
                    "snippet": r.get("snippet", ""),
                }
            )
        log.info(f"search_by_tags — {len(out)} results")
        return out
    except Exception as e:
        log_error(log, "search_by_tags FAILED", exc=e, tags=tags, n=n)
        return []


def read_note_by_title(title: str, folder_path: str = "") -> str:
    """Look up a note by its title (filename without extension) and return the full content.

    Args:
        title: filename without extension, e.g. ``"My Note"``.
        folder_path: optional vault-relative folder to disambiguate, e.g. ``"Folder"`` — not a full filesystem path.
    """
    folder_path = _normalize_path(folder_path) if folder_path else ""
    log.info(f"read_note_by_title — title={title}, folder_path={folder_path or '(none)'}")
    try:
        matches = chroma_store.get_by_title(title)
        if not matches:
            msg = f"No note found with title: {title}"
            log.warning(f"read_note_by_title — {msg}")
            return f"Error: {msg}"

        paths = [m["path"] for m in matches]

        if folder_path:
            prefix = folder_path.replace("\\", "/").rstrip("/") + "/"
            filtered = [p for p in paths if p.startswith(prefix)]
            if not filtered:
                msg = f'No note with title "{title}" found in folder: {folder_path}'
                log.warning(f"read_note_by_title — {msg}")
                return f"Error: {msg}"
            paths = filtered

        if len(paths) == 1:
            content = obsidian_client.get_note(paths[0])
            log.info(
                f"read_note_by_title — {title} → {paths[0]} — {len(content)} chars"
            )
            return content

        log.info(f"read_note_by_title — {title} — multiple matches ({len(paths)}): {paths}")
        parts = []
        for p in paths:
            content = obsidian_client.get_note(p)
            parts.append(f"─── {p} ───\n{content}")
        return f'Found {len(paths)} notes with title "{title}":\n\n' + "\n\n".join(parts)
    except Exception as e:
        log_error(log, f"read_note_by_title FAILED: title={title}, folder_path={folder_path}", exc=e)
        return f"Error: {e}"


def add_tags(path: str, tags: list[str], sync: bool = True) -> str:
    """Add tags to a note's YAML frontmatter. Creates frontmatter if absent.

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.
        tags: list of tag strings to add.
        sync: if True (default), re-index the note so changes are reflected in search.
    """
    path = _normalize_path(path)
    log.info(f"add_tags — {path} — tags={tags}")
    try:
        content = obsidian_client.get_note(path)
        new_content = fm_add_tags(content, tags)
        obsidian_client.put_note(path, new_content)
        if sync:
            indexer.index_note(path)
        meta, _ = fm_parse(new_content)
        log.info(f"add_tags — {path} — final tags={meta.get('tags', [])}")
        return f"Tags added to {path}: {meta.get('tags', [])}"
    except Exception as e:
        log_error(log, f"add_tags FAILED: {path}", exc=e, tags=tags)
        return f"Error: {e}"


def remove_tags(path: str, tags: list[str], sync: bool = True) -> str:
    """Remove specific tags from a note's YAML frontmatter.

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.
        tags: list of tags to remove.
        sync: if True (default), re-index the note so changes are reflected in search.
    """
    path = _normalize_path(path)
    log.info(f"remove_tags — {path} — tags={tags}")
    try:
        content = obsidian_client.get_note(path)
        new_content = fm_remove_tags(content, tags)
        obsidian_client.put_note(path, new_content)
        if sync:
            indexer.index_note(path)
        meta, _ = fm_parse(new_content)
        log.info(f"remove_tags — {path} — final tags={meta.get('tags', [])}")
        return f"Tags removed from {path}. Remaining: {meta.get('tags', [])}"
    except Exception as e:
        log_error(log, f"remove_tags FAILED: {path}", exc=e, tags=tags)
        return f"Error: {e}"


def set_tags(path: str, tags: list[str], sync: bool = True) -> str:
    """Replace all tags on a note with the given list.

    Args:
        path: vault-relative path, e.g. ``"Folder/Note.md"`` — not a full filesystem path.
        tags: new list of tags (replaces all existing).
        sync: if True (default), re-index the note so changes are reflected in search.
    """
    path = _normalize_path(path)
    log.info(f"set_tags — {path} — tags={tags}")
    try:
        content = obsidian_client.get_note(path)
        new_content = fm_set_tags(content, tags)
        obsidian_client.put_note(path, new_content)
        if sync:
            indexer.index_note(path)
        meta, _ = fm_parse(new_content)
        log.info(f"set_tags — {path} — final tags={meta.get('tags', [])}")
        return f"Tags set on {path}: {meta.get('tags', [])}"
    except Exception as e:
        log_error(log, f"set_tags FAILED: {path}", exc=e, tags=tags)
        return f"Error: {e}"


def batch_tag_notes(note_paths: list[str], tags: list[str], sync: bool = True) -> dict[str, str]:
    """Add tags to multiple notes at once. Returns ``dict[path, result_message]``.

    Args:
        note_paths: list of vault-relative note paths.
        tags: list of tags to add.
        sync: if True (default), re-index each note so changes are reflected in search.
    """
    log.info("batch_tag_notes — %d notes, tags=%s", len(note_paths), tags)
    results: dict[str, str] = {}
    for path in note_paths:
        try:
            content = obsidian_client.get_note(path)
            new_content = fm_add_tags(content, tags)
            obsidian_client.put_note(path, new_content)
            if sync:
                indexer.index_note(path)
            meta, _ = fm_parse(new_content)
            results[path] = f"Tags added: {meta.get('tags', [])}"
        except Exception as e:
            log_error(log, f"batch_tag_notes FAILED: {path}", exc=e, tags=tags)
            results[path] = f"Error: {e}"
    return results


def create_backlink(path_a: str, path_b: str, sync: bool = True) -> str:
    """Create mutual [[backlinks]] between two notes.

    Args:
        path_a: vault-relative path to the first note.
        path_b: vault-relative path to the second note.
        sync: if True (default), re-index both notes so changes are reflected in search.
    """
    path_a = _normalize_path(path_a)
    path_b = _normalize_path(path_b)
    log.info(f"create_backlink — {path_a} <-> {path_b}")
    try:
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
    except Exception as e:
        log_error(log, f"create_backlink FAILED: {path_a} <-> {path_b}", exc=e)
        return f"Error: {e}"


def sync_index(folder: str | None = None, subject: str | None = None) -> str:
    """Re-run the full indexer pipeline. Clears the embedding cache and BM25 index.

    Args:
        folder: If set, only re-index notes under this vault-relative folder path.
               Pass ``None`` (omit) to re-index the entire vault.
        subject: If set, find all notes mentioning this entity and force-re-index
                them (re-extract entities, relationships, and summaries via LLM).
                Mutually exclusive with ``folder``.
    """
    if subject:
        log.info(f"sync_index — re-indexing notes mentioning subject={subject!r}")
        try:
            found = entity_store.search(subject)
            paths = sorted(set(n["path"] for n in found))
            if not paths:
                return f"No notes found mentioning '{subject}'."
            count = 0
            for path in paths:
                if indexer.index_note(path, force=True):
                    count += 1
            log.info(f"sync_index — subject={subject!r}: {count}/{len(paths)} notes re-indexed")
            return f"Re-indexed {count} notes mentioning '{subject}'."
        except Exception as e:
            log_error(log, f"sync_index FAILED (subject={subject!r})", exc=e)
            return f"Error: {e}"

    log.info(f"sync_index — starting{' (folder=' + folder + ')' if folder else ''}")
    try:
        indexer.run_index(folder=folder)
        entity_store.rebuild()
        llm_client.clear_embed_cache()
        keyword_search.ensure_index()  # force BM25 rebuild
        log.info("sync_index — caches cleared")
        log.info("sync_index — complete")
        return f"Index sync complete{' for folder: ' + folder if folder else ''}. Check indexer.log for details."
    except Exception as e:
        log_error(log, "sync_index FAILED", exc=e)
        return f"Error: {e}"


def switch_embedding_model(model_name: str) -> str:
    """Switch the embedding model at runtime.

    Updates the config, clears the embed cache, resets the ChromaDB collection,
    and re-indexes the entire vault with the new model.

    The model must already exist in Ollama (pull it first with ``ollama pull <name>``).
    """
    log.info(f"switch_embedding_model — {model_name}")
    try:
        # Verify the model exists in Ollama
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

        log.info(f"switch_embedding_model — switched {old_model} -> {model_name}")
        return f"Switched embedding model: {old_model} → {model_name}. Vault re-indexed."
    except Exception as e:
        log_error(log, f"switch_embedding_model FAILED: {model_name}", exc=e)
        return f"Error: {e}"


def add_note_to_subject(
    subject: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    sync: bool = True,
    reindex_matches: bool = True,
) -> str:
    """Add a note to a subject, auto-linking it into the wiki-link graph.

    Creates the note under ``Subjects/{subject}/{title}.md`` with YAML
    frontmatter tagged with the subject name plus any extra tags, and a
    ``[[{subject}]]`` wiki-link to the subject hub note.

    The subject hub note (``Subjects/{subject}/{subject}.md``) is created
    automatically if it doesn't exist. A mutual backlink is added so the
    new note and hub note are connected in the wiki-link graph.

    The subject is also registered as an entity (type ``"Concept"``) so it
    appears in entity searches.

    Args:
        subject: subject name (e.g. ``"Maria"``). Used as folder name, tag,
            and entity name.
        title: note title (e.g. ``"First Impression"``). Used as filename
            (without ``.md``).
        content: Markdown body content. Frontmatter will be auto-generated
            — do not include a ``---`` header yourself.
        tags: optional extra tags (e.g. ``["poem", "church"]``). The subject
            tag is added automatically.
        sync: if True (default), re-index the new note (and hub if newly created)
            so they are immediately searchable. Set to False when adding
            multiple notes to batch indexing into a single sync call.
        reindex_matches: if True (default), scan the vault for notes that mention
            the subject name and force-re-index them so they pick up the new
            entity from the LLM.

    Returns:
        A confirmation message with the note path.
    """
    log.info(f"add_note_to_subject — subject={subject}, title={title}, tags={tags}")

    # Sanitize
    safe_subject = re.sub(r'[<>:"/\\|?*]', "", subject).strip() or "untitled"
    safe_title = re.sub(r'[<>:"/\\|?*]', "", title).strip() or "untitled"
    safe_subject_lower = safe_subject.casefold().replace(" ", "-")

    folder = f"Subjects/{safe_subject}"
    hub_path = f"{folder}/{safe_subject}.md"
    note_path = f"{folder}/{safe_title}.md"

    # Collect tags
    all_tags = [safe_subject_lower]
    if tags:
        for t in tags:
            t_clean = str(t).strip().lower().replace(" ", "-")
            if t_clean and t_clean not in all_tags:
                all_tags.append(t_clean)

    try:
        # 1. Ensure hub note exists
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

        # 2. Write the new note with frontmatter + content + wiki-link
        tag_lines = "\n".join(f"  - {t}" for t in all_tags)
        body = content.strip()
        backlink = f"[[{safe_subject}]]"
        if backlink not in body:
            body = body + f"\n\n{backlink}"
        new_note_content = f"---\ntags:\n{tag_lines}\n---\n\n{body}"
        obsidian_client.put_note(note_path, new_note_content)
        graph_store.register_title(note_path)

        # 3. Add backlink from hub note to new note
        hub_link = f"[[{safe_title}]]"
        if hub_link not in hub_content:
            hub_content = hub_content.rstrip() + f"\n\n{hub_link}"
            obsidian_client.put_note(hub_path, hub_content)

        # 4. Add edges to the in-memory wiki-link graph
        graph_store.add_edge(note_path, hub_path)
        graph_store.add_edge(hub_path, note_path)
        graph_store.flush()

        # 5. Register subject as an entity (if not already)
        entity_store.add_manual_entity(safe_subject, "Concept", aliases=[safe_subject])

        # 6. Add entity→note edge for graph
        if not graph_store.has_entity_edge("Concept", safe_subject, note_path):
            graph_store.add_entity_edge("Concept", safe_subject, note_path)
            graph_store.flush()

        # 7. Re-index the new note (and hub if newly created) so they're searchable
        if sync:
            indexer.index_note(note_path)
            if not hub_exists:
                indexer.index_note(hub_path)

        # 8. Scan vault for existing notes mentioning the subject and force-re-index
        reindexed = 0
        if reindex_matches:
            for p in _find_notes_mentioning(safe_subject):
                if p == note_path or p == hub_path:
                    continue
                if indexer.index_note(p, force=True):
                    reindexed += 1

        tag_summary = ", ".join(all_tags)
        log.info(f"add_note_to_subject — written: {note_path}, tags=[{tag_summary}]")
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

    except Exception as e:
        log_error(log, "add_note_to_subject FAILED", exc=e,
                  subject=subject, title=title)
        return f"Error: {e}"


__all_tools__ = [
    read_note,
    write_note,
    list_all_notes,
    list_folder,
    list_folder_deep,
    search_by_tags,
    read_note_by_title,
    add_tags,
    remove_tags,
    set_tags,
    batch_tag_notes,
    create_backlink,
    add_note_to_subject,
    sync_index,
    switch_embedding_model,
]
