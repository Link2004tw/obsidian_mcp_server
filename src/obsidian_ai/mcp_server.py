import os
import re
from datetime import datetime

from fastmcp import FastMCP

from . import chroma_store, config, indexer, llm_client, obsidian_client, pipelines
from .frontmatter import add_tags as fm_add_tags
from .frontmatter import parse as fm_parse
from .logger import get_logger, log_error

log = get_logger("obsidian_ai.mcp_server", log_file="mcp_calls.log")

mcp = FastMCP("obsidian-ai")

# ── filter builder ─────────────────────────────────────────────────


def _build_search_where(
    tags: list[str] | None = None,
    folder: str | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
) -> dict | None:
    """Build a ChromaDB ``where`` clause from optional search filters.
    Returns ``None`` when no filters are active."""
    conditions: list[dict] = []

    # Tags: $contains on tags_str with ",tag," delimiter
    if tags:
        for tag in tags:
            conditions.append({"tags_str": {"$contains": f",{tag},"}})

    # Folder: $contains on path (trailing slash ensures folder prefix match)
    if folder:
        f = folder.replace("\\", "/").strip("/")
        conditions.append({"path": {"$contains": f"{f}/"}})

    # Date range: mtime as Unix timestamp
    try:
        if date_after:
            ts = datetime.fromisoformat(date_after).timestamp()
            conditions.append({"mtime": {"$gte": ts}})
        if date_before:
            ts = datetime.fromisoformat(date_before).timestamp()
            conditions.append({"mtime": {"$lte": ts}})
    except ValueError as e:
        raise ValueError(f"Invalid ISO date format: {e}") from e

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── tools ──────────────────────────────────────────────────────────


@mcp.tool()
def get_index_stats() -> str:
    """Return index statistics (total chunks, unique notes, config info)."""
    log.info("get_index_stats")
    try:
        stats = chroma_store.get_index_stats()
        lines = [
            f"Unique notes indexed: {stats['unique_notes']}",
            f"Total chunks stored:  {stats['total_chunks']}",
            f"Embedding model:      {config.ollama_embed_model}",
            f"ChromaDB path:        {config.chroma_path}",
        ]
        result = "\n".join(lines)
        log.info(f"get_index_stats — {result.replace(chr(10), ' | ')}")
        return result
    except Exception as e:
        log_error(log, "get_index_stats FAILED", exc=e)
        return f"Error: {e}"


@mcp.tool()
def search_notes(
    query: str,
    n: int = 5,
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    folder: str | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
) -> list[dict]:
    """Search notes semantically with optional metadata filters.

    Filters (all optional):
    - ``tags`` — only notes having ALL of these YAML tags
    - ``exclude_tags`` — exclude notes having ANY of these tags
    - ``folder`` — only notes inside this vault-relative folder
    - ``date_after`` / ``date_before`` — ISO date strings (e.g. ``2024-06-01``) to filter by file mtime
    """
    log.info(f"search_notes — query={query}, n={n}, tags={tags}, exclude_tags={exclude_tags}, folder={folder}, date_after={date_after}, date_before={date_before}")
    try:
        where = _build_search_where(tags=tags, folder=folder, date_after=date_after, date_before=date_before)
        results = chroma_store.query(llm_client.embed(query), n=n * 3, where=where)
        seen: dict[str, bool] = {}
        deduped: list[dict] = []
        for r in results:
            if len(deduped) >= n:
                break
            path = r["metadata"]["path"]
            if path in seen:
                continue
            # Exclude_tags post-filter (not supported directly in ChromaDB where clause)
            if exclude_tags:
                tags_str = r["metadata"].get("tags_str", "")
                if any(f",{et}," in tags_str for et in exclude_tags):
                    continue
            seen[path] = True
            deduped.append({
                "path": path,
                "title": r["metadata"].get("title", ""),
                "chunk": r["metadata"].get("chunk", 0),
                "distance": r["distance"],
            })
        log.info(f"search_notes — {len(deduped)} results returned")
        return deduped
    except Exception as e:
        log_error(log, "search_notes FAILED", exc=e, query=query, n=n)
        return []


@mcp.tool()
def read_note(path: str) -> str:
    """Read the full content of a note by path."""
    log.info(f"read_note — {path}")
    try:
        content = obsidian_client.get_note(path)
        log.info(f"read_note — {path} — {len(content)} chars")
        return content
    except Exception as e:
        log_error(log, f"read_note FAILED: {path}", exc=e)
        return f"Error: {e}"


@mcp.tool()
def write_note(path: str, content: str) -> str:
    """Create or overwrite a note with the given content."""
    log.info(f"write_note — {path} — {len(content)} chars")
    try:
        obsidian_client.put_note(path, content)
        return f"Written: {path}"
    except Exception as e:
        log_error(log, f"write_note FAILED: {path}", exc=e, content_len=len(content))
        return f"Error: {e}"


@mcp.tool()
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


@mcp.tool()
def list_folder(folder_path: str) -> list[str]:
    """Return entries directly inside a specific folder (non-recursive).
    Includes both ``.md`` files and subdirectory names (with trailing ``/``)."""
    log.info(f"list_folder — {folder_path}")
    try:
        notes = obsidian_client.list_folder(folder_path)
        log.info(f"list_folder — {len(notes)} notes returned")
        return notes
    except Exception as e:
        log_error(log, f"list_folder FAILED: {folder_path}", exc=e)
        return []


@mcp.tool()
def list_folder_deep(folder_path: str) -> list[str]:
    """Return a list of note paths within a specific folder (recursive, includes subdirectories)."""
    log.info(f"list_folder_deep — {folder_path}")
    try:
        notes = obsidian_client.list_folder_deep(folder_path)
        log.info(f"list_folder_deep — {len(notes)} notes returned")
        return notes
    except Exception as e:
        log_error(log, f"list_folder_deep FAILED: {folder_path}", exc=e)
        return []


@mcp.tool()
def search_by_tags(tags: list[str], n: int = 10) -> list[dict]:
    """Find notes that have ALL of the given YAML frontmatter tags.
    Returns paths, titles, and matching tags.
    Note: requires tags to be stored during indexing — run `sync_index` first if you haven't re-indexed since this tool was added."""
    log.info(f"search_by_tags — tags={tags}, n={n}")
    try:
        results = chroma_store.search_by_tags(tags, n=n)
        out = []
        for r in results:
            # Parse tags_str back into a list for the response
            raw = r.get("tags_str", "")
            note_tags = [t for t in raw.strip(",").split(",") if t] if raw else []
            out.append({
                "path": r["path"],
                "title": r.get("title", ""),
                "tags": note_tags,
            })
        log.info(f"search_by_tags — {len(out)} results")
        return out
    except Exception as e:
        log_error(log, "search_by_tags FAILED", exc=e, tags=tags, n=n)
        return []


@mcp.tool()
def read_note_by_title(title: str, folder_path: str = "") -> str:
    """Look up a note by its title (filename without extension) and return the full content.
    Optionally scope to a specific folder to disambiguate duplicate titles.
    If multiple notes still match, returns all of them separated by headers."""
    log.info(f"read_note_by_title — title={title}, folder_path={folder_path or '(none)'}")
    try:
        matches = chroma_store.get_by_title(title)
        if not matches:
            msg = f"No note found with title: {title}"
            log.warning(f"read_note_by_title — {msg}")
            return f"Error: {msg}"

        paths = [m["path"] for m in matches]

        # Filter by folder if specified
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
            log.info(f"read_note_by_title — {title} → {paths[0]} — {len(content)} chars")
            return content

        # Multiple notes share this title — return all with labeled separators
        log.info(f"read_note_by_title — {title} — multiple matches ({len(paths)}): {paths}")
        parts = []
        for p in paths:
            content = obsidian_client.get_note(p)
            parts.append(f"─── {p} ───\n{content}")
        return f'Found {len(paths)} notes with title "{title}":\n\n' + "\n\n".join(parts)
    except Exception as e:
        log_error(log, f"read_note_by_title FAILED: title={title}, folder_path={folder_path}", exc=e)
        return f"Error: {e}"


@mcp.tool()
def add_tags(path: str, tags: list[str]) -> str:
    """Add tags to a note's YAML frontmatter. Creates frontmatter if absent."""
    log.info(f"add_tags — {path} — tags={tags}")
    try:
        content = obsidian_client.get_note(path)
        new_content = fm_add_tags(content, tags)
        obsidian_client.put_note(path, new_content)
        meta, _ = fm_parse(new_content)
        log.info(f"add_tags — {path} — final tags={meta.get('tags', [])}")
        return f"Tags added to {path}: {meta.get('tags', [])}"
    except Exception as e:
        log_error(log, f"add_tags FAILED: {path}", exc=e, tags=tags)
        return f"Error: {e}"


@mcp.tool()
def create_backlink(path_a: str, path_b: str) -> str:
    """Create mutual [[backlinks]] between two notes."""
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

        return f"Linked: {path_a} <-> {path_b}"
    except Exception as e:
        log_error(log, f"create_backlink FAILED: {path_a} <-> {path_b}", exc=e)
        return f"Error: {e}"


@mcp.tool()
def sync_index() -> str:
    """Re-run the full indexer pipeline. Returns count of notes indexed."""
    log.info("sync_index — starting")
    try:
        indexer.run_index()
        log.info("sync_index — complete")
        return "Index sync complete. Check indexer.log for details."
    except Exception as e:
        log_error(log, "sync_index FAILED", exc=e)
        return f"Error: {e}"


@mcp.tool()
def ask_vault(question: str, top_k: int = 3) -> str:
    """Ask a question about your Obsidian vault. Searches relevant notes and uses LLM to answer."""
    log.info(f"ask_vault — {question}")
    try:
        answer = pipelines.query(question, top_k=top_k)
        log.info(f"ask_vault — done, {len(answer)} chars")
        return answer
    except Exception as e:
        log_error(log, "ask_vault FAILED", exc=e, question=question)
        return f"Error: {e}"


@mcp.tool()
def tag_notes(query: str, top_k: int = 5) -> str:
    """Search notes matching a query and auto-suggest tags using LLM."""
    log.info(f"tag_notes — {query}")
    try:
        result = pipelines.tag_notes(query, top_k=top_k)
        log.info("tag_notes — done")
        return result
    except Exception as e:
        log_error(log, "tag_notes FAILED", exc=e, query=query)
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
