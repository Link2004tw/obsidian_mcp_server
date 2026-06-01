import os
from fastmcp import FastMCP
from . import config
from . import obsidian_client
from . import llm_client
from . import chroma_store
from . import indexer
from . import pipelines
from .frontmatter import add_tags as fm_add_tags
from .logger import get_logger, log_error

log = get_logger("obsidian_ai.mcp_server", log_file="mcp_calls.log")

mcp = FastMCP("obsidian-ai")


@mcp.tool()
def search_notes(query: str, n: int = 5) -> list[dict]:
    """Search notes semantically. Returns top-k results with paths and snippets."""
    log.info(f"search_notes — query={query}, n={n}")
    try:
        embedding = llm_client.embed(query)
        results = chroma_store.query(embedding, n=n)
        seen = {}
        deduped = []
        for r in results:
            path = r["metadata"]["path"]
            if path not in seen:
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
def list_notes() -> list[str]:
    """Return a list of all note paths in the vault."""
    log.info("list_notes")
    try:
        notes = obsidian_client.list_notes()
        log.info(f"list_notes — {len(notes)} notes returned")
        return notes
    except Exception as e:
        log_error(log, "list_notes FAILED", exc=e)
        return []


@mcp.tool()
def add_tags(path: str, tags: list[str]) -> str:
    """Add tags to a note's YAML frontmatter. Creates frontmatter if absent."""
    log.info(f"add_tags — {path} — tags={tags}")
    try:
        content = obsidian_client.get_note(path)
        new_content = fm_add_tags(content, tags)
        obsidian_client.put_note(path, new_content)
        from .frontmatter import parse
        meta, _ = parse(new_content)
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
        if link_to_b not in content_a:
            content_a = content_a.rstrip() + f"\n\n{link_to_b}"
            obsidian_client.put_note(path_a, content_a)

        content_b = obsidian_client.get_note(path_b)
        if link_to_a not in content_b:
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
        log.info(f"tag_notes — done")
        return result
    except Exception as e:
        log_error(log, "tag_notes FAILED", exc=e, query=query)
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
