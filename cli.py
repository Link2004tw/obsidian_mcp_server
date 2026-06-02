"""CLI wrapper for obsidian-ai MCP server.

Spawns the MCP server as a subprocess and communicates via stdio,
so all commands go through the same code path as any MCP agent would.
"""

import argparse
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── helpers ─────────────────────────────────────────────────────────


def _clean_args(d: dict) -> dict:
    """Remove keys where the value is None or an empty string (unset optional args)."""
    return {k: v for k, v in d.items() if v is not None and v != ""}


# ── MCP call ────────────────────────────────────────────────────────


async def _call_tool(tool_name: str, args: dict) -> None:
    """Connect to the MCP server via stdio, call *tool_name*, print the result."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "obsidian_ai.mcp_server"],
    )
    try:
        async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, args)

                # result.content is a list of TextContent | ImageContent | ...
                if result.content:
                    for item in result.content:
                        text = getattr(item, "text", None)
                        if text is not None:
                            print(text)
                        else:
                            print(str(item))
                else:
                    print("(empty result)")
    except FileNotFoundError:
        print(
            "Error: Could not start the MCP server. Make sure you're in the"
            " project directory and the package is installed.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error calling {tool_name}: {e}", file=sys.stderr)
        sys.exit(1)


# ── subcommands ─────────────────────────────────────────────────────


def cmd_watch():
    """Start the file watcher directly (not via MCP — it's a long-running process)."""
    from obsidian_ai.indexer import watch
    print("Starting file watcher (Ctrl+C to stop)...")
    watch()


def _dispatch(args) -> tuple[str, dict]:
    """Return (tool_name, tool_args) for the given parsed args.

    Uses a dispatch dict of callables so that ``args`` attributes are only
    accessed for the command being run (not eagerly for all commands).
    """
    dispatch = {
        "search": lambda: (
            "search_notes",
            _clean_args({
                "query": args.query,
                "n": args.n,
                "tags": args.tags,
                "exclude_tags": args.exclude_tags,
                "folder": args.folder,
                "date_after": args.date_after,
                "date_before": args.date_before,
                "expand_query": args.expand,
                "keyword_weight": args.keyword_weight,
                "min_similarity": args.min_similarity,
                "diversity_penalty": args.diversity_penalty,
            }),
        ),
        "read": lambda: ("read_note", {"path": args.path}),
        "write": lambda: ("write_note", {"path": args.path, "content": args.content}),
        "list-all": lambda: ("list_all_notes", {}),
        "list-folder": lambda: ("list_folder", {"folder_path": args.folder_path}),
        "list-folder-deep": lambda: ("list_folder_deep", {"folder_path": args.folder_path}),
        "read-by-title": lambda: (
            "read_note_by_title",
            _clean_args({"title": args.title, "folder_path": args.folder_path}),
        ),
        "search-by-tags": lambda: ("search_by_tags", {"tags": args.tags, "n": args.n}),
        "add-tags": lambda: ("add_tags", {"path": args.path, "tags": args.tags}),
        "create-backlink": lambda: (
            "create_backlink",
            {"path_a": args.path_a, "path_b": args.path_b},
        ),
        "stats": lambda: ("get_index_stats", {}),
        "sync": lambda: ("sync_index", {}),
        "ask": lambda: ("ask_vault", {"question": args.question, "top_k": args.top_k}),
        "tag-notes": lambda: ("tag_notes", {"query": args.query, "top_k": args.top_k}),
    }
    return dispatch[args.command]()


def main():
    parser = argparse.ArgumentParser(
        description="obsidian-ai CLI — talks to the MCP server via stdio"
    )
    sub = parser.add_subparsers(dest="command")

    # search
    p = sub.add_parser("search", help="Semantic search across indexed notes")
    p.add_argument("query", help="Search query")
    p.add_argument("-n", type=int, default=5, help="Number of results (default 5)")
    p.add_argument("--tags", nargs="*", default=None, help="Filter by tags (all must match)")
    p.add_argument("--exclude-tags", nargs="*", default=None, help="Exclude notes with these tags")
    p.add_argument("--folder", default=None, help="Filter by folder path")
    p.add_argument("--date-after", default=None, help="Filter by mtime after ISO date (2024-06-01)")
    p.add_argument("--date-before", default=None, help="Filter by mtime before ISO date (2024-12-31)")
    p.add_argument(
        "--expand",
        action="store_true",
        default=False,
        help="Expand query with LLM-generated synonyms for broader search",
    )
    p.add_argument(
        "--keyword-weight",
        type=float,
        default=0.0,
        help="BM25 keyword blend weight (0.0 = pure semantic, 0.3 = 30%% keyword)",
    )
    p.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Minimum similarity score threshold (0-1)",
    )
    p.add_argument(
        "--diversity-penalty",
        type=float,
        default=0.0,
        help="Diversity penalty factor (0.0=none, 0.5=moderate, 1.0=aggressive)",
    )

    # read
    p = sub.add_parser("read", help="Read full note content by path")
    p.add_argument("path", help="Vault-relative path to the note")

    # write
    p = sub.add_parser("write", help="Create or overwrite a note")
    p.add_argument("path", help="Vault-relative path to the note")
    p.add_argument("content", help="Full Markdown content (use quotes)")

    # list-all
    sub.add_parser("list-all", help="List all note paths in the vault")

    # list-folder
    p = sub.add_parser(
        "list-folder", help="List notes directly in a folder (non-recursive)"
    )
    p.add_argument("folder_path", help="Vault-relative folder path")

    # list-folder-deep
    p = sub.add_parser(
        "list-folder-deep", help="List all notes in a folder (recursive)"
    )
    p.add_argument("folder_path", help="Vault-relative folder path")

    # read-by-title
    p = sub.add_parser(
        "read-by-title", help="Look up a note by its filename (without .md)"
    )
    p.add_argument("title", help="Note title (e.g. README)")
    p.add_argument(
        "-f",
        "--folder-path",
        default="",
        help="Folder to scope the search (e.g. Projects)",
    )

    # add-tags
    p = sub.add_parser("add-tags", help="Add tags to a note's YAML frontmatter")
    p.add_argument("path", help="Vault-relative path to the note")
    p.add_argument("tags", nargs="+", help="One or more tags to add")

    # create-backlink
    p = sub.add_parser(
        "create-backlink", help="Create mutual [[backlinks]] between two notes"
    )
    p.add_argument("path_a", help="Path to the first note")
    p.add_argument("path_b", help="Path to the second note")

    # search-by-tags
    p = sub.add_parser(
        "search-by-tags", help="Find notes by YAML frontmatter tags (all tags must match)"
    )
    p.add_argument("tags", nargs="+", help="One or more tags to search for")
    p.add_argument("-n", type=int, default=10, help="Max results (default 10)")

    # watch
    sub.add_parser("watch", help="Start file watcher daemon (direct, not via MCP)")

    # sync
    sub.add_parser("sync", help="Re-run the full indexer pipeline")

    # stats
    sub.add_parser("stats", help="Show index statistics")

    # ask
    p = sub.add_parser("ask", help="Ask a natural-language question about the vault")
    p.add_argument("question", help="Your question")
    p.add_argument("-k", "--top-k", type=int, default=3, help="Notes to retrieve (default 3)")

    # tag-notes
    p = sub.add_parser("tag-notes", help="Auto-suggest & apply tags for a query")
    p.add_argument("query", help="Search query to find relevant notes")
    p.add_argument("-k", "--top-k", type=int, default=5, help="Number of notes to tag (default 5)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # ── commands that bypass MCP (long-running or not exposed) ──
    if args.command == "watch":
        cmd_watch()
        return

    # ── dispatch to MCP tool ────────────────────────────────────
    tool_name, tool_args = _dispatch(args)
    asyncio.run(_call_tool(tool_name, tool_args))


if __name__ == "__main__":
    main()
