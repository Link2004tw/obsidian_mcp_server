"""Obsidian AI MCP server — tool registration, re-exports, and entry point."""

from fastmcp import FastMCP

from . import config
from .logger import get_logger
from .tools import register_all
from .tools._shared import (  # noqa: F401
    _EXPAND_QUERY_CACHE,
    _build_search_where,
    _expand_query,
    _get_vault_terminology,
    _group_by_note,
    _hybrid_search,
    _matches_where,
    _normalize_path,
    _rewrite_query,
    _truncate_snippet,
)
from .tools.admin import admin  # noqa: F401
from .tools.ask import ask  # noqa: F401
from .tools.entities import entities  # noqa: F401
from .tools.graph import graph  # noqa: F401
from .tools.links import links  # noqa: F401
from .tools.notes import notes  # noqa: F401
from .tools.tags import tags  # noqa: F401
from .tools.todo import todo  # noqa: F401
from .tools.tools import tools  # noqa: F401

log = get_logger("obsidian_ai.mcp_server", log_file="mcp_calls.log")

mcp = FastMCP("obsidian-ai")
register_all(mcp)

if __name__ == "__main__":
    cfg_warnings = config.validate(verbose=True)
    if cfg_warnings:
        log.warning(f"Startup config validation found {len(cfg_warnings)} issue(s)")
    log.info("Starting MCP server")
    mcp.run()
