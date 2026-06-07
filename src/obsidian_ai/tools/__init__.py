"""Consolidated MCP tools — 9 tools that replace ~50 specialized ones."""

from . import _shared as _shared
from . import admin, ask, entities, graph, links, notes, tags, todo, tools
from ._tool_base import TOOL_MODULES

_TOOL_MODULES = [ask, notes, tags, links, graph, entities, todo, admin, tools]
TOOL_MODULES[:] = _TOOL_MODULES


def register_all(mcp):
    """Register all tool functions with a FastMCP instance."""
    for mod in _TOOL_MODULES:
        for fn in mod.__all_tools__:
            mcp.tool()(fn)
