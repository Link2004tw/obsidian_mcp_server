"""Tool modules for the Obsidian AI MCP server."""

from . import _shared as _shared
from . import graph, misc, notes, search, todos

_TOOL_MODULES = [search, notes, graph, todos, misc]


def register_all(mcp):
    """Register all tool functions with a FastMCP instance."""
    for mod in _TOOL_MODULES:
        for fn in mod.__all_tools__:
            mcp.tool()(fn)
