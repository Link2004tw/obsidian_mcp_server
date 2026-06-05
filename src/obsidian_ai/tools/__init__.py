"""Tool modules for the Obsidian AI MCP server."""

import functools
import inspect
import time

from ..logger import get_logger

from . import _shared as _shared
from . import graph, misc, notes, search, todos

_TOOL_MODULES = [search, notes, graph, todos, misc]

_log = get_logger("obsidian_ai.mcp_server", log_file="mcp_calls.log")


def _log_tool_call(fn):
    """Decorator that logs every tool invocation with params and duration."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        params = {}
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if name == "content":
                params[name] = f"{len(value)} chars" if isinstance(value, str) else value
            elif isinstance(value, (str, int, float, bool)):
                params[name] = value
            elif isinstance(value, (list, tuple)) and len(value) < 10:
                params[name] = value
            else:
                params[name] = type(value).__name__
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        _log.info("%s — %s", fn.__name__, param_str)
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            _log.info("%s — done (%.2fs)", fn.__name__, elapsed)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            _log.error("%s — FAILED after %.2fs — %s", fn.__name__, elapsed, exc)
            raise
    return wrapper


def register_all(mcp):
    """Register all tool functions with a FastMCP instance."""
    for mod in _TOOL_MODULES:
        for entry in mod.__all_tools__:
            fn = getattr(mod, entry) if isinstance(entry, str) else entry
            wrapped = _log_tool_call(fn)
            mcp.tool()(wrapped)
