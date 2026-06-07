"""Shared action-dispatch helper for consolidated MCP tools.

Each consolidated tool follows the same pattern::

    _VALID_ACTIONS = {"list", "add", ...}
    _HANDLERS = {
        "list": _handle_list,
        "add": _handle_add,
        ...
    }

    def my_tool(action: str, ...) -> Any:
        return _dispatch("my_tool", action, _VALID_ACTIONS, _HANDLERS, ...)

This keeps the MCP surface small (1 tool name per domain) while making
every operation explicit and predictable for small models.
"""

from functools import wraps

from ..logger import get_logger, log_error

log = get_logger("obsidian_ai.tools._tool_base")

# Populated by __init__.py after all modules are loaded.
# Used by tools.py to enumerate available tool modules.
TOOL_MODULES: list = []


def build_tool(name: str):
    """Decorator that wraps a handler-dispatch function with action validation,
    logging, and error handling.

    Usage::

        @build_tool("notes")
        def notes(action: str, ...) -> Any:
            ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            log.info("%s — action=%s, args=%s, kwargs=%s", name, kwargs.get("action", "?"), args, kwargs)
            try:
                result = fn(*args, **kwargs)
                log.info("%s — done", name)
                return result
            except Exception as e:
                log_error(log, f"{name} FAILED", exc=e)
                return f"Error: {e}"
        return wrapper
    return decorator
