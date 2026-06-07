"""Tool discovery — lists all available MCP tools with descriptions and parameters."""

import inspect

from ..logger import get_logger
from ._tool_base import build_tool
from ._tool_base import TOOL_MODULES

log = get_logger("obsidian_ai.tools.tools")



@build_tool("tools")
def tools() -> str:
    """Return a complete listing of all available MCP tools, their descriptions, and parameter schemas.

    Use this at the start of a session to discover what tools are available —
    especially useful for models that cannot see all tools at once due to client limitations.

    Returns:
        JSON array of tool objects, each with ``name``, ``description``, and ``parameters``.
    """
    from ..logger import get_logger
    result = []
    for mod in TOOL_MODULES:
        for entry in mod.__all_tools__:
            fn = getattr(mod, entry) if isinstance(entry, str) else entry
            sig = inspect.signature(fn)
            doc = inspect.getdoc(fn) or ""
            desc = doc.split("\n\n")[0].strip() if doc else ""

            params = []
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                raw_type = param.annotation
                type_str = str(raw_type) if raw_type is not inspect.Parameter.empty else "any"
                is_required = param.default is inspect.Parameter.empty
                raw_default = None if is_required else param.default
                if not is_required and raw_default is not None and not isinstance(raw_default, (str, int, float, bool, type(None))):
                    raw_default = str(raw_default)
                params.append({
                    "name": param_name,
                    "type": type_str,
                    "required": is_required,
                    "default": raw_default,
                })

            result.append({
                "name": fn.__name__,
                "description": desc,
                "parameters": params,
            })

    result.sort(key=lambda t: t["name"])
    import json
    return json.dumps(result, indent=2, ensure_ascii=False)


__all_tools__ = [tools]
