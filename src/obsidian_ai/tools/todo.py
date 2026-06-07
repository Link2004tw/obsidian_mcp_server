"""Consolidated todo tool — manage tasks, suggestions, and analysis."""

import json as _json

from .. import config, indexer
from .. import todos as _impl
from ..logger import get_logger, log_error
from ._shared import _normalize_path
from ._tool_base import build_tool

log = get_logger("obsidian_ai.tools.todo")

_TODO_PATH: str | None = None


def _todo_path() -> str:
    global _TODO_PATH
    if _TODO_PATH is None:
        _TODO_PATH = config.todo_file
    return _TODO_PATH


_VALID_ACTIONS = {
    "list", "add", "complete", "update", "delete",
    "stats", "suggest_priority", "suggest_date", "suggest_split",
    "overdue_summary", "link", "ask",
}


def _handle_list(project: str = "", status: str = "", overdue: bool = False, blocked: bool = False, search: str = "", priority: str = "") -> str:
    proj = project if project else None
    st = status if status else None
    pri = priority if priority else None
    todos = _impl.get_todos(project=proj, status=st, overdue=overdue, blocked=blocked, search=search, priority=pri)
    return _json.dumps(todos, ensure_ascii=False, indent=2)


def _handle_add(project: str, task: str, due: str = "", priority: str = "", tags: list[str] | None = None, sync: bool = True) -> str:
    todo = _impl.add_todo(
        project=project,
        task=task,
        due=due if due else None,
        priority=priority if priority else None,
        tags=tags or None,
    )
    if sync:
        indexer.index_note(_todo_path())
    return _json.dumps(todo, ensure_ascii=False, indent=2)


def _handle_complete(todo_id: str, sync: bool = True) -> str:
    todo = _impl.complete_todo(todo_id)
    if todo is None:
        return _json.dumps({"error": f"Todo not found: {todo_id}"})
    if sync:
        indexer.index_note(_todo_path())
    return _json.dumps(todo, ensure_ascii=False, indent=2)


def _handle_update(todo_id: str, task: str = "", due: str = "", priority: str = "", tags: list[str] | None = None, project: str = "", status: str = "", sync: bool = True) -> str:
    kwargs: dict = {}
    if task:
        kwargs["task"] = task
    if due:
        kwargs["due"] = due
    if priority:
        kwargs["priority"] = priority
    if tags is not None:
        kwargs["tags"] = tags
    if project:
        kwargs["project"] = project
    if status:
        kwargs["status"] = status
    todo = _impl.update_todo(todo_id, **kwargs)
    if todo is None:
        return _json.dumps({"error": f"Todo not found: {todo_id}"})
    if sync:
        indexer.index_note(_todo_path())
    return _json.dumps(todo, ensure_ascii=False, indent=2)


def _handle_delete(todo_id: str, sync: bool = True) -> str:
    ok = _impl.delete_todo(todo_id)
    if not ok:
        return _json.dumps({"success": False, "error": f"Todo not found: {todo_id}"})
    if sync:
        indexer.index_note(_todo_path())
    return _json.dumps({"success": True})


def _handle_stats() -> str:
    result = _impl.get_todo_stats()
    return _json.dumps(result, ensure_ascii=False, indent=2)


def _handle_suggest_priority(task: str) -> str:
    return _impl.suggest_task_priority(task)


def _handle_suggest_date(task: str) -> str:
    return _impl.suggest_due_date(task)


def _handle_suggest_split(task: str) -> str:
    sub_tasks = _impl.suggest_task_splitting(task)
    return _json.dumps(sub_tasks, ensure_ascii=False, indent=2)


def _handle_overdue_summary() -> str:
    return _impl.get_overdue_summary()


def _handle_link(todo_id: str, note_paths: list[str], sync: bool = True) -> str:
    note_paths = [_normalize_path(p) for p in note_paths]
    result = _impl.link_todo_to_notes(todo_id, note_paths)
    if sync:
        for np in note_paths:
            indexer.index_note(np)
        indexer.index_note(_todo_path())
    return _json.dumps(result, ensure_ascii=False, indent=2)


def _handle_ask(todo_id: str = "", query: str = "") -> str:
    if todo_id:
        return _impl.ask_vault_about_todo(todo_id)
    return _impl.ask_vault_about_todos(query)


_HANDLERS = {
    "list": _handle_list,
    "add": _handle_add,
    "complete": _handle_complete,
    "update": _handle_update,
    "delete": _handle_delete,
    "stats": _handle_stats,
    "suggest_priority": _handle_suggest_priority,
    "suggest_date": _handle_suggest_date,
    "suggest_split": _handle_suggest_split,
    "overdue_summary": _handle_overdue_summary,
    "link": _handle_link,
    "ask": _handle_ask,
}


@build_tool("todo")
def todo(
    action: str,
    project: str = "",
    task: str = "",
    due: str = "",
    priority: str = "",
    tags: list[str] | None = None,
    status: str = "",
    todo_id: str = "",
    note_paths: list[str] | None = None,
    query: str = "",
    overdue: bool = False,
    blocked: bool = False,
    search: str = "",
    sync: bool = True,
) -> str:
    """Manage tasks, get suggestions, and analyze your todo list.

    Args:
        action: ``list`` — list todos with optional filters (project, status, priority, etc.).
                ``add`` — create a new todo with explicit fields.
                ``complete`` — mark a todo as completed by id.
                ``update`` — update one or more fields of a todo.
                ``delete`` — permanently delete a todo by id.
                ``stats`` — get aggregated todo statistics (dashboard).
                ``suggest_priority`` — ask the LLM to suggest a priority for a task.
                ``suggest_date`` — ask the LLM to suggest a due date for a task.
                ``suggest_split`` — ask the LLM to split a large task into sub-tasks.
                ``overdue_summary`` — get an LLM summary of all overdue todos.
                ``link`` — link a todo to notes via [[wiki-links]].
                ``ask`` — ask the LLM about a specific todo or the entire task list.
        project: project name (for list, add).
        task: task description (for add, suggest_priority/suggest_date/suggest_split).
        due: due date in YYYY-MM-DD (for add, update).
        priority: ``"high"``, ``"medium"``, or ``"low"``.
        tags: list of tag strings (for add, update).
        status: ``"pending"`` or ``"completed"`` (for list, update).
        todo_id: the todo id (for complete, update, delete, link, ask).
        note_paths: list of note paths (for link).
        query: free-form question about todos (for ask without todo_id).
        overdue: if True, only show overdue (for list).
        blocked: if True, only show blocked (for list).
        search: free-text search (for list).
        sync: if True (default), re-index after changes.

    Returns:
        JSON data, a formatted string, or an LLM-generated answer.
    """
    if action not in _VALID_ACTIONS:
        valid = ", ".join(sorted(_VALID_ACTIONS))
        return f"Error: Invalid action '{action}'. Valid actions: {valid}"

    handler = _HANDLERS[action]
    try:
        if action == "list":
            return handler(project=project, status=status, overdue=overdue, blocked=blocked, search=search, priority=priority)
        elif action == "add":
            return handler(project=project, task=task, due=due, priority=priority, tags=tags, sync=sync)
        elif action == "complete":
            return handler(todo_id=todo_id, sync=sync)
        elif action == "update":
            return handler(todo_id=todo_id, task=task, due=due, priority=priority, tags=tags, project=project, status=status, sync=sync)
        elif action == "delete":
            return handler(todo_id=todo_id, sync=sync)
        elif action == "stats":
            return handler()
        elif action == "suggest_priority":
            return handler(task=task)
        elif action == "suggest_date":
            return handler(task=task)
        elif action == "suggest_split":
            return handler(task=task)
        elif action == "overdue_summary":
            return handler()
        elif action == "link":
            return handler(todo_id=todo_id, note_paths=note_paths or [], sync=sync)
        elif action == "ask":
            return handler(todo_id=todo_id, query=query)
        else:
            return f"Error: Unhandled action '{action}'"
    except Exception as e:
        log_error(log, f"todo — action={action} FAILED", exc=e)
        return f"Error: {e}"


__all_tools__ = [todo]
