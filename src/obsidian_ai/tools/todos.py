"""Todo management tools for the Obsidian MCP server."""

from .. import todos as _impl
from ..logger import get_logger, log_error
from ._shared import _normalize_path

log = get_logger("obsidian_ai.tools.todos")


def ensure_todo_file() -> str:
    """Create a default todos.md file in the vault if it doesn't exist."""
    log.info("ensure_todo_file")
    try:
        result = _impl.ensure_todos_file_exists()
        log.info(f"ensure_todo_file — {result}")
        return result
    except Exception as e:
        log_error(log, "ensure_todo_file FAILED", exc=e)
        return f"Error: {e}"


def get_todos(project: str = "", status: str = "", overdue: bool = False, blocked: bool = False, search: str = "", priority: str = "") -> list[dict]:
    """List todos from todos.md, optionally filtered by project and/or status.

    Args:
        project: filter by project name (case-sensitive). Empty string = all projects.
        status: ``"pending"`` or ``"completed"``. Empty string = all statuses.
        overdue: if True, only return pending todos past their due date.
        blocked: if True, only return todos with "blocked"/"blocking" in the task text or a "blocked" tag.
        search: free-text search against task description, tags, and project name (case-insensitive).
        priority: ``"high"``, ``"medium"``, or ``"low"``. Empty string = all priorities.

    Returns:
        List of todo dicts, each with id, task, status, due, priority, tags, project.
    """
    log.info(f"get_todos — project={project!r}, status={status!r}, overdue={overdue}, blocked={blocked}, search={search!r}, priority={priority!r}")
    try:
        proj = project if project else None
        st = status if status else None
        pri = priority if priority else None
        todos = _impl.get_todos(project=proj, status=st, overdue=overdue, blocked=blocked, search=search, priority=pri)
        log.info(f"get_todos — {len(todos)} results")
        return todos
    except Exception as e:
        log_error(log, "get_todos FAILED", exc=e)
        return []


def add_todo(project: str, task: str, due: str = "", priority: str = "", tags: list[str] | None = None) -> dict:
    """Add a new todo task to a project.

    Args:
        project: project name (e.g. ``"Work"``, ``"Personal"``). Created if it doesn't exist.
        task: the task description.
        due: optional due date in ``YYYY-MM-DD`` format.
        priority: ``"high"``, ``"medium"``, or ``"low"``.
        tags: optional list of tag strings.

    Returns:
        The created todo dict with its assigned id.
    """
    log.info(f"add_todo — project={project!r}, task={task!r}")
    try:
        todo = _impl.add_todo(
            project=project,
            task=task,
            due=due if due else None,
            priority=priority if priority else None,
            tags=tags or None,
        )
        log.info(f"add_todo — created {todo['id']}")
        return todo
    except Exception as e:
        log_error(log, "add_todo FAILED", exc=e)
        return {"error": str(e)}


def complete_todo(todo_id: str) -> dict:
    """Mark a todo as completed by its id.

    Args:
        todo_id: the id of the todo (returned by get_todos or add_todo).

    Returns:
        The updated todo dict, or an error dict if the id is not found.
    """
    log.info(f"complete_todo — {todo_id}")
    try:
        todo = _impl.complete_todo(todo_id)
        if todo is None:
            return {"error": f"Todo not found: {todo_id}"}
        log.info(f"complete_todo — {todo_id} done")
        return todo
    except Exception as e:
        log_error(log, "complete_todo FAILED", exc=e)
        return {"error": str(e)}


def update_todo(todo_id: str, task: str = "", due: str = "", priority: str = "", tags: list[str] | None = None, project: str = "", status: str = "") -> dict:
    """Update one or more fields of an existing todo.

    Args:
        todo_id: the id of the todo to update.
        task: new task description (leave empty to keep current).
        due: new due date (leave empty to keep current).
        priority: new priority (leave empty to keep current).
        tags: new tags list (leave empty to keep current).
        project: move to a different project (leave empty to keep current).
        status: ``"pending"`` or ``"completed"`` (leave empty to keep current).

    Returns:
        The updated todo dict, or an error dict if the id is not found.
    """
    log.info(f"update_todo — {todo_id}")
    try:
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
            return {"error": f"Todo not found: {todo_id}"}
        log.info(f"update_todo — {todo_id} updated")
        return todo
    except Exception as e:
        log_error(log, "update_todo FAILED", exc=e)
        return {"error": str(e)}


def delete_todo(todo_id: str) -> dict:
    """Delete a todo by its id.

    Args:
        todo_id: the id of the todo to delete.

    Returns:
        A dict with ``success: true`` or ``success: false`` with an error message.
    """
    log.info(f"delete_todo — {todo_id}")
    try:
        ok = _impl.delete_todo(todo_id)
        if not ok:
            return {"success": False, "error": f"Todo not found: {todo_id}"}
        log.info(f"delete_todo — {todo_id} deleted")
        return {"success": True}
    except Exception as e:
        log_error(log, "delete_todo FAILED", exc=e)
        return {"success": False, "error": str(e)}


def sync_todos() -> dict:
    """Recalculate todo counts in the todos.md frontmatter and rewrite the file."""
    log.info("sync_todos")
    try:
        result = _impl.sync_todos()
        log.info(f"sync_todos — {result}")
        return result
    except Exception as e:
        log_error(log, "sync_todos FAILED", exc=e)
        return {"error": str(e)}


def get_todo_stats() -> dict:
    """Return aggregated statistics about all todos in the vault.

    Returns:
        Dict with total, completed, pending, overdue counts, per-project breakdown,
        per-priority breakdown, due-date stats, and tag frequency.
    """
    log.info("get_todo_stats")
    try:
        result = _impl.get_todo_stats()
        log.info(f"get_todo_stats — {result['total']} todos")
        return result
    except Exception as e:
        log_error(log, "get_todo_stats FAILED", exc=e)
        return {"error": str(e)}


def get_todos_by_priority(priority: str, project: str = "", status: str = "") -> list[dict]:
    """Return todos filtered by priority level.

    Args:
        priority: ``"high"``, ``"medium"``, or ``"low"``.
        project: optional project name to narrow results.
        status: ``"pending"`` or ``"completed"``.

    Returns:
        List of matching todo dicts.
    """
    log.info(f"get_todos_by_priority — priority={priority!r}")
    try:
        proj = project if project else None
        st = status if status else None
        todos = _impl.get_todos(project=proj, status=st, priority=priority)
        log.info(f"get_todos_by_priority — {len(todos)} results")
        return todos
    except Exception as e:
        log_error(log, "get_todos_by_priority FAILED", exc=e)
        return []


def add_todo_from_natural_language(text: str) -> dict:
    """Parse a natural language description into a structured todo using the LLM.

    Example: ``"buy groceries tomorrow high priority"`` → creates a todo
    with task, due date, and priority inferred by the LLM.

    Args:
        text: free-form task description (e.g. ``"review PR by friday"``).

    Returns:
        The created todo dict with its assigned id.
    """
    log.info(f"add_todo_from_natural_language — {text[:80]}")
    try:
        todo = _impl.add_todo_from_natural_language(text)
        log.info(f"add_todo_from_natural_language — created {todo['id']}")
        return todo
    except Exception as e:
        log_error(log, "add_todo_from_natural_language FAILED", exc=e)
        return {"error": str(e)}


def suggest_task_priority(task: str) -> str:
    """Use the LLM to suggest a priority level for a task description.

    Args:
        task: the task description.

    Returns:
        ``"high"``, ``"medium"``, or ``"low"``.
    """
    log.info(f"suggest_task_priority — {task[:60]}")
    try:
        return _impl.suggest_task_priority(task)
    except Exception as e:
        log_error(log, "suggest_task_priority FAILED", exc=e)
        return "medium"


def suggest_due_date(task: str) -> str:
    """Use the LLM to suggest a due date for a task description.

    Args:
        task: the task description.

    Returns:
        A ``YYYY-MM-DD`` date string.
    """
    log.info(f"suggest_due_date — {task[:60]}")
    try:
        return _impl.suggest_due_date(task)
    except Exception as e:
        log_error(log, "suggest_due_date FAILED", exc=e)
        from datetime import date, timedelta
        return (date.today() + timedelta(days=7)).isoformat()


def suggest_task_splitting(task: str) -> list[str]:
    """Use the LLM to split a large task into smaller sub-tasks.

    Args:
        task: the task to decompose.

    Returns:
        A list of sub-task strings.
    """
    log.info(f"suggest_task_splitting — {task[:60]}")
    try:
        return _impl.suggest_task_splitting(task)
    except Exception as e:
        log_error(log, "suggest_task_splitting FAILED", exc=e)
        return [task]


def get_overdue_summary() -> str:
    """Generate an LLM-powered summary of all overdue todos.

    Returns:
        A paragraph-style summary identifying patterns, risks, and recommendations.
    """
    log.info("get_overdue_summary")
    try:
        result = _impl.get_overdue_summary()
        log.info("get_overdue_summary — done")
        return result
    except Exception as e:
        log_error(log, "get_overdue_summary FAILED", exc=e)
        return f"Error generating overdue summary: {e}"


def estimate_completion_date(project: str, days: int = 30) -> str:
    """Estimate when all pending todos in a project will be completed,
    based on historical completion rate.

    Args:
        project: the project name (e.g. ``"Work"``).
        days: lookback window for estimating completion rate (default 30).

    Returns:
        A ``YYYY-MM-DD`` date string or a message if estimation isn't possible.
    """
    log.info(f"estimate_completion_date — project={project!r}, days={days}")
    try:
        return _impl.estimate_completion_date(project=project, days=days)
    except Exception as e:
        log_error(log, "estimate_completion_date FAILED", exc=e)
        return f"Error: {e}"


def get_todos_for_note(path: str) -> list[dict]:
    """Find todos that reference a specific note (by filename in task text or tags).

    Args:
        path: vault-relative path, e.g. ``"Notes/project.md"`` — not a full filesystem path.

    Returns:
        List of todo dicts that mention the note.
    """
    path = _normalize_path(path)
    log.info(f"get_todos_for_note — {path}")
    try:
        return _impl.get_todos_for_note(path)
    except Exception as e:
        log_error(log, "get_todos_for_note FAILED", exc=e)
        return []


def get_notes_for_todo(todo_id: str) -> list[str]:
    """Find note paths referenced (via [[wiki-link]]) in a todo's task description.

    Args:
        todo_id: the id of the todo.

    Returns:
        List of vault-relative note paths linked from the todo.
    """
    log.info(f"get_notes_for_todo — {todo_id}")
    try:
        return _impl.get_notes_for_todo(todo_id)
    except Exception as e:
        log_error(log, "get_notes_for_todo FAILED", exc=e)
        return []


def link_todo_to_notes(todo_id: str, note_paths: list[str]) -> dict:
    """Link a todo to one or more notes by appending [[wiki-links]] to its task text.

    Args:
        todo_id: the id of the todo to update.
        note_paths: list of vault-relative note paths to link (not full filesystem paths).

    Returns:
        The updated todo dict.
    """
    note_paths = [_normalize_path(p) for p in note_paths]
    log.info(f"link_todo_to_notes — {todo_id} -> {note_paths}")
    try:
        return _impl.link_todo_to_notes(todo_id, note_paths)
    except Exception as e:
        log_error(log, "link_todo_to_notes FAILED", exc=e)
        return {"error": str(e)}


def ask_vault_about_todo(todo_id: str) -> str:
    """Ask the LLM about a specific todo — provides insights, suggestions,
    or context by combining the todo with related notes from the vault.

    Args:
        todo_id: the id of the todo.

    Returns:
        LLM-generated analysis and suggestions.
    """
    log.info(f"ask_vault_about_todo — {todo_id}")
    try:
        return _impl.ask_vault_about_todo(todo_id)
    except Exception as e:
        log_error(log, "ask_vault_about_todo FAILED", exc=e)
        return f"Error: {e}"


def ask_vault_about_todos(query: str) -> str:
    """Ask a natural-language question about all todos in the vault.

    The LLM answers using the full todo list and aggregate statistics.

    Args:
        query: your question (e.g. ``"What's my most overdue project?"``).

    Returns:
        LLM-generated answer.
    """
    log.info(f"ask_vault_about_todos — {query[:80]}")
    try:
        return _impl.ask_vault_about_todos(query)
    except Exception as e:
        log_error(log, "ask_vault_about_todos FAILED", exc=e)
        return f"Error: {e}"


__all_tools__ = [
    ensure_todo_file,
    get_todos,
    add_todo,
    complete_todo,
    update_todo,
    delete_todo,
    sync_todos,
    get_todo_stats,
    get_todos_by_priority,
    add_todo_from_natural_language,
    suggest_task_priority,
    suggest_due_date,
    suggest_task_splitting,
    get_overdue_summary,
    estimate_completion_date,
    get_todos_for_note,
    get_notes_for_todo,
    link_todo_to_notes,
    ask_vault_about_todo,
    ask_vault_about_todos,
]
