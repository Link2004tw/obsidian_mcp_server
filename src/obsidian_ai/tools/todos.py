"""Todo management tools for the Obsidian MCP server."""

from .. import config
from .. import indexer
from .. import todos as _impl
from ..logger import get_logger, log_error
from ._shared import _normalize_path

log = get_logger("obsidian_ai.tools.todos")

_TODO_PATH: str | None = None


def _todo_path() -> str:
    global _TODO_PATH
    if _TODO_PATH is None:
        _TODO_PATH = config.todo_file
    return _TODO_PATH


def ensure_todo_file(sync: bool = True) -> str:
    """Create a default todos.md file in the vault if it doesn't exist. Use this to initialise the todo system when the todos file is missing.

    Args:
        sync: if True (default), re-index the created file so it becomes searchable in the vault index.

    Returns:
        A string message indicating whether the file was created or already existed.
    """
    log.info("ensure_todo_file")
    try:
        result = _impl.ensure_todos_file_exists()
        if sync:
            indexer.index_note(_todo_path())
        log.info(f"ensure_todo_file — {result}")
        return result
    except Exception as e:
        log_error(log, "ensure_todo_file FAILED", exc=e)
        return f"Error: {e}"


def get_todos(project: str = "", status: str = "", overdue: bool = False, blocked: bool = False, search: str = "", priority: str = "") -> list[dict]:
    """List todos, optionally filtered by project or status. Use this to show the user their task list, check what needs to be done, or query todos with specific filters.

    Args:
        project: filter by project name (case-sensitive). Empty string (default) returns todos from all projects.
        status: ``"pending"`` or ``"completed"``. Empty string (default) returns all statuses.
        overdue: if True, only return pending todos past their due date. Default False.
        blocked: if True, only return todos containing "blocked"/"blocking" in the task text or a "blocked" tag. Default False.
        search: free-text search against task description, tags, and project name (case-insensitive). Empty string (default) returns all matches.
        priority: ``"high"``, ``"medium"``, or ``"low"``. Empty string (default) returns all priorities.

    Returns:
        A list of todo dicts, each with keys: id, task, status, due, priority, tags, project. Returns an empty list on error.
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


def add_todo(project: str, task: str, due: str = "", priority: str = "", tags: list[str] | None = None, sync: bool = True) -> dict:
    """Add a new todo task to a project. Use this when the user wants to create a task with explicit fields rather than natural language.

    Args:
        project: project name (e.g. ``"Work"``, ``"Personal"``). The project is created automatically if it doesn't exist yet.
        task: the task description (required).
        due: optional due date in ``YYYY-MM-DD`` format. Empty string (default) means no due date.
        priority: ``"high"``, ``"medium"``, or ``"low"``. Empty string (default) means no priority set.
        tags: optional list of tag strings. None (default) means no tags.
        sync: if True (default), re-index the todos file so changes are reflected in vault search.

    Returns:
        The created todo dict (including its auto-generated ``id``), or an error dict with an ``error`` key on failure.
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
        if sync:
            indexer.index_note(_todo_path())
        log.info(f"add_todo — created {todo['id']}")
        return todo
    except Exception as e:
        log_error(log, "add_todo FAILED", exc=e)
        return {"error": str(e)}


def complete_todo(todo_id: str, sync: bool = True) -> dict:
    """Mark a todo as completed by its id. Use this when the user finishes a task and wants to check it off.

    Args:
        todo_id: the id of the todo (returned by ``get_todos`` or ``add_todo``).
        sync: if True (default), re-index the todos file so changes are reflected in vault search.

    Returns:
        The updated todo dict with status set to ``"completed"``, or an error dict (``{"error": ...}``) if the id is not found.
    """
    log.info(f"complete_todo — {todo_id}")
    try:
        todo = _impl.complete_todo(todo_id)
        if todo is None:
            return {"error": f"Todo not found: {todo_id}"}
        if sync:
            indexer.index_note(_todo_path())
        log.info(f"complete_todo — {todo_id} done")
        return todo
    except Exception as e:
        log_error(log, "complete_todo FAILED", exc=e)
        return {"error": str(e)}


def update_todo(todo_id: str, task: str = "", due: str = "", priority: str = "", tags: list[str] | None = None, project: str = "", status: str = "", sync: bool = True) -> dict:
    """Update one or more fields of an existing todo. Use this to edit a task's description, due date, priority, tags, project, or status. Only the provided fields are changed; omitted fields keep their current values.

    Args:
        todo_id: the id of the todo to update (required).
        task: new task description. Leave empty (default) to keep the current value.
        due: new due date in ``YYYY-MM-DD`` format. Leave empty (default) to keep the current value.
        priority: ``"high"``, ``"medium"``, or ``"low"``. Leave empty (default) to keep the current value.
        tags: new tags list. Pass an empty list ``[]`` to clear tags; pass None (default) to keep current tags.
        project: move the todo to a different project (e.g. ``"Work"``). Leave empty (default) to keep the current project.
        status: ``"pending"`` or ``"completed"``. Leave empty (default) to keep the current status. Use ``complete_todo`` for simple completion.
        sync: if True (default), re-index the todos file so changes are reflected in vault search.

    Returns:
        The updated todo dict, or an error dict (``{"error": ...}``) if the id is not found.
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
        if sync:
            indexer.index_note(_todo_path())
        log.info(f"update_todo — {todo_id} updated")
        return todo
    except Exception as e:
        log_error(log, "update_todo FAILED", exc=e)
        return {"error": str(e)}


def delete_todo(todo_id: str, sync: bool = True) -> dict:
    """Delete a todo by its id. Use this to permanently remove a task from the todo list.

    Args:
        todo_id: the id of the todo to delete (required).
        sync: if True (default), re-index the todos file so changes are reflected in vault search.

    Returns:
        A dict with ``{"success": True}`` on success, or ``{"success": False, "error": "..."}`` on failure (e.g. id not found).
    """
    log.info(f"delete_todo — {todo_id}")
    try:
        ok = _impl.delete_todo(todo_id)
        if not ok:
            return {"success": False, "error": f"Todo not found: {todo_id}"}
        if sync:
            indexer.index_note(_todo_path())
        log.info(f"delete_todo — {todo_id} deleted")
        return {"success": True}
    except Exception as e:
        log_error(log, "delete_todo FAILED", exc=e)
        return {"success": False, "error": str(e)}


def sync_todos(sync: bool = True) -> dict:
    """Recalculate todo counts in the todos.md frontmatter and rewrite the file. Use this to fix stale todo statistics (total, completed, pending, overdue counts) after manual edits to the todos file.

    Args:
        sync: if True (default), re-index the todos file so updated stats are reflected in vault search.

    Returns:
        A dict with the updated frontmatter stats, or an error dict on failure.
    """
    log.info("sync_todos")
    try:
        result = _impl.sync_todos()
        if sync:
            indexer.index_note(_todo_path())
        log.info(f"sync_todos — {result}")
        return result
    except Exception as e:
        log_error(log, "sync_todos FAILED", exc=e)
        return {"error": str(e)}


def get_todo_stats() -> dict:
    """Get aggregated statistics about all todos in the vault. Use this to give the user a dashboard-style overview of their task completion, overdue items, and distribution across projects and priorities.

    Returns:
        A dict with keys: total, completed, pending, overdue counts, per-project breakdown, per-priority breakdown, due-date stats, and tag frequency. Returns an error dict on failure.
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
    """Get todos filtered by priority level. Use this when you need to show the user only high/medium/low priority tasks, optionally narrowed to a specific project or status.

    Args:
        priority: ``"high"``, ``"medium"``, or ``"low"`` (required).
        project: optional project name to narrow results (e.g. ``"Work"``). Empty string (default) means all projects.
        status: ``"pending"`` or ``"completed"``. Empty string (default) means all statuses.

    Returns:
        A list of todo dicts matching the filters. Returns an empty list on error.
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


def add_todo_from_natural_language(text: str, sync: bool = True) -> dict:
    """Parse a natural language description into a structured todo using the LLM. Use this when the user describes a task conversationally (e.g. "buy groceries tomorrow high priority") and the LLM should infer the task, due date, priority, and project automatically.

    Examples: ``"buy groceries tomorrow high priority"`` → creates a todo with inferred task, due date, and priority. ``"review PR by friday"`` → creates a todo with a Friday due date.

    Args:
        text: free-form task description (required). The LLM will parse this to extract the task text, due date, priority, and project.
        sync: if True (default), re-index the todos file so changes are reflected in vault search.

    Returns:
        The created todo dict (including its auto-generated ``id``), or an error dict on failure.
    """
    log.info(f"add_todo_from_natural_language — {text[:80]}")
    try:
        todo = _impl.add_todo_from_natural_language(text)
        if sync:
            indexer.index_note(_todo_path())
        log.info(f"add_todo_from_natural_language — created {todo['id']}")
        return todo
    except Exception as e:
        log_error(log, "add_todo_from_natural_language FAILED", exc=e)
        return {"error": str(e)}


def suggest_task_priority(task: str) -> str:
    """Ask the LLM to suggest a priority level for a task description. Use this to help the user decide whether a task should be high, medium, or low priority based on its content.

    Args:
        task: the task description (required). The LLM will analyse the urgency and importance of the task.

    Returns:
        A string: ``"high"``, ``"medium"``, or ``"low"``. Defaults to ``"medium"`` if the LLM call fails.
    """
    log.info(f"suggest_task_priority — {task[:60]}")
    try:
        return _impl.suggest_task_priority(task)
    except Exception as e:
        log_error(log, "suggest_task_priority FAILED", exc=e)
        return "medium"


def suggest_due_date(task: str) -> str:
    """Ask the LLM to suggest a due date for a task description. Use this to help the user set a deadline based on the task content and implied urgency.

    Args:
        task: the task description (required). The LLM will infer a reasonable deadline from the text.

    Returns:
        A ``YYYY-MM-DD`` date string. Falls back to 7 days from today if the LLM call fails.
    """
    log.info(f"suggest_due_date — {task[:60]}")
    try:
        return _impl.suggest_due_date(task)
    except Exception as e:
        log_error(log, "suggest_due_date FAILED", exc=e)
        from datetime import date, timedelta
        return (date.today() + timedelta(days=7)).isoformat()


def suggest_task_splitting(task: str) -> list[str]:
    """Ask the LLM to split a large task into smaller, actionable sub-tasks. Use this when a task is too vague or complex to tackle directly and the user needs help breaking it down.

    Args:
        task: the task description to decompose (required).

    Returns:
        A list of sub-task strings (e.g. ``["Sub-task 1", "Sub-task 2", ...]``). Returns the original task as a single-element list if the LLM call fails.
    """
    log.info(f"suggest_task_splitting — {task[:60]}")
    try:
        return _impl.suggest_task_splitting(task)
    except Exception as e:
        log_error(log, "suggest_task_splitting FAILED", exc=e)
        return [task]


def get_overdue_summary() -> str:
    """Generate an LLM-powered summary of all overdue todos. Use this to provide the user with a natural-language overview of what tasks are past due, highlighting patterns, risks, and recommendations.

    Returns:
        A paragraph-style string summarising overdue tasks, including patterns, risks, and suggested next actions. Returns an error message on failure.
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
    """Estimate when all pending todos in a project will be completed, based on historical completion rate. Use this to answer questions like "when will I finish all my Work tasks?" by analysing past completion speed.

    Args:
        project: the project name (e.g. ``"Work"`` — required).
        days: the lookback window in days for calculating the historical completion rate (default 30). Larger values smooth out fluctuations.

    Returns:
        A ``YYYY-MM-DD`` date string predicting completion of the remaining pending tasks, or a message explaining why estimation isn't possible (e.g. no history or no pending tasks).
    """
    log.info(f"estimate_completion_date — project={project!r}, days={days}")
    try:
        return _impl.estimate_completion_date(project=project, days=days)
    except Exception as e:
        log_error(log, "estimate_completion_date FAILED", exc=e)
        return f"Error: {e}"


def get_todos_for_note(path: str) -> list[dict]:
    """Find todos that reference a specific note (by filename mention in task text or tags). Use this to show which tasks are associated with a given vault note.

    Args:
        path: vault-relative path of the note, e.g. ``"Notes/project.md"`` — this is a vault path, not a full filesystem path (required).

    Returns:
        A list of todo dicts that mention the given note. Returns an empty list if none match or on error.
    """
    path = _normalize_path(path)
    log.info(f"get_todos_for_note — {path}")
    try:
        return _impl.get_todos_for_note(path)
    except Exception as e:
        log_error(log, "get_todos_for_note FAILED", exc=e)
        return []


def get_notes_for_todo(todo_id: str) -> list[str]:
    """Find note paths referenced (via ``[[wiki-link]]``) in a todo's task description. Use this to discover which vault notes are linked from a specific task.

    Args:
        todo_id: the id of the todo (required).

    Returns:
        A list of vault-relative note paths that are wiki-linked in the todo's task text. Returns an empty list if none found or on error.
    """
    log.info(f"get_notes_for_todo — {todo_id}")
    try:
        return _impl.get_notes_for_todo(todo_id)
    except Exception as e:
        log_error(log, "get_notes_for_todo FAILED", exc=e)
        return []


def link_todo_to_notes(todo_id: str, note_paths: list[str], sync: bool = True) -> dict:
    """Link a todo to one or more notes by appending ``[[wiki-link]]`` references to its task text. Use this to create bidirectional associations between a task and related vault notes.

    Args:
        todo_id: the id of the todo to update (required).
        note_paths: list of vault-relative note paths to link, e.g. ``["Notes/project.md", "Notes/meeting.md"]`` — not full filesystem paths (required).
        sync: if True (default), re-index both the linked notes and the todos file so changes are reflected in vault search.

    Returns:
        The updated todo dict with wiki-links appended to its task text, or an error dict on failure.
    """
    note_paths = [_normalize_path(p) for p in note_paths]
    log.info(f"link_todo_to_notes — {todo_id} -> {note_paths}")
    try:
        result = _impl.link_todo_to_notes(todo_id, note_paths)
        if sync:
            for np in note_paths:
                indexer.index_note(np)
            indexer.index_note(_todo_path())
        return result
    except Exception as e:
        log_error(log, "link_todo_to_notes FAILED", exc=e)
        return {"error": str(e)}


def ask_vault_about_todo(todo_id: str) -> str:
    """Ask the LLM to analyse a specific todo by combining it with related notes from the vault. Use this to get insights, suggestions, or context for a particular task — e.g. "what do I need to know before starting this?"

    Args:
        todo_id: the id of the todo (required).

    Returns:
        An LLM-generated natural-language analysis with insights, suggestions, and relevant context drawn from linked vault notes. Returns an error message on failure.
    """
    log.info(f"ask_vault_about_todo — {todo_id}")
    try:
        return _impl.ask_vault_about_todo(todo_id)
    except Exception as e:
        log_error(log, "ask_vault_about_todo FAILED", exc=e)
        return f"Error: {e}"


def ask_vault_about_todos(query: str) -> str:
    """Ask a natural-language question about all todos in the vault. The LLM answers using the full todo list and aggregate statistics. Use this for high-level questions like "What's my most overdue project?" or "How many tasks are due this week?"

    Args:
        query: your free-form question about the task list, e.g. ``"What's my most overdue project?"`` or ``"Which tasks have no due date?"`` (required).

    Returns:
        An LLM-generated natural-language answer based on the full todo list and statistics. Returns an error message on failure.
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
