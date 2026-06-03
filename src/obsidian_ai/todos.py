import json
import os
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from . import config, llm_client, obsidian_client
from .frontmatter import parse as fm_parse
from .logger import get_logger

log = get_logger("obsidian_ai.todos", log_file="todos.log")

_NL_TODO_SYSTEM = (
    "You are a todo parsing assistant. Given a natural language task description, "
    "extract structured todo fields and return JSON with this exact shape:\n"
    '{"task": str, "project": str, "due": str or null, "priority": str or null, '
    '"tags": [str] or []}\n'
    "Rules:\n"
    "- task: concise description (mandatory).\n"
    "- project: infer a project name from context (e.g. \"Work\", \"Personal\", \"Learning\"). "
    "Default \"General\" if unclear.\n"
    "- due: YYYY-MM-DD if a date is mentioned (\"tomorrow\" → today+1, \"next week\" → today+7, "
    "\"next month\" → today+30). null if no date.\n"
    "- priority: \"high\", \"medium\", or \"low\". Infer from urgency words (\"urgent\", "
    "\"asap\" → high; no urgency → medium).\n"
    "- tags: relevant short tags inferred from context (max 3).\n"
    "Return ONLY the JSON object, no other text."
)

_SUGGEST_PRIORITY_SYSTEM = (
    "You are a task priority estimator. Given a task description, suggest a priority: "
    "\"high\", \"medium\", or \"low\".\n"
    "Return ONLY one word: high, medium, or low."
)

_SUGGEST_DUE_SYSTEM = (
    "You are a due-date estimator. Given a task description, suggest a reasonable due date.\n"
    "Return ONLY a YYYY-MM-DD date string (e.g. 2026-06-10). "
    "Base it on the current date and the implied urgency of the task. "
    f"Current date: {date.today().isoformat()}"
)

_SPLIT_TASK_SYSTEM = (
    "You are a task decomposition assistant. Given a large task, split it into 2-5 smaller, "
    "actionable sub-tasks.\n"
    'Return a JSON array of strings: ["sub-task 1", "sub-task 2", ...]\n'
    "Each sub-task should be a concrete, completable action."
)

_OVERDUE_SUMMARY_SYSTEM = (
    "You are a todo review assistant. Given a list of overdue todos grouped by project, "
    "produce a concise summary highlighting:\n"
    "- Which projects have the most overdue items\n"
    "- Common themes or patterns in overdue tasks\n"
    "- Actionable recommendations (e.g., which to prioritize, which to defer or drop)\n"
    "Be direct and practical. 2-4 paragraphs."
)

_TODO_LINE_RE = re.compile(r"^\s*-\s+\[([ x])\]\s+(.+?)(?:\s+\((.+?)\))?\s*$")
_META_RE = re.compile(r"(\w+):\s*([^,)]+)")
_PROJECT_HEADER_RE = re.compile(r"^##\s+(.+)$")


def _generate_id() -> str:
    return uuid.uuid4().hex[:8]


def _todo_path() -> str:
    return config.todo_file


def _default_todos_content() -> str:
    now = datetime.now().strftime("%Y-%m-%d")
    return (
        f"---\n"
        f"last_synced: {now}\n"
        f"total: 0\n"
        f"completed: 0\n"
        f"pending: 0\n"
        f"---\n"
        f"\n"
        f"# Todos\n"
        f"\n"
        f"## General\n"
        f"\n"
    )


def _parse_metadata(meta_str: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not meta_str:
        return result
    for m in _META_RE.finditer(meta_str):
        result[m.group(1).strip()] = m.group(2).strip()
    return result


def _format_metadata(meta: dict[str, str]) -> str:
    if not meta:
        return ""
    parts = [f"{k}: {v}" for k, v in meta.items()]
    return "(" + ", ".join(parts) + ")"


def _parse_todo_line(line: str) -> dict[str, Any] | None:
    m = _TODO_LINE_RE.match(line)
    if not m:
        return None
    status = "completed" if m.group(1) == "x" else "pending"
    task = m.group(2).strip()
    meta_str = m.group(3)
    meta = _parse_metadata(meta_str or "")
    tags_raw = meta.get("tags", "")
    return {
        "id": meta.get("id", _generate_id()),
        "task": task,
        "status": status,
        "due": meta.get("due"),
        "priority": meta.get("priority"),
        "tags": [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else [],
    }


def _format_todo_line(todo: dict[str, Any]) -> str:
    meta: dict[str, str] = {}
    meta["id"] = todo["id"]
    if todo.get("due"):
        meta["due"] = todo["due"]
    if todo.get("priority"):
        meta["priority"] = todo["priority"]
    if todo.get("tags"):
        meta["tags"] = ", ".join(todo["tags"])
    status_char = "x" if todo.get("status") == "completed" else " "
    meta_str = _format_metadata(meta)
    line = f"- [{status_char}] {todo['task']}"
    if meta_str:
        line += f" {meta_str}"
    return line


def _read_raw() -> str:
    try:
        return obsidian_client.get_note(_todo_path())
    except Exception:
        return ""


def _write_raw(content: str) -> None:
    obsidian_client.put_note(_todo_path(), content)


def _recalculate_frontmatter(todos: list[dict[str, Any]]) -> str:
    total = len(todos)
    completed = sum(1 for t in todos if t["status"] == "completed")
    pending = total - completed
    now = datetime.now().strftime("%Y-%m-%d")
    return (
        f"---\n"
        f"last_synced: {now}\n"
        f"total: {total}\n"
        f"completed: {completed}\n"
        f"pending: {pending}\n"
        f"---\n"
    )


def ensure_todos_file_exists() -> str:
    try:
        obsidian_client.get_note(_todo_path())
        return "todos.md already exists"
    except Exception:
        _write_raw(_default_todos_content())
        log.info("Created default todos.md")
        return "Created default todos.md"


def parse_todos() -> dict[str, Any]:
    raw = _read_raw()
    if not raw:
        return {
            "frontmatter": {"last_synced": "", "total": 0, "completed": 0, "pending": 0},
            "projects": {},
            "flat": [],
        }

    fm, body = fm_parse(raw)
    if not fm:
        fm = {"last_synced": "", "total": 0, "completed": 0, "pending": 0}
        body = raw

    lines = body.split("\n")
    projects: dict[str, list[dict]] = {}
    current_project = "General"

    for line in lines:
        header_m = _PROJECT_HEADER_RE.match(line)
        if header_m:
            current_project = header_m.group(1).strip()
            projects.setdefault(current_project, [])
            continue
        todo = _parse_todo_line(line)
        if todo:
            projects.setdefault(current_project, []).append(todo)

    flat = []
    for proj, todos_list in projects.items():
        for t in todos_list:
            flat.append({**t, "project": proj})

    return {
        "frontmatter": fm,
        "projects": projects,
        "flat": flat,
    }


def _rebuild_file(projects: dict[str, list[dict]]) -> str:
    all_todos = []
    for todos_list in projects.values():
        all_todos.extend(todos_list)

    frontmatter = _recalculate_frontmatter(all_todos)
    lines = [frontmatter, ""]
    for proj, todos_list in projects.items():
        lines.append(f"## {proj}")
        lines.append("")
        for t in todos_list:
            lines.append(_format_todo_line(t))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _is_blocked(todo: dict) -> bool:
    task = (todo.get("task") or "").lower()
    if "blocked" in task or "blocking" in task:
        return True
    return "blocked" in [t.lower() for t in (todo.get("tags") or [])]


def _score_search(todo: dict, query_lower: str) -> int:
    score = 0
    task = (todo.get("task") or "").lower()
    if query_lower in task:
        score += 2
    for tag in todo.get("tags") or []:
        if query_lower in tag.lower():
            score += 1
    proj = (todo.get("project") or "").lower()
    if query_lower in proj:
        score += 1
    return score


def get_todos(project: str | None = None, status: str | None = None, *,
              overdue: bool = False, blocked: bool = False, search: str = "",
              priority: str | None = None) -> list[dict]:
    ensure_todos_file_exists()
    data = parse_todos()
    results = list(data["flat"])
    if project:
        results = [t for t in results if t["project"] == project]
    if status:
        results = [t for t in results if t["status"] == status]
    if priority:
        results = [t for t in results if (t.get("priority") or "").lower() == priority.lower()]
    if overdue:
        today = date.today().isoformat()
        results = [t for t in results if t["status"] == "pending" and t.get("due") and t["due"] < today]
    if blocked:
        results = [t for t in results if _is_blocked(t)]
    if search:
        q = search.lower()
        results = [t for t in results if _score_search(t, q) > 0]
        results.sort(key=lambda t: _score_search(t, q), reverse=True)
    return results


def get_todo_stats() -> dict:
    data = parse_todos()
    todos = data["flat"]
    today = date.today().isoformat()

    total = len(todos)
    completed = sum(1 for t in todos if t["status"] == "completed")
    pending = total - completed
    overdue = sum(1 for t in todos if t["status"] == "pending" and t.get("due") and t["due"] < today)

    projects: dict[str, dict[str, int]] = {}
    for t in todos:
        proj = t["project"]
        if proj not in projects:
            projects[proj] = {"total": 0, "completed": 0, "pending": 0, "overdue": 0}
        projects[proj]["total"] += 1
        if t["status"] == "completed":
            projects[proj]["completed"] += 1
        else:
            projects[proj]["pending"] += 1
            if t.get("due") and t["due"] < today:
                projects[proj]["overdue"] += 1

    priorities: dict[str, int] = {}
    has_due = 0
    no_due_pending = 0
    tags: dict[str, int] = {}
    for t in todos:
        pri = t.get("priority")
        if pri:
            priorities[pri] = priorities.get(pri, 0) + 1
        if t.get("due"):
            has_due += 1
        elif t["status"] == "pending":
            no_due_pending += 1
        for tag in t.get("tags") or []:
            tags[tag] = tags.get(tag, 0) + 1

    return {
        "total": total,
        "completed": completed,
        "pending": pending,
        "overdue": overdue,
        "projects": projects,
        "priorities": priorities,
        "has_due_dates": has_due,
        "no_due_dates": no_due_pending,
        "tags": dict(sorted(tags.items(), key=lambda x: -x[1])),
        "last_synced": data["frontmatter"].get("last_synced", ""),
    }


def _find_todo_by_id(data: dict, todo_id: str) -> tuple[str, int, dict] | None:
    for proj, todos in data["projects"].items():
        for idx, t in enumerate(todos):
            if t["id"] == todo_id:
                return proj, idx, t
    return None


def add_todo(
    project: str,
    task: str,
    due: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    data = parse_todos()
    todo: dict[str, Any] = {
        "id": _generate_id(),
        "task": task,
        "status": "pending",
    }
    if due is not None:
        todo["due"] = due
    if priority is not None:
        todo["priority"] = priority
    if tags is not None:
        todo["tags"] = tags
    data["projects"].setdefault(project, []).append(todo)
    content = _rebuild_file(data["projects"])
    _write_raw(content)
    log.info(f"Added todo {todo['id']} to project '{project}'")
    return todo


def complete_todo(todo_id: str) -> dict | None:
    data = parse_todos()
    found = _find_todo_by_id(data, todo_id)
    if not found:
        return None
    proj, idx, todo = found
    todo["status"] = "completed"
    data["projects"][proj][idx] = todo
    content = _rebuild_file(data["projects"])
    _write_raw(content)
    log.info(f"Completed todo {todo_id}")
    return todo


def update_todo(todo_id: str, **kwargs: Any) -> dict | None:
    data = parse_todos()
    found = _find_todo_by_id(data, todo_id)
    if not found:
        return None
    proj, idx, todo = found

    new_project = kwargs.pop("project", None)
    if new_project is not None and new_project != proj:
        data["projects"][proj].pop(idx)
        if not data["projects"][proj]:
            del data["projects"][proj]
        for key, val in kwargs.items():
            if val is not None:
                todo[key] = val
        data["projects"].setdefault(new_project, []).append(todo)
    else:
        for key, val in kwargs.items():
            if val is not None:
                todo[key] = val
        data["projects"][proj][idx] = todo

    content = _rebuild_file(data["projects"])
    _write_raw(content)
    log.info(f"Updated todo {todo_id}: {kwargs}")
    return todo


def delete_todo(todo_id: str) -> bool:
    data = parse_todos()
    found = _find_todo_by_id(data, todo_id)
    if not found:
        return False
    proj, idx, _ = found
    data["projects"][proj].pop(idx)
    if not data["projects"][proj]:
        del data["projects"][proj]
    content = _rebuild_file(data["projects"])
    _write_raw(content)
    log.info(f"Deleted todo {todo_id}")
    return True


def sync_todos() -> dict:
    data = parse_todos()
    content = _rebuild_file(data["projects"])
    _write_raw(content)
    log.info("synced todos")
    return {"last_synced": datetime.now().strftime("%Y-%m-%d"), "status": "ok"}


# ── 6.5 — LLM-powered todo features ──────────────────────────────────


def add_todo_from_natural_language(text: str) -> dict:
    """Parse a natural language string into a structured todo using the LLM.

    Example: "buy groceries tomorrow high priority" → adds todo with
    task="buy groceries", due=tomorrow, priority="high".
    """
    messages = [
        {"role": "system", "content": _NL_TODO_SYSTEM},
        {"role": "user", "content": f"Parse: {text}"},
    ]
    response = llm_client.chat(messages, think=False)
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        parsed = json.loads(match.group()) if match else {"task": text, "project": "General"}
    task = parsed.get("task", text)
    project = parsed.get("project", "General")
    due = parsed.get("due") or None
    priority = parsed.get("priority") or None
    tags = parsed.get("tags") or None
    todo = add_todo(project=project, task=task, due=due, priority=priority, tags=tags)
    log.info("add_todo_from_natural_language — %s -> %s", text[:60], todo["id"])
    return todo


def suggest_task_priority(task: str) -> str:
    """Use LLM to suggest a priority for a task description."""
    messages = [
        {"role": "system", "content": _SUGGEST_PRIORITY_SYSTEM},
        {"role": "user", "content": f"Task: {task}"},
    ]
    response = llm_client.chat(messages, think=False).strip().lower()
    if response in ("high", "medium", "low"):
        return response
    return "medium"


def suggest_due_date(task: str) -> str:
    """Use LLM to suggest a due date for a task description.

    Returns a ``YYYY-MM-DD`` string.
    """
    messages = [
        {"role": "system", "content": _SUGGEST_DUE_SYSTEM},
        {"role": "user", "content": f"Task: {task}"},
    ]
    response = llm_client.chat(messages, think=False).strip()
    try:
        datetime.strptime(response, "%Y-%m-%d")
        return response
    except (ValueError, TypeError):
        return (date.today() + timedelta(days=7)).isoformat()


def suggest_task_splitting(task: str) -> list[str]:
    """Use LLM to split a large task into smaller sub-tasks.

    Returns a list of sub-task strings.
    """
    messages = [
        {"role": "system", "content": _SPLIT_TASK_SYSTEM},
        {"role": "user", "content": f"Task: {task}"},
    ]
    response = llm_client.chat(messages, think=False)
    try:
        subtasks = json.loads(response)
        if isinstance(subtasks, list):
            return [str(s) for s in subtasks]
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                subtasks = json.loads(match.group())
                if isinstance(subtasks, list):
                    return [str(s) for s in subtasks]
            except (json.JSONDecodeError, TypeError):
                pass
    return [task]


# ── 6.6 — Todo reporting & metrics ────────────────────────────────────


def get_overdue_summary() -> str:
    """Generate an LLM-powered summary of all overdue todos."""
    todos = get_todos(overdue=True)
    if not todos:
        return "No overdue todos. Great work!"

    groups: dict[str, list[dict]] = {}
    for t in todos:
        groups.setdefault(t["project"], []).append(t)

    lines = ["Overdue todos:"]
    for proj, items in groups.items():
        lines.append(f"\n## {proj} ({len(items)} overdue)")
        for t in items:
            due_str = f" (due: {t['due']})" if t.get("due") else ""
            pri_str = f" [{t['priority']}]" if t.get("priority") else ""
            lines.append(f"- {t['task']}{pri_str}{due_str}")
    context = "\n".join(lines)

    messages = [
        {"role": "system", "content": _OVERDUE_SUMMARY_SYSTEM},
        {"role": "user", "content": context},
    ]
    summary = llm_client.chat(messages, think=False)
    log.info("get_overdue_summary — %s chars", len(summary))
    return summary


def estimate_completion_date(project: str, days: int = 30) -> str:
    """Estimate completion date for all pending todos in a project based
    on historical completion rate over the given lookback period.

    Since we don't store completion timestamps, this uses a simple heuristic:
    if any todos are completed in the project, estimate based on ratio.
    Otherwise returns a default estimate.
    """
    all_todos = get_todos(project=project)
    if not all_todos:
        return f"No todos found in project '{project}'."
    total = len(all_todos)
    completed = sum(1 for t in all_todos if t["status"] == "completed")
    pending = total - completed
    if pending == 0:
        return f"All {total} todos in '{project}' are already completed."
    if completed == 0:
        return (date.today() + timedelta(days=days)).isoformat()
    rate = completed / max(total, 1)
    est_days = int(pending / rate) if rate > 0 else days
    est = date.today() + timedelta(days=est_days)
    return est.isoformat()


# ── 6.7 — Todo-vault integration ──────────────────────────────────────


def get_todos_for_note(path: str) -> list[dict]:
    """Find todos whose task or tags reference a given note path.

    Searches the full vault ``todos.md`` for any todo whose task text
    (or tags) contain the note's filename (without ``.md``) or the full path.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    all_todos = get_todos()
    results = []
    for t in all_todos:
        task = (t.get("task") or "").lower()
        if name.lower() in task or path.lower() in task:
            results.append(t)
            continue
        for tag in t.get("tags") or []:
            if name.lower() in tag.lower():
                results.append(t)
                break
    return results


def get_notes_for_todo(todo_id: str) -> list[str]:
    """Find note paths referenced (via [[wiki-link]] or path mention)
    in the given todo's task description or tags."""
    data = parse_todos()
    found = _find_todo_by_id(data, todo_id)
    if not found:
        return []
    _, _, todo = found
    task = todo.get("task", "")
    links = re.findall(r"\[\[([^\]]+)\]\]", task)
    paths = [link.split("|")[0].split("#")[0].strip() + ".md" for link in links]
    for tag in todo.get("tags") or []:
        if tag.endswith(".md"):
            paths.append(tag)
    return list(dict.fromkeys(paths))


def link_todo_to_notes(todo_id: str, note_paths: list[str]) -> dict:
    """Link a todo to one or more notes by adding wiki-links to its task.

    Appends ``[[NoteName]]`` references to the end of the todo's task text.
    Returns the updated todo dict.
    """
    data = parse_todos()
    found = _find_todo_by_id(data, todo_id)
    if not found:
        return {"error": f"Todo not found: {todo_id}"}
    proj, idx, todo = found
    existing_links = set(re.findall(r"\[\[([^\]]+)\]\]", todo.get("task", "")))
    for note_path in note_paths:
        name = os.path.splitext(os.path.basename(note_path))[0]
        if name not in existing_links:
            todo["task"] = todo["task"] + f" [[{name}]]"
            existing_links.add(name)
    data["projects"][proj][idx] = todo
    content = _rebuild_file(data["projects"])
    _write_raw(content)
    log.info("link_todo_to_notes — %s linked to %s", todo_id, note_paths)
    return todo


def ask_vault_about_todo(todo_id: str) -> str:
    """Ask the LLM to answer questions about a specific todo by combining
    the todo details with related notes from the vault."""
    data = parse_todos()
    found = _find_todo_by_id(data, todo_id)
    if not found:
        return f"Todo not found: {todo_id}"
    _, _, todo = found
    related_notes = get_notes_for_todo(todo_id)
    context = (
        f"Todo: {todo['task']}\n"
        f"Project: {todo.get('project', '')}\n"
        f"Status: {todo['status']}\n"
        f"Priority: {todo.get('priority', 'none')}\n"
        f"Due: {todo.get('due', 'none')}\n"
    )

    # Fetch note content for context
    note_texts = []
    for p in related_notes:
        try:
            raw = obsidian_client.get_note(p)
            truncated = llm_client.truncate_to_budget(raw, max_words=500)
            note_texts.append(f"## {p}\n{truncated}")
        except Exception:
            pass
    if note_texts:
        context += "\nRelated notes:\n" + "\n\n".join(note_texts)

    messages = [
        {"role": "system", "content": "You are a productivity assistant. Given a todo and related "
         "notes from the vault, provide insights, suggestions, or context about this task. "
         "Be concise and actionable."},
        {"role": "user", "content": context},
    ]
    response = llm_client.chat(messages, think=False)
    return response


def ask_vault_about_todos(query: str) -> str:
    """Ask a natural-language question about all todos.

    Uses ``get_todo_stats()`` for aggregate context and ``get_todos()``
    for the full list, then lets the LLM answer the question.
    """
    stats = get_todo_stats()
    todos = get_todos()
    summary_lines = [f"Total: {stats['total']}", f"Completed: {stats['completed']}",
                     f"Pending: {stats['pending']}", f"Overdue: {stats['overdue']}"]
    context = "Todo overview:\n" + "\n".join(summary_lines)
    context += "\n\nAll todos:\n"
    for t in todos:
        pri = f" [{t.get('priority', '')}]" if t.get("priority") else ""
        due = f" (due: {t['due']})" if t.get("due") else ""
        status = "✓" if t["status"] == "completed" else "○"
        context += f"- {status} {t['task']}{pri}{due} [{t['project']}]\n"

    messages = [
        {"role": "system", "content": "You are a productivity assistant with access to the user's "
         "todo list. Answer questions about their tasks, suggest priorities, identify patterns, "
         "and give practical advice. Be concise."},
        {"role": "user", "content": f"Context:\n\n{context}\n\n\nQuestion: {query}"},
    ]
    response = llm_client.chat(messages, think=False)
    log.info("ask_vault_about_todos — %s chars", len(response))
    return response
