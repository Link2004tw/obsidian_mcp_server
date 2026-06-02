import re
import uuid
from datetime import datetime
from typing import Any

from . import config, obsidian_client
from .frontmatter import parse as fm_parse
from .logger import get_logger

log = get_logger("obsidian_ai.todos", log_file="todos.log")

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


def get_todos(project: str | None = None, status: str | None = None) -> list[dict]:
    data = parse_todos()
    results = list(data["flat"])
    if project:
        results = [t for t in results if t["project"] == project]
    if status:
        results = [t for t in results if t["status"] == status]
    return results


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
