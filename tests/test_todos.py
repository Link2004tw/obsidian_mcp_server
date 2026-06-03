"""Tests for todos.py — CRUD and utility functions with mocked Obsidian client."""
from unittest.mock import MagicMock, patch

SAMPLE_TODOS = """---
last_synced: "2026-06-03"
total: 4
completed: 1
pending: 3
overdue: 0
---

## General

- [ ] review quarterly report (id: todo-001, due: 2026-06-15, priority: high, tags: work)
- [x] buy groceries (id: todo-002, due: 2026-06-01, priority: low, tags: personal)
- [ ] fix leaking faucet (id: todo-003, due: , priority: medium, tags: home)
- [ ] plan summer vacation (id: todo-004, due: 2026-07-01, priority: low, tags: personal)
"""

EMPTY_TODOS = """---
last_synced: "2026-06-03"
total: 0
completed: 0
pending: 0
overdue: 0
---

"""


def _make_mock_get(content: str):
    return MagicMock(return_value=content)


def _track_put():
    written = []

    def _put_side_effect(path, content):
        written.append(content)

    m = MagicMock(side_effect=_put_side_effect)
    return m, written


def test_get_todos_all():
    from obsidian_ai import todos
    with patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)):
        results = todos.get_todos()
    assert len(results) == 4
    assert results[0]["task"] == "review quarterly report"
    assert results[0]["priority"] == "high"


def test_get_todos_by_project():
    from obsidian_ai import todos
    with patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)):
        results = todos.get_todos(project="General")
    assert len(results) == 4
    results2 = todos.get_todos(project="NonExistent")
    assert len(results2) == 0


def test_get_todos_filter_status():
    from obsidian_ai import todos
    with patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)):
        pending = todos.get_todos(status="pending")
        completed = todos.get_todos(status="completed")
    assert len(pending) == 3
    assert len(completed) == 1
    assert completed[0]["task"] == "buy groceries"


def test_get_todos_by_priority():
    from obsidian_ai import todos
    with patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)):
        high = todos.get_todos(priority="high")
        todos.get_todos(priority="low")
    assert len(high) == 1
    assert high[0]["task"] == "review quarterly report"


def test_get_todos_overdue():
    from obsidian_ai import todos
    with patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)):
        overdue = todos.get_todos(overdue=True)
    assert len(overdue) == 0


def test_get_todos_blocked():
    from obsidian_ai import todos
    with patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)):
        blocked = todos.get_todos(blocked=True)
    assert len(blocked) == 0


def test_get_todo_stats():
    from obsidian_ai import todos
    with patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)):
        stats = todos.get_todo_stats()
    assert stats["total"] == 4
    assert stats["completed"] == 1
    assert stats["pending"] == 3


def test_add_todo():
    from obsidian_ai import todos
    put_mock, written = _track_put()
    with (
        patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)),
        patch("obsidian_ai.obsidian_client.put_note", put_mock),
    ):
        result = todos.add_todo("General", "test new task", priority="medium")
    assert isinstance(result, dict)
    assert result["task"] == "test new task"
    assert "test new task" in written[0]


def test_complete_todo():
    from obsidian_ai import todos
    put_mock, written = _track_put()
    with (
        patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)),
        patch("obsidian_ai.obsidian_client.put_note", put_mock),
    ):
        result = todos.complete_todo("todo-001")
    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert "[x]" in written[0]


def test_delete_todo():
    from obsidian_ai import todos
    put_mock, written = _track_put()
    with (
        patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)),
        patch("obsidian_ai.obsidian_client.put_note", put_mock),
    ):
        result = todos.delete_todo("todo-001")
    assert result is True
    assert "review quarterly report" not in written[0]


def test_ensure_todos_file_exists():
    from obsidian_ai import todos
    put_mock, written = _track_put()
    with (
        patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(EMPTY_TODOS)),
        patch("obsidian_ai.obsidian_client.put_note", put_mock),
    ):
        result = todos.ensure_todos_file_exists()
    assert isinstance(result, str)


def test_estimate_completion_date():
    from obsidian_ai import todos
    with patch("obsidian_ai.obsidian_client.get_note", _make_mock_get(SAMPLE_TODOS)):
        result = todos.estimate_completion_date("General")
    assert isinstance(result, str)


def test_get_todos_for_note():
    from obsidian_ai import todos
    note_content = "Some text with [[review quarterly report]] and [[vacation]]"
    with patch("obsidian_ai.obsidian_client.get_note") as mock_get:
        mock_get.side_effect = lambda p: SAMPLE_TODOS if p == "todos.md" else note_content
        result = todos.get_todos_for_note("TestNote.md")
    assert isinstance(result, list)


def test_link_todo_to_notes():
    from obsidian_ai import todos
    with patch("obsidian_ai.obsidian_client.get_note") as mock_get:
        mock_get.side_effect = lambda p: {
            "todos.md": SAMPLE_TODOS,
            "Project.md": "# Project\n\nSome content\n",
        }.get(p, "")
        with patch("obsidian_ai.obsidian_client.put_note", MagicMock()):
            result = todos.link_todo_to_notes("todo-001", ["Project.md"])
    assert isinstance(result, dict)
    assert "error" not in result
    assert "[[Project]]" in result["task"]
