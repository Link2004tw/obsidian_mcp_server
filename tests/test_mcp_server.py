"""Tests for consolidated MCP tools with mocked dependencies."""
from unittest.mock import patch

from obsidian_ai import mcp_server

# ── _truncate_snippet ──────────────────────────────────────────────


def test_truncate_short():
    assert mcp_server._truncate_snippet("short") == "short"


def test_truncate_long():
    text = "a" * 500
    result = mcp_server._truncate_snippet(text, max_chars=100)
    assert len(result) == 103  # 100 + "..."
    assert result.endswith("...")


def test_truncate_exact():
    text = "a" * 300
    result = mcp_server._truncate_snippet(text, max_chars=300)
    assert result == text
    assert not result.endswith("...")


def test_truncate_empty():
    assert mcp_server._truncate_snippet("") == ""


def test_truncate_default_chars():
    text = "x" * 400
    result = mcp_server._truncate_snippet(text)
    assert len(result) == 303  # SNIPPET_MAX_CHARS=300 + "..."


# ── _matches_where ─────────────────────────────────────────────────


def test_matches_where_none():
    assert mcp_server._matches_where({"path": "a.md"}, None) is True


def test_matches_where_exact():
    meta = {"path": "a.md", "title": "A"}
    where = {"title": "A"}
    assert mcp_server._matches_where(meta, where) is True


def test_matches_where_no_match():
    meta = {"path": "a.md", "title": "A"}
    where = {"title": "B"}
    assert mcp_server._matches_where(meta, where) is True


def test_matches_where_and():
    meta = {"path": "a.md", "title": "A", "tags_str": ",python,"}
    where = {"$and": [{"title": "A"}, {"tags_str": {"$contains": ",python,"}}]}
    assert mcp_server._matches_where(meta, where) is True


def test_matches_where_and_fail():
    meta = {"path": "a.md", "title": "A", "tags_str": ",java,"}
    where = {"$and": [{"tags_str": {"$contains": ",python,"}}]}
    assert mcp_server._matches_where(meta, where) is False


def test_matches_where_missing_key():
    meta = {"path": "a.md"}
    where = {"title": "A"}
    assert mcp_server._matches_where(meta, where) is False


# ── _build_search_where ────────────────────────────────────────────


def test_build_search_where_empty():
    assert mcp_server._build_search_where() is None


def test_build_search_where_folder():
    where = mcp_server._build_search_where(folder="notes/")
    assert where == {"path": {"$contains": "notes/"}}


def test_build_search_where_single_tag():
    where = mcp_server._build_search_where(tags=["python"])
    assert where == {"tags_str": {"$contains": ",python,"}}


def test_build_search_where_multiple_tags():
    where = mcp_server._build_search_where(tags=["python", "ai"])
    assert "$and" in where
    assert len(where["$and"]) == 2


def test_build_search_where_dates():
    where = mcp_server._build_search_where(date_after="2024-01-01", date_before="2024-12-31")
    assert "$and" in where
    assert len(where["$and"]) == 2


def test_build_search_where_combined():
    where = mcp_server._build_search_where(tags=["x"], folder="docs/")
    assert "$and" in where
    assert len(where["$and"]) == 2


# ── notes tool: tags actions ────────────────────────────────────────


@patch("obsidian_ai.tools.tags.obsidian_client")
def test_tags_add(mock_obsidian):
    mock_obsidian.get_note.return_value = "---\ntags:\n  - existing\n---\nBody"
    result = mcp_server.tags(action="add", path="test.md", tags=["new-tag"])
    assert "Tags added" in result
    mock_obsidian.get_note.assert_called_once_with("test.md")
    mock_obsidian.put_note.assert_called_once()


@patch("obsidian_ai.tools.tags.obsidian_client")
def test_tags_add_error(mock_obsidian):
    mock_obsidian.get_note.side_effect = Exception("not found")
    result = mcp_server.tags(action="add", path="missing.md", tags=["tag"])
    assert "Error" in result


@patch("obsidian_ai.tools.tags.obsidian_client")
def test_tags_remove(mock_obsidian):
    mock_obsidian.get_note.return_value = "---\ntags:\n  - keep\n  - remove\n---\nBody"
    result = mcp_server.tags(action="remove", path="test.md", tags=["remove"])
    assert "Tags removed" in result
    call_args = mock_obsidian.put_note.call_args
    content = call_args[0][1]
    assert "remove" not in content
    assert "keep" in content


@patch("obsidian_ai.tools.tags.obsidian_client")
def test_tags_remove_error(mock_obsidian):
    mock_obsidian.get_note.side_effect = Exception("fail")
    result = mcp_server.tags(action="remove", path="x.md", tags=["tag"])
    assert "Error" in result


@patch("obsidian_ai.tools.tags.obsidian_client")
def test_tags_set(mock_obsidian):
    mock_obsidian.get_note.return_value = "---\ntags:\n  - old\n---\nBody"
    result = mcp_server.tags(action="set", path="test.md", tags=["new1", "new2"])
    assert "Tags set" in result
    call_args = mock_obsidian.put_note.call_args
    content = call_args[0][1]
    assert "new1" in content
    assert "new2" in content
    assert "old" not in content


@patch("obsidian_ai.tools.tags.obsidian_client")
def test_tags_set_error(mock_obsidian):
    mock_obsidian.get_note.side_effect = Exception("fail")
    result = mcp_server.tags(action="set", path="x.md", tags=["tag"])
    assert "Error" in result


# ── notes tool: read/write/list actions ──────────────────────────────


@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_read(mock_obsidian):
    mock_obsidian.get_note.return_value = "# Hello\nContent here"
    result = mcp_server.notes(action="read", path="test.md")
    assert "Hello" in result
    mock_obsidian.get_note.assert_called_once_with("test.md")


@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_read_error(mock_obsidian):
    mock_obsidian.get_note.side_effect = Exception("404")
    result = mcp_server.notes(action="read", path="missing.md")
    assert "Error" in result


@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_list(mock_obsidian):
    mock_obsidian.list_all_notes.return_value = ["a.md", "b.md"]
    result = mcp_server.notes(action="list")
    assert "a.md" in result
    assert "b.md" in result


@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_list_error(mock_obsidian):
    mock_obsidian.list_all_notes.side_effect = Exception("fail")
    result = mcp_server.notes(action="list")
    assert "Error" in result


@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_list_folder(mock_obsidian):
    mock_obsidian.list_folder.return_value = ["notes/a.md", "notes/sub/"]
    result = mcp_server.notes(action="list_folder", folder="notes/")
    assert "notes/a.md" in result


@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_write(mock_obsidian):
    result = mcp_server.notes(action="write", path="new.md", content="# Title\nContent")
    assert "Written" in result
    mock_obsidian.put_note.assert_called_once_with("new.md", "# Title\nContent")


@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_write_error(mock_obsidian):
    mock_obsidian.put_note.side_effect = Exception("fail")
    result = mcp_server.notes(action="write", path="x.md", content="content")
    assert "Error" in result


# ── links tool ───────────────────────────────────────────────────────


@patch("obsidian_ai.tools.links.obsidian_client")
def test_links_create(mock_obsidian):
    mock_obsidian.get_note.side_effect = ["# A\nContent A", "# B\nContent B"]
    result = mcp_server.links(action="create", path_a="a.md", path_b="b.md")
    assert "Linked" in result
    assert mock_obsidian.put_note.call_count == 2


@patch("obsidian_ai.tools.links.obsidian_client")
def test_links_create_already_exists(mock_obsidian):
    mock_obsidian.get_note.side_effect = ["# A\n[[b]]\nContent", "# B\n[[a]]\nContent"]
    result = mcp_server.links(action="create", path_a="a.md", path_b="b.md")
    assert "Linked" in result
    mock_obsidian.put_note.assert_not_called()


# ── admin tool: health check ─────────────────────────────────────────


@patch("obsidian_ai.tools.admin.llm_client")
@patch("obsidian_ai.tools.admin.obsidian_client")
@patch("obsidian_ai.tools.admin.chroma_store")
def test_admin_health(mock_chroma, mock_obsidian, mock_llm):
    mock_llm.check_health.return_value = {
        "ollama": {"status": "ok", "available": True},
    }
    mock_obsidian.list_all_notes.return_value = ["note1.md", "note2.md"]
    mock_chroma.count.return_value = 42
    result = mcp_server.admin(action="health")
    import json
    data = json.loads(result)
    assert data["ollama"]["status"] == "ok"
    assert data["note_count"] == 2
    assert data["chunk_count"] == 42


# ── admin tool: index stats ──────────────────────────────────────────


@patch("obsidian_ai.tools.admin.chroma_store")
@patch("obsidian_ai.tools.admin.llm_client")
@patch("obsidian_ai.tools.admin.entity_store")
@patch("obsidian_ai.tools.admin.config")
def test_admin_stats(mock_config, mock_entity_store, mock_llm, mock_chroma):
    mock_chroma.get_index_stats.return_value = {"total_chunks": 100, "unique_notes": 50}
    mock_llm.embed_cache_info.return_value = {"currsize": 10, "maxsize": 1000, "hits": 5, "misses": 3}
    mock_entity_store.stats.return_value = {"total_entities": 20, "total_mentions": 45}
    result = mcp_server.admin(action="stats")
    assert "100" in result
    assert "50" in result


# ── _expand_query ──────────────────────────────────────────────────


@patch("obsidian_ai.tools._shared.llm_client")
def test_expand_query(mock_llm):
    mcp_server._EXPAND_QUERY_CACHE.clear()
    mock_llm.chat.return_value = "synonym one\nsynonym two\nrelated concept"
    result = mcp_server._expand_query("test query")
    assert len(result) == 3
    assert "synonym one" in result


@patch("obsidian_ai.tools._shared.llm_client")
def test_expand_query_empty(mock_llm):
    mcp_server._EXPAND_QUERY_CACHE.clear()
    mock_llm.chat.return_value = ""
    result = mcp_server._expand_query("query")
    assert result == []


# ── entities tool ────────────────────────────────────────────────────


@patch("obsidian_ai.tools.entities.chroma_store")
@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_search_by_name(mock_entity_store, mock_chroma):
    mock_chroma._collection = None
    mock_entity_store.search.return_value = [
        {"path": "note1.md", "entity_name": "ESP32", "entity_type": "Hardware",
         "snippet": "using ESP32", "confidence": 0.95},
        {"path": "note2.md", "entity_name": "ESP32", "entity_type": "Hardware",
         "snippet": "ESP32 module", "confidence": 0.9},
    ]
    result = mcp_server.entities(action="search", entity_name="ESP32")
    import json
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["entity_name"] == "ESP32"


@patch("obsidian_ai.tools.entities.chroma_store")
@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_search_empty(mock_entity_store, mock_chroma):
    mock_chroma._collection = None
    mock_entity_store.search.return_value = []
    result = mcp_server.entities(action="search", entity_name="Nonexistent")
    assert "No notes found" in result


@patch("obsidian_ai.tools.entities.chroma_store")
@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_note_entities(mock_entity_store, mock_chroma):
    mock_chroma._collection = None
    mock_entity_store.get_note_entities.return_value = [
        {"entity_name": "Alice", "entity_type": "Person", "confidence": 0.95},
        {"entity_name": "ProjectX", "entity_type": "Project", "confidence": 0.9},
    ]
    result = mcp_server.entities(action="note_entities", path="note1.md")
    import json
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["entity_name"] == "Alice"


def test_entities_types():
    result = mcp_server.entities(action="types")
    import json
    data = json.loads(result)
    assert isinstance(data, list)
    assert "Person" in data
    assert "Hardware" in data


# ── Invalid action handling ──────────────────────────────────────────


def test_notes_invalid_action():
    result = mcp_server.notes(action="fly")
    assert "Error" in result
    assert "Invalid action" in result
