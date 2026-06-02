"""Tests for mcp_server.py — MCP tools with mocked dependencies."""
from unittest.mock import MagicMock, patch, PropertyMock

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
    """_matches_where only supports $contains/$gte, not direct equality.
    Direct value comparisons are ignored (pass-through)."""
    meta = {"path": "a.md", "title": "A"}
    where = {"title": "B"}
    # No $contains or $gte op → condition passes
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


# ── add_tags tool ──────────────────────────────────────────────────


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_add_tags(mock_obsidian):
    mock_obsidian.get_note.return_value = "---\ntags:\n  - existing\n---\nBody"
    result = mcp_server.add_tags("test.md", ["new-tag"])
    assert "Tags added" in result
    mock_obsidian.get_note.assert_called_once_with("test.md")
    mock_obsidian.put_note.assert_called_once()


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_add_tags_error(mock_obsidian):
    mock_obsidian.get_note.side_effect = Exception("not found")
    result = mcp_server.add_tags("missing.md", ["tag"])
    assert "Error" in result


# ── remove_tags tool ───────────────────────────────────────────────


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_remove_tags(mock_obsidian):
    mock_obsidian.get_note.return_value = "---\ntags:\n  - keep\n  - remove\n---\nBody"
    result = mcp_server.remove_tags("test.md", ["remove"])
    assert "Tags removed" in result
    call_args = mock_obsidian.put_note.call_args
    content = call_args[0][1]
    assert "remove" not in content
    assert "keep" in content


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_remove_tags_error(mock_obsidian):
    mock_obsidian.get_note.side_effect = Exception("fail")
    result = mcp_server.remove_tags("x.md", ["tag"])
    assert "Error" in result


# ── set_tags tool ──────────────────────────────────────────────────


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_set_tags(mock_obsidian):
    mock_obsidian.get_note.return_value = "---\ntags:\n  - old\n---\nBody"
    result = mcp_server.set_tags("test.md", ["new1", "new2"])
    assert "Tags set" in result
    call_args = mock_obsidian.put_note.call_args
    content = call_args[0][1]
    assert "new1" in content
    assert "new2" in content
    assert "old" not in content


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_set_tags_error(mock_obsidian):
    mock_obsidian.get_note.side_effect = Exception("fail")
    result = mcp_server.set_tags("x.md", ["tag"])
    assert "Error" in result


# ── read_note tool ─────────────────────────────────────────────────


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_read_note(mock_obsidian):
    mock_obsidian.get_note.return_value = "# Hello\nContent here"
    result = mcp_server.read_note("test.md")
    assert "Hello" in result
    mock_obsidian.get_note.assert_called_once_with("test.md")


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_read_note_error(mock_obsidian):
    mock_obsidian.get_note.side_effect = Exception("404")
    result = mcp_server.read_note("missing.md")
    assert "Error" in result


# ── list_all_notes tool ────────────────────────────────────────────


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_list_all_notes(mock_obsidian):
    mock_obsidian.list_all_notes.return_value = ["a.md", "b.md"]
    result = mcp_server.list_all_notes()
    assert result == ["a.md", "b.md"]


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_list_all_notes_error(mock_obsidian):
    mock_obsidian.list_all_notes.side_effect = Exception("fail")
    result = mcp_server.list_all_notes()
    # Returns empty list on error, not an error string
    assert result == []


# ── list_folder tool ───────────────────────────────────────────────


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_list_folder(mock_obsidian):
    mock_obsidian.list_folder.return_value = ["notes/a.md", "notes/sub/"]
    result = mcp_server.list_folder("notes/")
    assert result == ["notes/a.md", "notes/sub/"]


# ── write_note tool ────────────────────────────────────────────────


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_write_note(mock_obsidian):
    result = mcp_server.write_note("new.md", "# Title\nContent")
    assert "Written" in result or "wrote" in result.lower() or "success" in result.lower()
    mock_obsidian.put_note.assert_called_once_with("new.md", "# Title\nContent")


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_write_note_error(mock_obsidian):
    mock_obsidian.put_note.side_effect = Exception("fail")
    result = mcp_server.write_note("x.md", "content")
    assert "Error" in result


# ── create_backlink tool ───────────────────────────────────────────


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_create_backlink(mock_obsidian):
    mock_obsidian.get_note.side_effect = ["# A\nContent A", "# B\nContent B"]
    result = mcp_server.create_backlink("a.md", "b.md")
    assert "backlink" in result.lower() or "linked" in result.lower()
    # Should have called put_note twice (once for each note)
    assert mock_obsidian.put_note.call_count == 2


@patch("obsidian_ai.mcp_server.obsidian_client")
def test_create_backlink_already_exists(mock_obsidian):
    """When [[b]] is already in note A, it should not be added again.
    Note: the function matches by display name ([[b]]), not full path ([[b.md]])."""
    mock_obsidian.get_note.side_effect = ["# A\n[[b]]\nContent", "# B\n[[a]]\nContent"]
    result = mcp_server.create_backlink("a.md", "b.md")
    assert "already" in result.lower() or "exist" in result.lower() or "linked" in result.lower()
    # put_note should NOT be called since links already exist
    mock_obsidian.put_note.assert_not_called()


# ── get_index_stats tool ───────────────────────────────────────────


@patch("obsidian_ai.mcp_server.chroma_store")
def test_get_index_stats(mock_chroma):
    mock_chroma.get_index_stats.return_value = {"total_chunks": 100, "unique_notes": 50}
    result = mcp_server.get_index_stats()
    assert "100" in result
    assert "50" in result


# ── _expand_query ──────────────────────────────────────────────────


@patch("obsidian_ai.mcp_server.llm_client")
def test_expand_query(mock_llm):
    mcp_server._expand_query.cache_clear()
    mock_llm.chat.return_value = "synonym one\nsynonym two\nrelated concept"
    result = mcp_server._expand_query("test query")
    assert len(result) == 3
    assert "synonym one" in result


@patch("obsidian_ai.mcp_server.llm_client")
def test_expand_query_empty(mock_llm):
    mcp_server._expand_query.cache_clear()
    mock_llm.chat.return_value = ""
    result = mcp_server._expand_query("query")
    assert result == []


# ── Entity Tools ────────────────────────────────────────────────────


@patch("obsidian_ai.mcp_server.chroma_store")
@patch("obsidian_ai.mcp_server.entity_store")
def test_search_entities_by_name(mock_entity_store, mock_chroma):
    mock_chroma._collection = None
    mock_entity_store.search.return_value = [
        {"path": "note1.md", "entity_name": "ESP32", "entity_type": "Hardware",
         "snippet": "using ESP32", "confidence": 0.95},
        {"path": "note2.md", "entity_name": "ESP32", "entity_type": "Hardware",
         "snippet": "ESP32 module", "confidence": 0.9},
    ]
    result = mcp_server.search_entities("ESP32")
    assert len(result) == 2
    assert result[0]["entity_name"] == "ESP32"
    assert result[0]["entity_type"] == "Hardware"


@patch("obsidian_ai.mcp_server.chroma_store")
@patch("obsidian_ai.mcp_server.entity_store")
def test_search_entities_empty(mock_entity_store, mock_chroma):
    mock_chroma._collection = None
    mock_entity_store.search.return_value = []
    result = mcp_server.search_entities("Nonexistent")
    assert result == []


@patch("obsidian_ai.mcp_server.chroma_store")
@patch("obsidian_ai.mcp_server.entity_store")
def test_search_entities_with_type_filter(mock_entity_store, mock_chroma):
    mock_chroma._collection = None
    mock_entity_store.search.return_value = [
        {"path": "note1.md", "entity_name": "Alice", "entity_type": "Person",
         "snippet": "Alice worked on", "confidence": 0.95},
    ]
    result = mcp_server.search_entities("Alice", entity_type="Person")
    assert len(result) == 1


@patch("obsidian_ai.mcp_server.chroma_store")
@patch("obsidian_ai.mcp_server.entity_store")
def test_get_note_entities(mock_entity_store, mock_chroma):
    mock_chroma._collection = None
    mock_entity_store.get_note_entities.return_value = [
        {"entity_name": "Alice", "entity_type": "Person", "confidence": 0.95},
        {"entity_name": "ProjectX", "entity_type": "Project", "confidence": 0.9},
    ]
    result = mcp_server.get_note_entities("note1.md")
    assert len(result) == 2
    assert result[0]["entity_name"] == "Alice"


def test_get_entity_types():
    result = mcp_server.get_entity_types()
    assert isinstance(result, list)
    assert "Person" in result
    assert "Hardware" in result
