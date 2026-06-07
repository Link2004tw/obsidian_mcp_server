"""Tests for obsidian_client.py — list_all_notes and list_folder (with mocked HTTP)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from obsidian_ai import obsidian_client


@pytest.fixture(autouse=True)
def _clear_note_cache():
    obsidian_client.clear_note_cache()
    # Also wipe the on-disk cache to prevent stale data from affecting tests
    cache_path = obsidian_client._get_note_list_cache_path()
    import os
    if os.path.exists(cache_path):
        os.remove(cache_path)


def _mock_response(json_data: dict, status: int = 200) -> MagicMock:
    """Build a mock requests.Response object."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status >= 400:
        from requests import HTTPError
        http_error = HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_error
    return resp


# ── list_all_notes ──────────────────────────────────────────────────


@patch.object(obsidian_client, "requests")
def test_list_all_notes_empty(mock_requests):
    """Empty vault returns empty list."""
    mock_requests.get.return_value = _mock_response({"files": []})
    assert obsidian_client.list_all_notes() == []


@patch.object(obsidian_client, "requests")
def test_list_all_notes_flat(mock_requests):
    """Flat list of .md files at root."""
    mock_requests.get.return_value = _mock_response({
        "files": ["alpha.md", "beta.md", "gamma.md"]
    })
    result = obsidian_client.list_all_notes()
    assert result == ["alpha.md", "beta.md", "gamma.md"]


@patch.object(obsidian_client, "requests")
def test_list_all_notes_excludes_non_md(mock_requests):
    """Non-.md files are filtered out."""
    mock_requests.get.return_value = _mock_response({
        "files": ["note.md", "image.png", "data.csv"]
    })
    result = obsidian_client.list_all_notes()
    assert result == ["note.md"]


@patch.object(obsidian_client, "requests")
def test_list_all_notes_nested(mock_requests):
    """Recursive directory traversal returns paths with folder prefix."""
    # Call 1: root
    # Call 2: projects/
    # Call 3: projects/archive/
    mock_requests.get.side_effect = [
        _mock_response({"files": ["root.md", "projects/"]}),
        _mock_response({"files": ["active.md", "archive/"]}),
        _mock_response({"files": ["old.md"]}),
    ]
    result = obsidian_client.list_all_notes()
    assert result == [
        "root.md",
        "projects/active.md",
        "projects/archive/old.md",
    ]
    # Verify the URLs called
    urls = [call[0][0] for call in mock_requests.get.call_args_list]
    assert "/vault/" in urls[0]
    assert "/vault/projects" in urls[1]
    assert "/vault/projects/archive" in urls[2]


@patch.object(obsidian_client, "requests")
def test_list_all_notes_with_exclusions(mock_requests):
    """Entries matching EXCLUDE_PATTERNS are skipped."""
    mock_requests.get.return_value = _mock_response({
        "files": [
            "good.md",
            ".git",
            ".gsbak",
            "_gsdata_/",
            ".excalidraw.md",
        ]
    })
    result = obsidian_client.list_all_notes()
    assert result == ["good.md"]


@patch.object(obsidian_client, "requests")
def test_list_all_notes_only_directories(mock_requests):
    """A directory with only subdirectories and no .md files returns empty."""
    mock_requests.get.side_effect = [
        _mock_response({"files": ["docs/"]}),
        _mock_response({"files": ["images/"]}),
        _mock_response({"files": []}),
    ]
    result = obsidian_client.list_all_notes()
    assert result == []


# ── list_folder ─────────────────────────────────────────────────────


@patch.object(obsidian_client, "requests")
def test_list_folder_empty(mock_requests):
    """Empty folder returns empty list."""
    mock_requests.get.return_value = _mock_response({"files": []})
    assert obsidian_client.list_folder("empty") == []


@patch.object(obsidian_client, "requests")
def test_list_folder_flat(mock_requests):
    """List notes directly inside a folder (non-recursive)."""
    mock_requests.get.return_value = _mock_response({
        "files": ["doc1.md", "doc2.md"]
    })
    result = obsidian_client.list_folder("docs")
    assert result == ["docs/doc1.md", "docs/doc2.md"]


@patch.object(obsidian_client, "requests")
def test_list_folder_includes_subdirs(mock_requests):
    """Includes subdirectory entries (with trailing /) alongside .md files."""
    mock_requests.get.return_value = _mock_response({
        "files": ["readme.md", "sub/", "notes.md"]
    })
    result = obsidian_client.list_folder("projects")
    assert result == ["projects/readme.md", "projects/sub/", "projects/notes.md"]


@patch.object(obsidian_client, "requests")
def test_list_folder_nested_path(mock_requests):
    """Folder path with multiple levels works correctly."""
    mock_requests.get.side_effect = [
        _mock_response({"files": ["a.md", "b.md"]}),
    ]
    result = obsidian_client.list_folder("deeply/nested/path")
    assert result == ["deeply/nested/path/a.md", "deeply/nested/path/b.md"]
    # Verify correct URL
    call_url = mock_requests.get.call_args[0][0]
    assert "/vault/deeply/nested/path" in call_url


@patch.object(obsidian_client, "requests")
def test_list_folder_exclusions(mock_requests):
    """Excluded entries are filtered within a folder context."""
    mock_requests.get.return_value = _mock_response({
        "files": ["keep.md", ".git", ".excalidraw.md"]
    })
    result = obsidian_client.list_folder("myfolder")
    assert result == ["myfolder/keep.md"]


# ── list_folder — leading slash defense-in-depth ────────────────────


@patch.object(obsidian_client, "requests")
def test_list_folder_leading_slash(mock_requests):
    """Leading slash is stripped before passing to _walk_dir."""
    mock_requests.get.return_value = _mock_response({
        "files": ["note.md"]
    })
    result = obsidian_client.list_folder("/Projects")
    assert result == ["Projects/note.md"]
    call_url = mock_requests.get.call_args[0][0]
    assert "/vault/Projects" in call_url
    assert "//" not in call_url.replace("//localhost","")


@patch.object(obsidian_client, "requests")
def test_list_folder_root_slash(mock_requests):
    """list_folder('/') strips to root and lists immediate files and folders."""
    mock_requests.get.return_value = _mock_response({
        "files": ["root.md", "sub/"]
    })
    result = obsidian_client.list_folder("/")
    assert result == ["root.md", "sub/"]
    call_url = mock_requests.get.call_args[0][0]
    assert call_url.endswith("/vault/")


# ── list_folder_deep ────────────────────────────────────────────────


@patch.object(obsidian_client, "requests")
def test_list_folder_deep_empty(mock_requests):
    """Empty folder returns empty list."""
    mock_requests.get.return_value = _mock_response({"files": []})
    assert obsidian_client.list_folder_deep("empty") == []


@patch.object(obsidian_client, "requests")
def test_list_folder_deep_flat(mock_requests):
    """List notes inside a folder with no subdirectories."""
    mock_requests.get.return_value = _mock_response({
        "files": ["doc1.md", "doc2.md"]
    })
    result = obsidian_client.list_folder_deep("docs")
    assert result == ["docs/doc1.md", "docs/doc2.md"]


@patch.object(obsidian_client, "requests")
def test_list_folder_deep_recursive(mock_requests):
    """Recursively traverses subdirectories within the folder."""
    mock_requests.get.side_effect = [
        _mock_response({"files": ["readme.md", "sub/"]}),
        _mock_response({"files": ["deep.md", "notes.md"]}),
    ]
    result = obsidian_client.list_folder_deep("projects")
    assert result == ["projects/readme.md", "projects/sub/deep.md", "projects/sub/notes.md"]


@patch.object(obsidian_client, "requests")
def test_list_folder_deep_nested_path(mock_requests):
    """Deeply nested folder path works correctly."""
    mock_requests.get.side_effect = [
        _mock_response({"files": ["outer/"]}),
        _mock_response({"files": ["inner/"]}),
        _mock_response({"files": ["a.md", "b.md"]}),
    ]
    result = obsidian_client.list_folder_deep("deeply/nested/path")
    assert result == ["deeply/nested/path/outer/inner/a.md", "deeply/nested/path/outer/inner/b.md"]


@patch.object(obsidian_client, "requests")
def test_list_folder_deep_exclusions(mock_requests):
    """Excluded entries are filtered within the recursive walk."""
    mock_requests.get.side_effect = [
        _mock_response({"files": ["keep.md", ".git", ".gsbak", "sub/"]}),
        _mock_response({"files": ["nested.md", ".excalidraw.md"]}),
    ]
    result = obsidian_client.list_folder_deep("myfolder")
    assert result == ["myfolder/keep.md", "myfolder/sub/nested.md"]


# ── _list_dir — URL construction edge cases ─────────────────────────


def _get_list_dir_url(mock_requests, path: str) -> str:
    """Call _list_dir(path) and return the first URL called."""
    mock_requests.get.reset_mock()
    mock_requests.get.return_value = _mock_response({"files": []})
    obsidian_client._list_dir(path)
    return mock_requests.get.call_args[0][0]  # type: ignore


@patch.object(obsidian_client, "requests")
def test_list_dir_root(mock_requests):
    """Empty path produces URL ending in /vault/."""
    url = _get_list_dir_url(mock_requests, "")
    assert url.endswith("/vault/")


@patch.object(obsidian_client, "requests")
def test_list_dir_leading_slash(mock_requests):
    """Leading slash is stripped: /Courses → Courses → /vault/Courses."""
    url = _get_list_dir_url(mock_requests, "/Courses")
    assert url.endswith("/vault/Courses/")


@patch.object(obsidian_client, "requests")
def test_list_dir_root_slash(mock_requests):
    """Single slash / strips to empty → /vault/ (no double slash)."""
    url = _get_list_dir_url(mock_requests, "/")
    assert url.endswith("/vault/")


@patch.object(obsidian_client, "requests")
def test_list_dir_multi_leading_slash(mock_requests):
    """Multiple leading slashes are stripped: //Courses → Courses → /vault/Courses/."""
    url = _get_list_dir_url(mock_requests, "//Courses")
    assert url.endswith("/vault/Courses/")


@patch.object(obsidian_client, "requests")
def test_list_dir_trailing_slash(mock_requests):
    """Trailing slash is preserved: Courses/ → /vault/Courses/."""
    url = _get_list_dir_url(mock_requests, "Courses/")
    assert url.endswith("/vault/Courses/")


@patch.object(obsidian_client, "requests")
def test_list_dir_normal_path_no_double_slash(mock_requests):
    """Normal path produces clean URL with no double slashes."""
    url = _get_list_dir_url(mock_requests, "deeply/nested/folder")
    assert url.endswith("/vault/deeply/nested/folder/")
    assert "//vault" not in url
