import json
import os
import threading
import time

import requests

from . import config

REQUEST_TIMEOUT = 30
_NOTE_LIST_CACHE: list[str] | None = None
_NOTE_LIST_CACHE_LOCK = threading.Lock()
_NOTE_LIST_CACHE_TTL = 300  # 5 minutes
_NOTE_LIST_CACHE_PATH: str | None = None


def _validate_path(path: str) -> str:
    """Validate and normalise a vault-relative path.

    Rejects paths containing ``..`` segments to prevent directory traversal.
    Strips leading slashes and ensures the path ends with ``.md`` for note operations.
    """
    path = path.lstrip("/")
    # Reject path traversal
    if ".." in path.split("/") or ".." in path.split(os.sep):
        raise ValueError(f"Invalid path (contains '..'): {path}")
    return path


def _is_excluded(entry: str) -> bool:
    return any(pattern in entry for pattern in config.EXCLUDE_PATTERNS)


def _base_url():
    return f"http://{config.obsidian_host}:{config.obsidian_port}"


def _headers():
    headers = {
        "Content-Type": "text/markdown",
    }
    if config.obsidian_api_key:
        headers["Authorization"] = f"Bearer {config.obsidian_api_key}"
    return headers


def _list_dir(path: str = "") -> list[str]:
    clean_path = _validate_path(path) if path else ""
    url = f"{_base_url()}/vault/{clean_path}" if clean_path else f"{_base_url()}/vault/"
    resp = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    entries = data.get("files", [])
    assert isinstance(entries, list)
    return entries


def _walk_dir(path: str = "") -> list[str]:
    results = []
    for entry in _list_dir(path):
        if _is_excluded(entry):
            continue
        entry = entry.lstrip("/")
        full_path = f"{path}/{entry}" if path else entry
        if entry.endswith("/"):
            results.extend(_walk_dir(full_path.rstrip("/")))
        elif entry.endswith(".md"):
            results.append(full_path)
    return results


def _get_note_list_cache_path() -> str:
    global _NOTE_LIST_CACHE_PATH
    if _NOTE_LIST_CACHE_PATH is not None:
        return _NOTE_LIST_CACHE_PATH
    cache_dir = config.data_dir
    os.makedirs(cache_dir, exist_ok=True)
    _NOTE_LIST_CACHE_PATH = os.path.join(cache_dir, "note_paths.json")
    return _NOTE_LIST_CACHE_PATH


def list_all_notes(*, force_rebuild: bool = False) -> list[str]:
    """Return a list of all note paths in the vault, with local caching.

    Caches to ``data/note_paths.json`` with a 5-minute TTL to avoid
    repeated slow API directory walks.

    Args:
        force_rebuild: if True, skip cache and re-fetch from the API.

    Returns:
        A list of vault-relative ``.md`` paths.
    """
    global _NOTE_LIST_CACHE

    with _NOTE_LIST_CACHE_LOCK:
        if not force_rebuild and _NOTE_LIST_CACHE is not None:
            return list(_NOTE_LIST_CACHE)

        cache_path = _get_note_list_cache_path()

        if not force_rebuild:
            try:
                age = time.time() - os.path.getmtime(cache_path)
                if age < _NOTE_LIST_CACHE_TTL:
                    with open(cache_path, encoding="utf-8") as f:
                        cached = json.load(f)
                    _NOTE_LIST_CACHE = list(cached)
                    return list(_NOTE_LIST_CACHE)
            except (OSError, json.JSONDecodeError, ValueError):
                pass

        notes = _walk_dir()

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(notes, f, ensure_ascii=False)
        except OSError:
            pass

        _NOTE_LIST_CACHE = list(notes)
        return notes


def clear_note_cache() -> None:
    """Clear the in-memory note list cache (next call re-fetches from API)."""
    global _NOTE_LIST_CACHE
    with _NOTE_LIST_CACHE_LOCK:
        _NOTE_LIST_CACHE = None


def list_folder(folder_path: str) -> list[str]:
    """List entries directly inside a specific folder (non-recursive).
    Returns both ``.md`` files and subdirectory names (with trailing ``/``).
    """
    clean = _validate_path(folder_path) if folder_path else ""
    results = []
    for entry in _list_dir(clean):
        if _is_excluded(entry):
            continue
        entry = entry.lstrip("/")
        if entry.endswith(".md") or entry.endswith("/"):
            full_path = f"{clean}/{entry}" if clean else entry
            results.append(full_path)
    return results


def list_folder_deep(folder_path: str) -> list[str]:
    """List all .md file paths within a specific folder (recursive, includes subdirectories)."""
    return _walk_dir(_validate_path(folder_path) if folder_path else "")


def get_note(path: str) -> str:
    path = _validate_path(path)
    if not path.endswith(".md"):
        path = path.rstrip("/") + ".md"
    resp = requests.get(f"{_base_url()}/vault/{path}", headers=_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def put_note(path: str, content: str) -> None:
    path = _validate_path(path)
    resp = requests.put(
        f"{_base_url()}/vault/{path}",
        headers=_headers(),
        data=content.encode("utf-8"),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
