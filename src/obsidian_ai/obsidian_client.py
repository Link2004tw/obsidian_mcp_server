import os
import requests

from . import config

REQUEST_TIMEOUT = 30


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


def list_all_notes() -> list[str]:
    return _walk_dir()


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
