import requests
from . import config

REQUEST_TIMEOUT = 30


def _is_excluded(entry: str) -> bool:
    return any(pattern in entry for pattern in config.EXCLUDE_PATTERNS)


def _base_url():
    return f"http://{config.obsidian_host}:{config.obsidian_port}"


def _headers():
    return {
        "Authorization": f"Bearer {config.obsidian_api_key}",
        "Content-Type": "text/markdown",
    }


def _list_dir(path: str = "") -> list[str]:
    url = f"{_base_url()}/vault/{path}" if path else f"{_base_url()}/vault/"
    resp = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    entries = resp.json().get("files", [])
    return entries


def _walk_dir(path: str = "") -> list[str]:
    results = []
    for entry in _list_dir(path):
        if _is_excluded(entry):
            continue
        full_path = f"{path}/{entry}" if path else entry
        if entry.endswith("/"):
            results.extend(_walk_dir(full_path.rstrip("/")))
        elif entry.endswith(".md"):
            results.append(full_path)
    return results


def list_notes() -> list[str]:
    return _walk_dir()


def get_note(path: str) -> str:
    if not path.endswith(".md"):
        path = path.rstrip("/") + ".md"
    resp = requests.get(f"{_base_url()}/vault/{path}", headers=_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def put_note(path: str, content: str) -> None:
    resp = requests.put(
        f"{_base_url()}/vault/{path}",
        headers=_headers(),
        data=content.encode("utf-8"),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
