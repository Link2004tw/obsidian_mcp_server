"""Wiki-link parsing helpers for Obsidian Markdown notes."""

import json
import os
import re
import time
import threading

_WIKI_LINK_RE = re.compile(r"(?<!!)\[\[([^\[\]\n]+)\]\]")

_TITLE_CACHE: dict[str, list[str]] | None = None
_TITLE_CACHE_LOCK = threading.Lock()
_TITLE_CACHE_PATH: str | None = None
_CACHE_TTL = 300  # seconds before cache is considered stale


def normalize_wiki_link_target(target: str) -> str:
    """Normalize an Obsidian wiki-link target."""
    target = target.strip()
    if "|" in target:
        target = target.split("|", 1)[0].strip()
    if "#" in target:
        target = target.split("#", 1)[0].strip()
    target = target.strip().strip("/").replace("\\", "/")
    target = re.sub(r"/+", "/", target)
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target.casefold()


def extract_wiki_links(content: str) -> list[str]:
    """Extract normalized wiki-link targets from note content in first-seen order."""
    links: list[str] = []
    seen: set[str] = set()
    for match in _WIKI_LINK_RE.finditer(content):
        target = normalize_wiki_link_target(match.group(1))
        if not target or target in seen:
            continue
        seen.add(target)
        links.append(target)
    return links


def _get_cache_path() -> str:
    from . import config
    cache_dir = os.path.dirname(os.path.abspath(config.chroma_path)) if os.path.isabs(config.chroma_path) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(config.chroma_path))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "title_to_path.json")


def _build_title_to_path_map_uncached() -> dict[str, list[str]]:
    """Build the mapping from scratch via Obsidian REST API."""
    from . import obsidian_client

    paths = obsidian_client.list_all_notes()
    mapping: dict[str, list[str]] = {}

    for path in paths:
        norm_path = normalize_wiki_link_target(path)
        mapping.setdefault(norm_path, []).append(path)

        title = os.path.splitext(os.path.basename(path))[0]
        norm_title = normalize_wiki_link_target(title)
        if norm_title != norm_path:
            mapping.setdefault(norm_title, []).append(path)

    return mapping


def build_title_to_path_map(*, force_rebuild: bool = False) -> dict[str, list[str]]:
    """Build a cached mapping of normalized wiki-link targets to vault-relative paths.

    Results are cached to ``data/title_to_path.json`` and reused for up to 5 minutes.
    Pass ``force_rebuild=True`` to refresh from the API.

    Returns a dict where keys are normalized identifiers
    and values are lists of matching vault-relative paths.
    """
    global _TITLE_CACHE, _TITLE_CACHE_PATH

    with _TITLE_CACHE_LOCK:
        if _TITLE_CACHE is not None and not force_rebuild:
            return _TITLE_CACHE

        cache_path = _get_cache_path()

        if not force_rebuild and os.path.isfile(cache_path):
            try:
                age = time.time() - os.path.getmtime(cache_path)
                if age < _CACHE_TTL:
                    with open(cache_path, encoding="utf-8") as f:
                        cached = json.load(f)
                    _TITLE_CACHE = {k: v for k, v in cached.items()}
                    return _TITLE_CACHE
            except (OSError, json.JSONDecodeError):
                pass

        mapping = _build_title_to_path_map_uncached()

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

        _TITLE_CACHE = mapping
        return mapping


def resolve_link_target(
    target: str,
    title_to_path: dict[str, list[str]],
    current_path: str | None = None,
) -> str | None:
    """Resolve a wiki-link target to a vault-relative file path.

    Resolution strategy:
    1. Normalize the target and look up in the mapping
    2. Prefer exact path match over title match when both exist
    3. For duplicate titles, prefer same-folder match
    4. Fall back to alphabetical sort

    Returns ``None`` if no match is found.
    """
    norm = normalize_wiki_link_target(target)

    candidates = title_to_path.get(norm)
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Prefer exact path match over title match
    exact = [p for p in candidates if normalize_wiki_link_target(p) == norm]
    if exact:
        return exact[0]

    # Prefer same-folder match for duplicate titles
    if current_path:
        current_dir = os.path.dirname(current_path)
        same_dir = [p for p in candidates if os.path.dirname(p) == current_dir]
        if same_dir:
            return same_dir[0]

    return sorted(candidates)[0]
