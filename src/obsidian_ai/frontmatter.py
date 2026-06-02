"""YAML frontmatter parsing and manipulation."""
import yaml


def parse(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from note content.

    Returns:
        (meta, body) where meta is the parsed YAML dict and body is the content after frontmatter.
        If no frontmatter exists, meta is empty dict and body is the full content.
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            return meta, body
    return {}, content


def build(meta: dict, body: str) -> str:
    """Reconstruct note content from frontmatter dict and body."""
    if not meta:
        return body
    return f"---\n{yaml.dump(meta, default_flow_style=False).strip()}\n---\n\n{body}"


def add_tags(content: str, tags: list[str]) -> str:
    """Add tags to note content's YAML frontmatter.

    - Creates frontmatter if absent
    - Appends new tags (no duplicates)
    - Converts string tags field to list if needed
    """
    meta, body = parse(content)

    existing = meta.get("tags", [])
    if isinstance(existing, str):
        existing = [existing]
    for tag in tags:
        if tag not in existing:
            existing.append(tag)
    meta["tags"] = existing

    return build(meta, body)


def remove_tags(content: str, tags: list[str]) -> str:
    """Remove specific tags from note content's YAML frontmatter.

    Silently ignores tags that don't exist.
    """
    meta, body = parse(content)
    existing = meta.get("tags", [])
    if isinstance(existing, str):
        existing = [existing]
    tags_set = set(tags)
    meta["tags"] = [t for t in existing if t not in tags_set]
    return build(meta, body)


def set_tags(content: str, tags: list[str]) -> str:
    """Replace all tags in note content's YAML frontmatter with the given list.

    Creates frontmatter if absent.
    """
    meta, body = parse(content)
    meta["tags"] = list(tags)
    return build(meta, body)
