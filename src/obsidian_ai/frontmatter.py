"""YAML frontmatter parsing and manipulation."""
import yaml


def validate(content: str) -> list[str]:
    """Check YAML frontmatter for common issues.

    Args:
        content: raw note Markdown (may or may not have frontmatter).

    Returns:
        List of validation warnings (empty means all good).
    """
    if not content.startswith("---"):
        return []  # no frontmatter = nothing to validate

    parts = content.split("---", 2)
    if len(parts) < 3:
        return ["Frontmatter block is not properly closed (needs trailing ---)"]

    raw_yaml = parts[1]
    if not raw_yaml.strip():
        return ["Frontmatter block is empty"]

    warnings: list[str] = []

    # Check for duplicate keys
    seen_keys: set[str] = set()
    for line in raw_yaml.splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key = line.split(":", 1)[0].strip()
            if key in seen_keys:
                warnings.append(f"Duplicate key in frontmatter: \"{key}\"")
            seen_keys.add(key)

    # Check YAML parses correctly
    try:
        meta = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as e:
        warnings.append(f"YAML parse error in frontmatter: {e}")
        return warnings

    if not isinstance(meta, dict):
        warnings.append("Frontmatter must be a YAML mapping (key: value pairs)")
        return warnings

    # Validate common field types
    if "tags" in meta:
        tags = meta["tags"]
        if tags is None:
            warnings.append("Field \"tags\" is null — remove the key or provide a list")
        elif isinstance(tags, str):
            warnings.append("Field \"tags\" should be a list, not a string")
        elif isinstance(tags, (int, float)):
            warnings.append("Field \"tags\" should be a list of strings, not a number")
        elif isinstance(tags, list):
            non_strings = [t for t in tags if not isinstance(t, str)]
            if non_strings:
                warnings.append(f"Field \"tags\" contains non-string values: {non_strings}")
        else:
            warnings.append(f"Field \"tags\" has unexpected type: {type(tags).__name__}")

    if "aliases" in meta:
        aliases = meta["aliases"]
        if not isinstance(aliases, list):
            warnings.append("Field \"aliases\" should be a list")
        else:
            non_strings = [a for a in aliases if not isinstance(a, str)]
            if non_strings:
                warnings.append(f"Field \"aliases\" contains non-string values: {non_strings}")

    return warnings


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
