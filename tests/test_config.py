"""Tests for config.py — env var loading and _RedactedString."""
import os


def test_default_values():
    """Config loads sensible defaults when env vars are unset."""
    from obsidian_ai import config

    assert config.obsidian_host == os.getenv("OBSIDIAN_HOST", "localhost")
    assert isinstance(config.obsidian_port, int)
    assert config.ollama_base_url == os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    assert config.ollama_embed_model == os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    assert config.ollama_chat_model == os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
    assert isinstance(config.EXCLUDE_PATTERNS, list)
    assert len(config.EXCLUDE_PATTERNS) > 0


def test_redacted_string_masks_key():
    """_RedactedString hides the API key in repr/str."""
    from obsidian_ai.config import _RedactedString

    key = _RedactedString("sk-abc123456789")
    assert "abc123456789" not in repr(key)
    assert "***" in repr(key)
    # Last 4 chars preserved
    assert "6789" in repr(key)


def test_redacted_string_empty():
    """_RedactedString handles empty string."""
    from obsidian_ai.config import _RedactedString

    key = _RedactedString("")
    assert repr(key) == "''"


def test_redacted_string_short():
    """_RedactedString handles very short strings."""
    from obsidian_ai.config import _RedactedString

    key = _RedactedString("ab")
    assert repr(key) == "'****'"


def test_redacted_string_is_str():
    """_RedactedString is still usable as a regular string."""
    from obsidian_ai.config import _RedactedString

    key = _RedactedString("my-api-key")
    assert key == "my-api-key"
    assert key.startswith("my")
    assert len(key) == 10


def test_obsidian_api_key_is_redacted():
    """The module-level obsidian_api_key should be a _RedactedString."""
    from obsidian_ai.config import _RedactedString, obsidian_api_key

    assert isinstance(obsidian_api_key, _RedactedString)


def test_exclude_patterns_content():
    """EXCLUDE_PATTERNS contains expected entries."""
    from obsidian_ai import config

    assert ".git" in config.EXCLUDE_PATTERNS
    assert "__pycache__" in config.EXCLUDE_PATTERNS
    assert ".excalidraw.md" in config.EXCLUDE_PATTERNS
