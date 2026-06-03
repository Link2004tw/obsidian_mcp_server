"""Tests for pipelines.py — LLM-based pipelines."""
from unittest.mock import patch

from obsidian_ai.pipelines import _EXTRACT_ENTITIES_CACHE, extract_entities


@patch("obsidian_ai.pipelines.llm_client")
def test_extract_entities_empty(mock_llm):
    _EXTRACT_ENTITIES_CACHE.clear()
    mock_llm.chat.return_value = '{"entities": []}'
    result, rels = extract_entities("This note has no entities worth mentioning.", path="test_none.md")
    assert isinstance(result, list)
    assert result == []
    assert rels == []


@patch("obsidian_ai.pipelines.llm_client")
def test_extract_entities_parses_json(mock_llm):
    _EXTRACT_ENTITIES_CACHE.clear()
    mock_llm.chat.return_value = '{"entities": [{"name": "ESP32", "type": "Hardware", "confidence": 0.95}]}'
    result, rels = extract_entities("Using ESP32 for IoT", path="test_hw.md")
    assert len(result) == 1
    assert result[0]["name"] == "ESP32"
    assert result[0]["type"] == "Hardware"
    assert result[0]["confidence"] == 0.95


@patch("obsidian_ai.pipelines.llm_client")
def test_extract_entities_multiple(mock_llm):
    _EXTRACT_ENTITIES_CACHE.clear()
    mock_llm.chat.return_value = (
        '{"entities": ['
        '{"name": "Alice", "type": "Person", "confidence": 0.95},'
        '{"name": "ProjectX", "type": "Project", "confidence": 0.9},'
        '{"name": "ESP32", "type": "Hardware", "confidence": 0.98}'
        "]}"
    )
    result, rels = extract_entities("Alice worked on ProjectX using ESP32", path="test_multi.md")
    assert len(result) == 3
    names = {e["name"] for e in result}
    assert names == {"Alice", "ProjectX", "ESP32"}


@patch("obsidian_ai.pipelines.llm_client")
def test_extract_entities_invalid_json_fallback(mock_llm):
    _EXTRACT_ENTITIES_CACHE.clear()
    # LLM sometimes wraps response in extra text
    mock_llm.chat.return_value = (
        "Here are the entities:\n"
        '{"entities": [{"name": "ESP32", "type": "Hardware", "confidence": 0.95}]}\n'
        "That's all."
    )
    result, rels = extract_entities("Using ESP32", path="test_fallback.md")
    assert len(result) == 1
    assert result[0]["name"] == "ESP32"


@patch("obsidian_ai.pipelines.llm_client")
def test_extract_entities_broken_json(mock_llm):
    _EXTRACT_ENTITIES_CACHE.clear()
    mock_llm.chat.return_value = "I don't see any entities here."
    result, rels = extract_entities("Some text", path="test_broken.md")
    assert result == []


@patch("obsidian_ai.pipelines.llm_client")
def test_extract_entities_caching(mock_llm):
    _EXTRACT_ENTITIES_CACHE.clear()
    mock_llm.chat.return_value = '{"entities": [{"name": "Alice", "type": "Person", "confidence": 0.95}]}'
    result1, rels1 = extract_entities("Alice text", path="test_cache.md")
    assert len(result1) == 1
    # Second call with same path should use cache, not LLM
    mock_llm.chat.reset_mock()
    result2, rels2 = extract_entities("Alice text", path="test_cache.md")
    assert len(result2) == 1
    mock_llm.chat.assert_not_called()


@patch("obsidian_ai.pipelines.llm_client")
def test_extract_entities_validates_types(mock_llm):
    _EXTRACT_ENTITIES_CACHE.clear()
    mock_llm.chat.return_value = (
        '{"entities": ['
        '{"name": "FooBar", "type": "InvalidType", "confidence": 0.9},'
        '{"name": "", "type": "Person", "confidence": 0.9},'
        '{"name": "A", "type": "Person", "confidence": 0.9}'
        "]}"
    )
    result, rels = extract_entities("test", path="test_validate.md")
    # FooBar gets type Concept (fallback for invalid type), empty string skipped, "A" too short (< 2)
    assert len(result) == 1
    assert result[0]["name"] == "FooBar"
    assert result[0]["type"] == "Concept"
