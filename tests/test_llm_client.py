"""Tests for llm_client.py — embed, chat, truncate_to_budget (mocked HTTP)."""
from unittest.mock import MagicMock, patch

from obsidian_ai import llm_client


def _mock_response(json_data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return resp


# ── truncate_to_budget ─────────────────────────────────────────────


def test_truncate_short_text():
    text = "short text here"
    assert llm_client.truncate_to_budget(text) == text


def test_truncate_long_text():
    words = " ".join([f"word{i}" for i in range(5000)])
    result = llm_client.truncate_to_budget(words, max_words=100)
    assert result.endswith("[truncated]")
    # 100 words joined by spaces + "\n\n[truncated]" — split gives 100 + 1 = 101
    assert len(result.split()) == 101


def test_truncate_exact_budget():
    words = " ".join([f"word{i}" for i in range(100)])
    result = llm_client.truncate_to_budget(words, max_words=100)
    assert "[truncated]" not in result


def test_truncate_empty():
    assert llm_client.truncate_to_budget("") == ""


# ── embed ──────────────────────────────────────────────────────────


@patch("obsidian_ai.llm_client.requests")
def test_embed_calls_ollama(mock_requests):
    llm_client.clear_embed_cache()
    mock_requests.request.return_value = _mock_response({"embedding": [0.1, 0.2, 0.3]})
    result = llm_client.embed("test text")
    assert result == [0.1, 0.2, 0.3]
    mock_requests.request.assert_called_once()


@patch("obsidian_ai.llm_client.requests")
def test_embed_caches_results(mock_requests):
    llm_client.clear_embed_cache()
    mock_requests.request.return_value = _mock_response({"embedding": [0.1, 0.2]})
    llm_client.embed("cached query")
    llm_client.embed("cached query")
    # Second call should use cache — only one HTTP request
    assert mock_requests.request.call_count == 1


def test_embed_cache_info():
    info = llm_client.embed_cache_info()
    assert "hits" in info
    assert "misses" in info
    assert "maxsize" in info
    assert "currsize" in info


def test_clear_embed_cache():
    llm_client.clear_embed_cache()
    info = llm_client.embed_cache_info()
    assert info["currsize"] == 0


# ── chat ───────────────────────────────────────────────────────────


@patch("obsidian_ai.llm_client.requests")
def test_chat_basic(mock_requests):
    mock_requests.request.return_value = _mock_response({
        "message": {"content": "Hello there!"}
    })
    result = llm_client.chat([{"role": "user", "content": "hi"}])
    assert result == "Hello there!"


@patch("obsidian_ai.llm_client.requests")
def test_chat_think_false(mock_requests):
    """think=False prepends /no_think to the first message."""
    mock_requests.request.return_value = _mock_response({
        "message": {"content": "ok"}
    })
    llm_client.chat([{"role": "user", "content": "test"}], think=False)
    call_args = mock_requests.request.call_args
    payload = call_args[1]["json"]
    assert payload["messages"][0]["content"].startswith("/no_think")


@patch("obsidian_ai.llm_client.requests")
def test_chat_custom_model(mock_requests):
    mock_requests.request.return_value = _mock_response({
        "message": {"content": "done"}
    })
    llm_client.chat([{"role": "user", "content": "hi"}], model="custom:model")
    call_args = mock_requests.request.call_args
    payload = call_args[1]["json"]
    assert payload["model"] == "custom:model"
