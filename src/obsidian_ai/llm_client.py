import functools
import json
import os
import time

import requests

from . import config

REQUEST_TIMEOUT = int(os.getenv("OLLAMA_CHAT_TIMEOUT", "45"))
EMBED_TIMEOUT = 60
MAX_RETRIES = 2
INITIAL_BACKOFF = 2
MAX_CONTEXT_WORDS = 3000

RETRYABLE_STATUSES = {429, 502, 503}

# ── Degraded-mode support ──────────────────────────────────────────
_AVAILABLE: bool | None = None
_AVAILABLE_LAST_CHECK: float = 0
_AVAILABLE_CACHE_SECONDS = 30


class ServiceUnavailableError(Exception):
    """Raised when Ollama is not reachable or a required model is missing."""


def is_available() -> bool:
    """Check if Ollama is reachable and has the required models. Results are cached for 30s."""
    global _AVAILABLE, _AVAILABLE_LAST_CHECK
    now = time.time()
    if _AVAILABLE is not None and (now - _AVAILABLE_LAST_CHECK) < _AVAILABLE_CACHE_SECONDS:
        return _AVAILABLE
    try:
        resp = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        has_chat = any(config.ollama_chat_model in m for m in models)
        has_embed = any(config.ollama_embed_model in m for m in models)
        _AVAILABLE = bool(models)  # At least one model available
        _AVAILABLE_LAST_CHECK = now
        if not has_chat:
            from .logger import get_logger
            get_logger(__name__).warning(f"Chat model '{config.ollama_chat_model}' not found in Ollama")
        if not has_embed:
            from .logger import get_logger
            get_logger(__name__).warning(f"Embed model '{config.ollama_embed_model}' not found in Ollama")
        return _AVAILABLE
    except Exception:
        _AVAILABLE = False
        _AVAILABLE_LAST_CHECK = now
        return False


def check_health() -> dict:
    """Return a dict with health status of all dependencies."""
    ollama_ok = is_available()
    status = {
        "ollama": {
            "available": ollama_ok,
            "url": config.ollama_base_url,
            "chat_model": config.ollama_chat_model,
            "embed_model": config.ollama_embed_model,
        },
        "overall": "healthy" if ollama_ok else "degraded",
    }
    if ollama_ok:
        try:
            resp = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            status["ollama"]["models"] = models
        except Exception:
            status["ollama"]["models"] = []
    return status

_EMBED_CACHE_SIZE = 100
EMBED_CACHE_PATH = os.path.join(config.data_dir, "embed_cache.json")
_EMBED_PERSISTENT_CACHE: dict[str, list[float]] = {}


def _load_embed_cache() -> dict[str, list[float]]:
    try:
        if os.path.isfile(EMBED_CACHE_PATH):
            with open(EMBED_CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {k: list(v) for k, v in data.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}


def save_embed_cache() -> None:
    """Write the persistent embedding cache to disk."""
    try:
        os.makedirs(os.path.dirname(EMBED_CACHE_PATH), exist_ok=True)
        with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_EMBED_PERSISTENT_CACHE, f, ensure_ascii=False)
    except Exception as e:
        from .logger import get_logger
        get_logger(__name__).warning(f"Failed to save embed cache: {e}")


# Load persistent cache on module init
_EMBED_PERSISTENT_CACHE.update(_load_embed_cache())


def _request_with_retry(method, url, *, timeout, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.ReadTimeout:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = INITIAL_BACKOFF * (2 ** attempt)
            time.sleep(wait)
        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = INITIAL_BACKOFF * (2 ** attempt)
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in RETRYABLE_STATUSES and attempt < MAX_RETRIES - 1:
                wait = INITIAL_BACKOFF * (2 ** attempt)
                time.sleep(wait)
            else:
                raise


def embed(text: str) -> list[float]:
    cached = _EMBED_PERSISTENT_CACHE.get(text)
    if cached is not None:
        return cached
    resp = _request_with_retry(
        "POST",
        f"{config.ollama_base_url}/api/embeddings",
        json={"model": config.ollama_embed_model, "prompt": text},
        timeout=EMBED_TIMEOUT,
    )
    data = resp.json()
    embedding = data["embedding"]
    assert isinstance(embedding, list)
    _EMBED_PERSISTENT_CACHE[text] = embedding
    return embedding


def batch_embed(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single Ollama API call via ``/api/embed``.

    Checks the persistent cache first; only uncached texts are sent to Ollama.
    Handles 1 or many texts. Falls back to sequential ``embed()`` if the
    batch endpoint is not available (Ollama < 0.3).
    """
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = _EMBED_PERSISTENT_CACHE.get(text)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        try:
            resp = _request_with_retry(
                "POST",
                f"{config.ollama_base_url}/api/embed",
                json={"model": config.ollama_embed_model, "input": uncached_texts},
                timeout=EMBED_TIMEOUT,
            )
            data = resp.json()
            embeddings = data["embeddings"]
            for idx, emb in zip(uncached_indices, embeddings, strict=False):
                _EMBED_PERSISTENT_CACHE[texts[idx]] = emb
                results[idx] = emb
        except Exception:
            # Fallback to sequential embed (handles Ollama < 0.3 gracefully)
            for idx in uncached_indices:
                results[idx] = embed(texts[idx])

    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


def clear_embed_cache() -> None:
    """Clear the LRU and persistent cache for ``embed()``."""
    embed.cache_clear()
    _EMBED_PERSISTENT_CACHE.clear()
    try:
        if os.path.isfile(EMBED_CACHE_PATH):
            os.remove(EMBED_CACHE_PATH)
    except Exception:
        pass


def switch_embed_model(model_name: str) -> None:
    """Switch the embedding model at runtime.

    Updates ``config.ollama_embed_model`` and clears the embed cache
    so subsequent ``embed()`` calls use the new model.
    """
    config.ollama_embed_model = model_name
    clear_embed_cache()


def embed_cache_info() -> dict:
    """Return embedding cache stats: hits, misses, maxsize, currsize, persistent_size."""
    info = embed.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
        "persistent_size": len(_EMBED_PERSISTENT_CACHE),
    }


# Apply LRU cache after the function definition (works around the decoration ordering)
embed = functools.lru_cache(maxsize=_EMBED_CACHE_SIZE)(embed)  # type: ignore[misc]


def chat(messages: list[dict], model: str | None = None, think: bool = True) -> str:
    model = model or config.ollama_chat_model
    msgs = list(messages)
    if not think and msgs:
        msgs[0] = {**msgs[0], "content": "/no_think\n" + msgs[0]["content"]}
    payload = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "keep_alive": "5m",
    }
    resp = _request_with_retry(
        "POST",
        f"{config.ollama_base_url}/api/chat",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    data = resp.json()
    content = data["message"]["content"]
    assert isinstance(content, str)
    return content


def chat_safe(messages: list[dict], model: str | None = None, think: bool = True) -> str | None:
    """Like ``chat()`` but returns ``None`` if Ollama is unavailable (no exception)."""
    if not is_available():
        return None
    try:
        return chat(messages, model=model, think=think)
    except Exception:
        return None


def truncate_to_budget(text: str, max_words: int = MAX_CONTEXT_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n\n[truncated]"
