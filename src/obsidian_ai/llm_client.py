import functools
import time

import requests

from . import config

REQUEST_TIMEOUT = 300
EMBED_TIMEOUT = 180
MAX_RETRIES = 3
INITIAL_BACKOFF = 2
MAX_CONTEXT_WORDS = 3000

RETRYABLE_STATUSES = {429, 502, 503}

_EMBED_CACHE_SIZE = 100


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
    resp = _request_with_retry(
        "POST",
        f"{config.ollama_base_url}/api/embeddings",
        json={"model": config.ollama_embed_model, "prompt": text},
        timeout=EMBED_TIMEOUT,
    )
    data = resp.json()
    embedding = data["embedding"]
    assert isinstance(embedding, list)
    return embedding


def clear_embed_cache() -> None:
    """Clear the LRU cache for ``embed()``."""
    embed.cache_clear()


def switch_embed_model(model_name: str) -> None:
    """Switch the embedding model at runtime.

    Updates ``config.ollama_embed_model`` and clears the embed cache
    so subsequent ``embed()`` calls use the new model.
    """
    config.ollama_embed_model = model_name
    clear_embed_cache()


def embed_cache_info() -> dict:
    """Return embedding cache stats: hits, misses, maxsize, currsize."""
    info = embed.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
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


def truncate_to_budget(text: str, max_words: int = MAX_CONTEXT_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n\n[truncated]"
