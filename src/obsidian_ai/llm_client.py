import functools
import json
import os
import time

from . import config
from .providers import get_provider
from .providers.base import BaseLLMProvider

MAX_CONTEXT_WORDS = 3000

# ── Shared embedding cache (provider-agnostic) ──────────────────────
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
    try:
        os.makedirs(os.path.dirname(EMBED_CACHE_PATH), exist_ok=True)
        with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_EMBED_PERSISTENT_CACHE, f, ensure_ascii=False)
    except Exception as e:
        from .logger import get_logger
        get_logger(__name__).warning(f"Failed to save embed cache: {e}")


_EMBED_PERSISTENT_CACHE.update(_load_embed_cache())

# ── Provider instances (lazy) ───────────────────────────────────────
_CHAT_PROVIDER: BaseLLMProvider | None = None
_EMBED_PROVIDER: BaseLLMProvider | None = None
_PROVIDER_INIT_TIME: dict[str, float] = {}


def _get_chat_provider() -> BaseLLMProvider:
    global _CHAT_PROVIDER
    if _CHAT_PROVIDER is None:
        _CHAT_PROVIDER = get_provider(config.llm_provider)
        _PROVIDER_INIT_TIME["chat"] = time.time()
    p = _CHAT_PROVIDER
    assert p is not None
    return p


def _get_embed_provider() -> BaseLLMProvider:
    global _EMBED_PROVIDER
    if _EMBED_PROVIDER is None:
        _EMBED_PROVIDER = get_provider(config.embed_provider)
        _PROVIDER_INIT_TIME["embed"] = time.time()
    p = _EMBED_PROVIDER
    assert p is not None
    return p


# ── Public API ──────────────────────────────────────────────────────

def is_available() -> bool:
    try:
        chat_ok = _get_chat_provider().is_available()
    except Exception:
        chat_ok = False
    try:
        embed_ok = _get_embed_provider().is_available()
    except Exception:
        embed_ok = False
    return chat_ok and embed_ok


def check_health() -> dict:
    try:
        chat_health = _get_chat_provider().check_health()
    except Exception as e:
        chat_health = {"error": str(e)}

    try:
        embed_health = _get_embed_provider().check_health()
    except Exception as e:
        embed_health = {"error": str(e)}

    chat_status = chat_health.get("overall") or next(
        (v.get("available") for v in chat_health.values() if isinstance(v, dict)),
        False,
    )
    embed_status = embed_health.get("overall") or next(
        (v.get("available") for v in embed_health.values() if isinstance(v, dict)),
        False,
    )

    both_ok = (
        (chat_status == "healthy" or chat_status is True)
        and (embed_status == "healthy" or embed_status is True)
    )

    merged = {}
    merged.update(chat_health)
    merged.update(embed_health)
    merged["overall"] = "healthy" if both_ok else "degraded"
    return merged


def embed(text: str) -> list[float]:
    cached = _EMBED_PERSISTENT_CACHE.get(text)
    if cached is not None:
        return cached
    result = _get_embed_provider().embed(text)
    _EMBED_PERSISTENT_CACHE[text] = result
    return result


def batch_embed(texts: list[str]) -> list[list[float]]:
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
            embeddings = _get_embed_provider().batch_embed(uncached_texts)
            for idx, emb in zip(uncached_indices, embeddings, strict=False):
                _EMBED_PERSISTENT_CACHE[texts[idx]] = emb
                results[idx] = emb
        except Exception:
            for idx in uncached_indices:
                results[idx] = embed(texts[idx])

    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


def clear_embed_cache() -> None:
    embed.cache_clear()  # type: ignore[attr-defined]
    _EMBED_PERSISTENT_CACHE.clear()
    _get_embed_provider().clear_embed_cache()
    try:
        if os.path.isfile(EMBED_CACHE_PATH):
            os.remove(EMBED_CACHE_PATH)
    except Exception:
        pass


def switch_embed_model(model_name: str) -> None:
    _get_embed_provider().switch_embed_model(model_name)
    clear_embed_cache()


def embed_cache_info() -> dict:
    info = embed.cache_info()  # type: ignore[attr-defined]
    provider_info = _get_embed_provider().embed_cache_info()
    result = {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
        "persistent_size": len(_EMBED_PERSISTENT_CACHE),
    }
    result.update(provider_info)
    return result


def chat(messages: list[dict], model: str | None = None, think: bool = True) -> str:
    msgs = list(messages)
    if not think and msgs:
        msgs[0] = {**msgs[0], "content": "/no_think\n" + msgs[0]["content"]}
    return _get_chat_provider().chat(msgs, model=model)


def chat_safe(messages: list[dict], model: str | None = None, think: bool = True) -> str | None:
    msg = list(messages)
    if not think and msg:
        msg[0] = {**msg[0], "content": "/no_think\n" + msg[0]["content"]}
    return _get_chat_provider().chat_safe(msg, model=model)


def truncate_to_budget(text: str, max_words: int = MAX_CONTEXT_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n\n[truncated]"


# Apply LRU cache after the function definition
embed = functools.lru_cache(maxsize=_EMBED_CACHE_SIZE)(embed)  # type: ignore[misc]
