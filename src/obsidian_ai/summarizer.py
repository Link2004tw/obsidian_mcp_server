"""Summary generation module — generates note summaries via LLM."""

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time

from . import (
    chroma_store,
    config,
    llm_client,
    summary_store,
)
from .logger import get_logger

log = get_logger(__name__, log_file="indexer.log")

_SUMMARY_CACHE_PATH = os.path.join(config.data_dir, "summary_cache.json")

# Persistent summary cache: content_hash -> str
_summary_cache: dict[str, str] = {}
_summary_cache_lock = threading.Lock()

def _check_gpu_health() -> None:
    """Check GPU temperature and VRAM usage via nvidia-smi. Exits if overheating or OOM."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) == 3:
                temp = int(parts[0])
                vram_used = int(parts[1])
                vram_total = int(parts[2])
                vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0

                if temp >= config.gpu_temp_limit:
                    log.critical(
                        f"GPU temperature {temp}°C >= limit {config.gpu_temp_limit}°C — "
                        "aborting extraction to prevent system crash"
                    )
                    sys.exit(1)

                if vram_pct >= config.gpu_vram_limit:
                    log.critical(
                        f"GPU VRAM at {vram_pct:.0f}% ({vram_used}MiB/{vram_total}MiB)"
                        f" >= limit {config.gpu_vram_limit}% — "
                        "aborting extraction to prevent OOM crash"
                    )
                    sys.exit(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        pass


SUMMARY_SYSTEM = (
    "You are a note summarizer. Given a note from an Obsidian vault, "
    "produce a concise 1-2 sentence summary capturing the key information. "
    "Be factual, specific, and use the same language as the original note. "
    "Return ONLY the summary text — no preamble, no labels."
)


def _load_cache() -> dict[str, str]:
    try:
        if os.path.isfile(_SUMMARY_CACHE_PATH):
            with open(_SUMMARY_CACHE_PATH, encoding="utf-8") as f:
                return dict(json.load(f))
    except Exception as e:
        log.warning(f"Failed to load summary cache: {e}")
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    try:
        os.makedirs(os.path.dirname(_SUMMARY_CACHE_PATH), exist_ok=True)
        tmp = _SUMMARY_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, _SUMMARY_CACHE_PATH)
    except Exception as e:
        log.warning(f"Failed to save summary cache: {e}")


def load_cache() -> None:
    """Load summary cache from disk into memory."""
    _summary_cache.clear()
    _summary_cache.update(_load_cache())
    log.info(f"Loaded {len(_summary_cache)} summary cache entries")


def save_cache() -> None:
    """Save summary cache to disk."""
    _save_cache(_summary_cache)
    log.info(f"Summary cache saved — {len(_summary_cache)} entries")


def _generate_summary_cached(sanitized: str, content_hash: str | None) -> str:
    """Generate a summary via LLM with caching.

    Checks the entity extraction cache first — if entities were already
    extracted with a summary, that summary is returned directly (no LLM call).
    Falls through to the summarizer LLM only when no pre-computed summary exists.
    """
    if content_hash:
        with _summary_cache_lock:
            cached = _summary_cache.get(content_hash)
        if cached is not None:
            return cached

        # Check entity extraction cache — summary may already exist
        from .entity_extractor import _entity_cache
        entity_entry = _entity_cache.get(content_hash)
        if entity_entry is not None and len(entity_entry) == 3 and entity_entry[2]:
            summary = entity_entry[2]
            with _summary_cache_lock:
                _summary_cache[content_hash] = summary
            log.debug(f"Reused summary from entity cache for hash {content_hash[:8]}...")
            return summary

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"Note content:\n\n{sanitized[:3000]}"},
    ]

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            _check_gpu_health()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: llm_client.chat(messages, think=False))
                deadline = time.monotonic() + config.llm_call_hard_timeout
                while True:
                    try:
                        summary = future.result(timeout=0.5).strip()
                        break
                    except concurrent.futures.TimeoutError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError()
            if content_hash:
                with _summary_cache_lock:
                    _summary_cache[content_hash] = summary
            return summary
        except TimeoutError:
            log.error(
                f"LLM call timed out after {config.llm_call_hard_timeout}s — "
                "GPU may be hanging. Aborting extraction."
            )
            raise
        except Exception as e:
            last_exc = e
            if attempt < 2:
                wait = 2 ** attempt
                log.warning(f"Summary generation attempt {attempt + 1} failed, retrying in {wait}s: {e}")
                time.sleep(wait)
    raise last_exc


def summarize(
    path: str,
    sanitized: str,
    content_hash: str | None,
    llm_chat_lock: threading.Semaphore,
    thermal_throttle: "callable",
    call_delay: float,
    delay_multiplier: float = 1.0,
) -> str:
    """Generate a summary for a single note and store it.

    Args:
        path: vault-relative note path.
        sanitized: sanitized note content.
        content_hash: content hash for cache lookup.
        llm_chat_lock: semaphore to serialize LLM calls.
        thermal_throttle: callback to track LLM call duration.
        call_delay: base delay between LLM calls.
        delay_multiplier: current thermal throttle multiplier.

    Returns:
        The summary text (empty string if generation failed).
    """
    summary = ""

    try:
        with llm_chat_lock:
            t0 = time.perf_counter()
            summary = _generate_summary_cached(sanitized, content_hash)
            elapsed = time.perf_counter() - t0
            thermal_throttle(elapsed)
    except TimeoutError:
        raise
    except Exception as e:
        log.warning(f"Summary generation failed for {path}: {e}")
        return ""

    # Update ChromaDB metadata
    if summary:
        try:
            chroma_store.update_metadata(path, {"summary": summary})
        except Exception as e:
            log.warning(f"Failed to update ChromaDB metadata for {path}: {e}")

        try:
            summary_store.add(
                path=path,
                title=os.path.splitext(os.path.basename(path))[0],
                summary=summary,
            )
        except Exception as e:
            log.warning(f"Failed to store summary embedding for {path}: {e}")

    return summary


def cache_size() -> int:
    """Return the number of entries in the summary cache."""
    return len(_summary_cache)
