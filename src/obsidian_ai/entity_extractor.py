"""Entity extraction module — extracts entities and relationships from notes via LLM."""

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time

from . import (
    config,
    entity_relations,
    entity_store,
    graph_store,
)
from .logger import get_logger

log = get_logger(__name__, log_file="indexer.log")

ENTITY_CACHE_PATH = os.path.join(config.data_dir, "entity_cache.json")

# Persistent entity cache: content_hash -> tuple[list[dict], list[dict], str] (entities, relationships, summary)
_entity_cache: dict[str, tuple[list[dict], list[dict], str]] = {}
_entity_cache_lock = threading.Lock()


def _load_cache() -> dict[str, tuple[list[dict], list[dict], str]]:
    result: dict[str, tuple[list[dict], list[dict], str]] = {}
    try:
        if os.path.isfile(ENTITY_CACHE_PATH):
            with open(ENTITY_CACHE_PATH, encoding="utf-8") as f:
                raw = dict(json.load(f))
            for key, val in raw.items():
                if isinstance(val, list) and len(val) == 3:
                    result[key] = (val[0], val[1], val[2] if isinstance(val[2], str) else "")
                elif isinstance(val, list) and len(val) == 2 and isinstance(val[1], list):
                    result[key] = (val[0], val[1], "")
                elif isinstance(val, list):
                    result[key] = (val, [], "")
    except Exception as e:
        log.warning(f"Failed to load entity cache: {e}")
    return result


def _save_cache(cache: dict[str, tuple[list[dict], list[dict], str]]) -> None:
    try:
        os.makedirs(os.path.dirname(ENTITY_CACHE_PATH), exist_ok=True)
        serializable = {k: [v[0], v[1], v[2]] for k, v in cache.items()}
        tmp = ENTITY_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, ENTITY_CACHE_PATH)
    except Exception as e:
        log.warning(f"Failed to save entity cache: {e}")


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


def load_cache() -> None:
    """Load entity cache from disk into memory."""
    _entity_cache.clear()
    _entity_cache.update(_load_cache())
    log.info(f"Loaded {len(_entity_cache)} entity cache entries")


def save_cache() -> None:
    """Save entity cache to disk."""
    _save_cache(_entity_cache)
    log.info(f"Entity cache saved — {len(_entity_cache)} entries")


def clear_cache_for_hash(content_hash: str) -> None:
    """Remove a single entry from the entity cache by content hash.

    This forces the next extraction for the same content to re-call the LLM
    rather than returning a cached result.
    """
    with _entity_cache_lock:
        _entity_cache.pop(content_hash, None)


def _extract_entities_cached(
    sanitized: str, path: str, content_hash: str | None
) -> tuple[list[dict], list[dict], str]:
    """Extract entities via LLM with caching. Returns (entities, relationships, summary)."""
    if content_hash:
        with _entity_cache_lock:
            cached = _entity_cache.get(content_hash)
        if cached is not None:
            return cached

    from .pipelines import extract_entities

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            _check_gpu_health()
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(extract_entities, sanitized, path=path)
            deadline = time.monotonic() + config.llm_call_hard_timeout
            try:
                while True:
                    try:
                        entities, relationships, summary = future.result(timeout=0.5)
                        break
                    except concurrent.futures.TimeoutError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError()
            except TimeoutError:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                pool.shutdown(wait=False)
            if content_hash:
                with _entity_cache_lock:
                    _entity_cache[content_hash] = (entities, relationships, summary)
            return entities, relationships, summary
        except TimeoutError:
            log.error(
                f"LLM call timed out after {config.llm_call_hard_timeout}s for {path} — "
                "GPU may be hanging. Aborting extraction."
            )
            raise
        except Exception as e:
            last_exc = e
            if attempt < 2:
                wait = 2 ** attempt
                log.warning(f"Entity extraction attempt {attempt + 1} failed for {path}, retrying in {wait}s: {e}")
                time.sleep(wait)
    raise last_exc


def extract(
    path: str,
    sanitized: str,
    content_hash: str | None,
    llm_chat_lock: threading.Semaphore,
    thermal_throttle: "callable",
    call_delay: float,
    delay_multiplier: float = 1.0,
) -> tuple[list[dict], list[dict]]:
    """Extract entities for a single note and store them.

    Args:
        path: vault-relative note path.
        sanitized: sanitized note content.
        content_hash: content hash for cache lookup.
        llm_chat_lock: semaphore to serialize LLM calls.
        thermal_throttle: callback to track LLM call duration.
        call_delay: base delay between LLM calls.
        delay_multiplier: current thermal throttle multiplier.

    Returns:
        (entities, relationships) — the extracted data.
    """
    entities: list[dict] = []
    relationships: list[dict] = []

    try:
        with llm_chat_lock:
            t0 = time.perf_counter()
            entities, relationships, _summary = _extract_entities_cached(sanitized, path, content_hash)
            elapsed = time.perf_counter() - t0
            thermal_throttle(elapsed)
    except TimeoutError:
        raise
    except Exception as e:
        log.warning(f"Entity extraction failed for {path}: {e}")
        return [], []

    # Store entities
    if entities:
        for ent in entities:
            entity_store.add(
                name=ent["name"],
                type=ent["type"],
                confidence=ent["confidence"],
                path=path,
                chunk_idx=0,
                context=sanitized[:200],
                aliases=ent.get("aliases"),
            )
            graph_store.add_entity_edge(ent["type"], ent["name"], path)

    # Store relationships
    if relationships:
        for rel in relationships:
            entity_relations.add(
                source=rel["source"],
                type=rel["type"],
                target=rel["target"],
                confidence=rel.get("confidence", 0.5),
                source_note=path,
            )

    return entities, relationships


def cache_size() -> int:
    """Return the number of entries in the entity cache."""
    return len(_entity_cache)
