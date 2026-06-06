"""Standalone pipeline: entity + relationship extraction from all notes.

Usage:
    python -m src.obsidian_ai.extract_entities_pipeline [--force]
"""

import sys
import threading
import time

from . import _index_utils as utils
from . import (
    chroma_store,
    config,
    entity_extractor,
    entity_relations,
    entity_store,
    graph_store,
    obsidian_client,
)
from .logger import get_logger, log_error

log = get_logger(__name__, log_file="indexer.log")

_llm_chat_lock = threading.Semaphore(config.llm_chat_concurrency)


def extract_note_entities(path: str, *, force: bool = False) -> bool:
    """Extract entities + relationships for a single note.

    Returns True if successful.
    """
    if not force:
        try:
            existing = chroma_store.get_by_path(path)
            if existing and existing[0].get("entities_str"):
                log.debug(f"Entities already exist for {path}, skipping")
                return True
        except Exception:
            pass

    try:
        raw = obsidian_client.get_note(path)
    except Exception as e:
        log_error(log, f"Failed to read {path}", exc=e)
        return False

    sanitized = utils._sanitize(raw)
    wc = utils._word_count(sanitized)
    if wc < config.skip_min_tokens:
        log.debug(f"Skipped (too short): {path} — {wc} words")
        return False

    content_hash = utils.compute_hash(raw)

    entities: list[dict] = []
    relationships: list[dict] = []
    try:
        entities, relationships = entity_extractor.extract(
            path, sanitized, content_hash,
            _llm_chat_lock, utils.thermal_throttle,
            config.llm_call_delay,
        )
    except TimeoutError:
        log.critical(
            f"LLM call timed out for {path} — GPU may be hanging. "
            "Saving caches and exiting to prevent system crash."
        )
        entity_extractor.save_cache()
        entity_store.save()
        entity_relations.save()
        graph_store.save()
        return False
    except Exception as e:
        log_error(log, f"Entity extraction failed for {path}", exc=e)
        return False

    time.sleep(utils.get_llm_delay())

    entities_str = ""
    if entities:
        serialised = ",".join(f"{e['type']}:{e['name']}" for e in entities)
        entities_str = f",{serialised},"

    if entities_str:
        try:
            chroma_store.update_metadata(path, {"entities_str": entities_str})
        except Exception as e:
            log.warning(f"Failed to update ChromaDB metadata for {path}: {e}")

    entity_extractor.save_cache()

    log.info(f"Entities extracted: {path} ({len(entities)} entities, {len(relationships)} relationships)")
    return True


def _reconcile_cached_entities_with_graph(content_hashes: dict[str, str]) -> int:
    hash_to_path = {h: p for p, h in content_hashes.items()}
    added = 0
    for content_hash, (entities, _rels, _summary) in entity_extractor._entity_cache.items():
        note_path = hash_to_path.get(content_hash)
        if not note_path:
            continue
        for ent in entities:
            name = ent.get("name", "")
            etype = ent.get("type", "Concept")
            if not name or len(name) < 2:
                continue
            if not graph_store.has_entity_edge(etype, name, note_path):
                graph_store.add_entity_edge(etype, name, note_path)
                added += 1
    if added > 0:
        graph_store.save()
        log.info(f"Reconciled {added} missing entity edges into graph")
    return added


def run_entity_extraction(*, force: bool = False, skip_cached: bool = True, enable_temp_check: bool = True, folder: str | None = None):
    """Phase: extract entities + relationships from all notes (or a folder)."""
    notes = obsidian_client.list_all_notes()
    if folder:
        folder_prefix = folder.strip("/").rstrip("/") + "/"
        notes = [n for n in notes if n == folder or n.startswith(folder_prefix)]
        log.info(f"Filtered to folder '{folder}' — {len(notes)} notes")
    log.info(f"Starting entity extraction — {len(notes)} notes in vault")
    log.info(f"LLM model: {config.ollama_chat_model} (embed: {config.ollama_embed_model})")
    if enable_temp_check:
        log.info(f"Disk temperature monitor active (limit {config.disk_temp_limit}°C)")

    entity_extractor.load_cache()

    log.info("Computing content hashes...")
    content_hashes: dict[str, str] = {}
    for path in notes:
        try:
            raw = obsidian_client.get_note(path)
            content_hashes[path] = utils.compute_hash(raw)
        except Exception as e:
            log.warning(f"Failed to read {path}: {e}")

    _reconcile_cached_entities_with_graph(content_hashes)

    extracted = 0
    skipped_short = 0
    skipped_cached_count = 0
    failed = 0
    interrupted = False

    try:
        for idx, path in enumerate(notes, 1):
            if enable_temp_check and idx % 5 == 0:
                try:
                    utils.check_disk_temp(threshold=config.disk_temp_limit)
                except utils.DiskTempExceededError as e:
                    log.critical(str(e))
                    entity_extractor.save_cache()
                    entity_store.save()
                    entity_relations.save()
                    graph_store.save()
                    return

            content_hash = content_hashes.get(path)
            if not content_hash:
                try:
                    raw = obsidian_client.get_note(path)
                    content_hash = utils.compute_hash(raw)
                except Exception:
                    pass

            log.info(f"Processing: {path}")

            if not force and skip_cached and content_hash:
                cached = entity_extractor._entity_cache.get(content_hash)
                if cached is not None:
                    log.debug(f"Skipped (cached): {path}")
                    skipped_cached_count += 1
                    continue

            success = extract_note_entities(path, force=force)
            if success:
                extracted += 1
            else:
                try:
                    raw = obsidian_client.get_note(path)
                    if utils._word_count(utils._sanitize(raw)) < config.skip_min_tokens:
                        skipped_short += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

            if idx % 10 == 0 or idx == len(notes):
                log.info(f"Entity extraction progress: {idx}/{len(notes)} — "
                         f"{extracted} extracted, {skipped_cached_count} cached, "
                         f"{skipped_short} too short, {failed} failed")

    except KeyboardInterrupt:
        interrupted = True
        log.warning("Interrupted — saving caches...")

    entity_extractor.save_cache()
    entity_store.save()
    entity_relations.save()

    log.info(f"Entity store saved — {entity_store.stats()['total_entities']} entities")
    log.info(f"Entity relationships saved — {entity_relations.stats()['total_relationships']} relationships")
    graph_store.save()
    log.info(f"Graph store saved — {graph_store.node_count()} nodes")

    if interrupted:
        log.info(f"Partial state saved — re-run to continue ({extracted} extracted)")
    else:
        log.info(f"Entity extraction done — Extracted: {extracted}, "
                 f"Cached: {skipped_cached_count}, Too short: {skipped_short}, Failed: {failed}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    no_temp = "--no-temp-check" in sys.argv
    enable_monitor = "--monitor" in sys.argv
    start = time.time()

    if enable_monitor:
        utils.launch_disk_temp_monitor()

    run_entity_extraction(force=force, enable_temp_check=not no_temp)
    print(f"Entity extraction elapsed: {time.time() - start:.1f}s")
