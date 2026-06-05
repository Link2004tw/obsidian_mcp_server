"""Standalone pipeline: summary generation for all notes.

Usage:
    python -m src.obsidian_ai.summarize_pipeline [--force]
"""

import sys
import threading
import time

from . import _index_utils as utils
from . import (
    chroma_store,
    config,
    obsidian_client,
    summarizer,
)
from .logger import get_logger, log_error

log = get_logger(__name__, log_file="indexer.log")

_llm_chat_lock = threading.Semaphore(config.llm_chat_concurrency)


def extract_note_summary(path: str, *, force: bool = False) -> bool:
    """Generate a summary for a single note.

    Returns True if successful.
    """
    if not force:
        try:
            existing = chroma_store.get_by_path(path)
            if existing and existing[0].get("summary"):
                log.debug(f"Summary already exists for {path}, skipping")
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

    summary = ""
    try:
        summary = summarizer.summarize(
            path, sanitized, content_hash,
            _llm_chat_lock, utils.thermal_throttle,
            config.llm_call_delay,
        )
    except TimeoutError:
        log.critical(
            f"LLM call timed out for {path} — GPU may be hanging. "
            "Saving caches and exiting to prevent system crash."
        )
        summarizer.save_cache()
        return False
    except Exception as e:
        log_error(log, f"Summary generation failed for {path}", exc=e)
        return False

    time.sleep(utils.get_llm_delay())

    if summary:
        try:
            chroma_store.update_metadata(path, {"summary": summary})
        except Exception as e:
            log.warning(f"Failed to update ChromaDB metadata for {path}: {e}")

    summarizer.save_cache()

    log.info(f"Summary generated: {path} (summary={bool(summary)})")
    return True


def run_summary_generation(*, force: bool = False, skip_cached: bool = True, enable_temp_check: bool = True):
    """Phase: generate summaries for all notes."""
    notes = obsidian_client.list_all_notes()
    log.info(f"Starting summary generation — {len(notes)} notes in vault")
    if enable_temp_check:
        log.info(f"Disk temperature monitor active (limit {config.disk_temp_limit}°C)")

    summarizer.load_cache()

    log.info("Computing content hashes...")
    content_hashes: dict[str, str] = {}
    for path in notes:
        try:
            raw = obsidian_client.get_note(path)
            content_hashes[path] = utils.compute_hash(raw)
        except Exception as e:
            log.warning(f"Failed to read {path}: {e}")

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
                    summarizer.save_cache()
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
                cached = summarizer._summary_cache.get(content_hash)
                if cached is not None:
                    log.debug(f"Skipped (cached): {path}")
                    skipped_cached_count += 1
                    continue

            success = extract_note_summary(path, force=force)
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
                log.info(f"Summary progress: {idx}/{len(notes)} — "
                         f"{extracted} generated, {skipped_cached_count} cached, "
                         f"{skipped_short} too short, {failed} failed")

    except KeyboardInterrupt:
        interrupted = True
        log.warning("Interrupted — saving caches...")

    summarizer.save_cache()

    if interrupted:
        log.info(f"Partial state saved — re-run to continue ({extracted} generated)")
    else:
        log.info(f"Summary generation done — Generated: {extracted}, "
                 f"Cached: {skipped_cached_count}, Too short: {skipped_short}, Failed: {failed}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    no_temp = "--no-temp-check" in sys.argv
    enable_monitor = "--monitor" in sys.argv
    start = time.time()

    if enable_monitor:
        utils.launch_disk_temp_monitor()

    run_summary_generation(force=force, enable_temp_check=not no_temp)
    print(f"Summary generation elapsed: {time.time() - start:.1f}s")
