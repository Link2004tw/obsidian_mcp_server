"""Orchestrator — runs Phase 1 (chunking/embedding) then Phase 2 (entity extraction)."""

import os
import sys
import threading
import time

from . import _index_utils as utils
from . import chunker, config, extractor, graph_store, llm_client, obsidian_client
from ._index_utils import (
    _extract_frontmatter_fields,
    _extract_tags,
    _links_to_meta,
    _sanitize,
    _tags_to_meta,
    _word_count,
    chunk_text,
    chunk_text_heading_aware,
    split_by_headings,
)
from .frontmatter import add_tags as fm_add_tags
from .logger import get_logger, log_error
from .wiki_links import extract_wiki_links

log = get_logger(__name__, log_file="indexer.log")

# When True, skip LLM-based entity extraction for faster indexing.
SKIP_ENTITIES = False

# When True, skip LLM-based summary generation for faster indexing.
SKIP_SUMMARIES = False


def add_tags_to_note(path: str, tags: list[str]) -> None:
    """Add tags to a note's YAML frontmatter."""
    content = obsidian_client.get_note(path)
    new_content = fm_add_tags(content, tags)
    obsidian_client.put_note(path, new_content)


def run_index():
    """Full index pipeline: Phase 1 (chunk+embed) then Phase 2 (extract entities)."""
    start = time.time()

    # Phase 1: chunk + embed (nomic) — no LLM chat calls
    chunker.run_chunking()

    # Phase 2: entity extraction + summary (Qwen8b) — skipped if flag is set
    if not SKIP_ENTITIES and not SKIP_SUMMARIES:
        log.info("Starting Phase 2 — entity extraction + summary generation")
        extractor.run_extraction(force=False, skip_cached=True)
    elif SKIP_ENTITIES and not SKIP_SUMMARIES:
        log.info("Phase 2: entities skipped, generating summaries only")
        extractor.run_extraction(force=False, skip_cached=True)
    else:
        log.info("Phase 2 skipped (SKIP_ENTITIES set)")

    log.info(f"Total elapsed: {time.time() - start:.1f}s")


def watch():
    """Watch mode — chunk/embed on file changes, then extract entities."""
    vault_path = config.vault_path
    if not vault_path:
        log.error("VAULT_PATH not set — add VAULT_PATH=/path/to/obsidian/vault to your .env file")
        print("ERROR: VAULT_PATH not set.")
        return
    if not os.path.isdir(vault_path):
        log.error(f"VAULT_PATH does not exist: {vault_path}")
        print(f"ERROR: VAULT_PATH '{vault_path}' does not exist.")
        return
    vault_path_normalized = vault_path.replace("\\", "/").rstrip("/")
    log.info(f"Starting watcher on vault: {vault_path}")

    hash_map = utils.load_hash_map()

    def _to_rel_path(abs_path: str) -> str:
        rel = abs_path.replace("\\", "/")
        if rel.startswith(vault_path_normalized + "/"):
            rel = rel[len(vault_path_normalized) + 1:]
        return rel

    _pending: dict[str, tuple[str, float]] = {}
    _pending_lock = threading.Lock()
    _wake = threading.Event()
    _debounce_secs = 2.0

    def _queue_event(path: str, action: str) -> None:
        with _pending_lock:
            _pending[path] = (action, time.time())
        _wake.set()

    def _worker() -> None:
        while True:
            _wake.wait(timeout=0.5)
            _wake.clear()
            now = time.time()
            with _pending_lock:
                ready = [
                    (path, action)
                    for path, (action, ts) in _pending.items()
                    if now - ts >= _debounce_secs
                ]
                for path, _ in ready:
                    del _pending[path]
            for path, action in ready:
                try:
                    if action == "index":
                        content = obsidian_client.get_note(path)
                        current_hash = utils.compute_hash(content)
                        # Phase 1: chunk + embed
                        chunker.index_note(path, content=content, _content_hash=current_hash)
                        graph_store.register_title(path)
                        for link in extract_wiki_links(content):
                            resolved = graph_store.resolve_link(link)
                            if resolved and resolved != path:
                                graph_store.add_edge(path, resolved)
                        graph_store.save()
                        with _pending_lock:
                            hash_map[path] = current_hash
                        utils.save_hash_map(hash_map)
                        # Phase 2: extract entities (if not skipped)
                        if not SKIP_ENTITIES:
                            extractor.extract_note(path)
                    elif action == "delete":
                        chunker.delete_note(path)
                    elif action.startswith("rename:"):
                        old_path, new_path = action.split(":", 1)[1].split("->")
                        graph_store.rename_node(old_path, new_path)
                        graph_store.save()
                except Exception as e:
                    log_error(log, f"Worker failed ({action}): {path}", exc=e)

    threading.Thread(target=_worker, daemon=True).start()

    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            _queue_event(_to_rel_path(event.src_path), "index")

        def on_modified(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            _queue_event(_to_rel_path(event.src_path), "index")

        def on_deleted(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            _queue_event(_to_rel_path(event.src_path), "delete")

        def on_moved(self, event):
            if event.is_directory:
                return
            if event.src_path.endswith(".md") and event.dest_path.endswith(".md"):
                old_rel = _to_rel_path(event.src_path)
                new_rel = _to_rel_path(event.dest_path)
                _queue_event(old_rel, f"rename:{old_rel}->{new_rel}")
            elif event.src_path.endswith(".md"):
                _queue_event(_to_rel_path(event.src_path), "delete")
            elif event.dest_path.endswith(".md"):
                _queue_event(_to_rel_path(event.dest_path), "index")

    observer = Observer()
    observer.schedule(Handler(), vault_path, recursive=True)
    observer.start()
    log.info("Watcher started — press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    if "--skip-entities" in sys.argv:
        SKIP_ENTITIES = True
        log.info("Entity extraction disabled via --skip-entities")
    if "--skip-summaries" in sys.argv:
        SKIP_SUMMARIES = True
        log.info("Summary generation disabled via --skip-summaries")
    if "--watch" in sys.argv:
        watch()
    else:
        start = time.time()
        run_index()
        print(f"Elapsed: {time.time() - start:.1f}s")
