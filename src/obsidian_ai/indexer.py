import os
import re
import sys
import time

from . import chroma_store, config, llm_client, obsidian_client
from .frontmatter import add_tags as fm_add_tags, parse as fm_parse
from .logger import get_logger, log_error

log = get_logger(__name__, log_file="indexer.log")

SKIP_MIN_TOKENS = 20
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def _word_count(text: str) -> int:
    return len(text.split())


def _sanitize(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks


def _get_file_mtime(path: str) -> float | None:
    """Get the filesystem mtime for a vault-relative path, or None if unavailable."""
    if not config.vault_path:
        return None
    abs_path = os.path.join(config.vault_path, path)
    try:
        return os.path.getmtime(abs_path)
    except OSError:
        return None


def _should_skip_by_mtime(path: str) -> bool:
    """Check if a note's mtime is unchanged since last index. Returns True to skip."""
    current_mtime = _get_file_mtime(path)
    if current_mtime is None:
        return False
    stored = chroma_store.get_by_path(path)
    if not stored:
        return False
    stored_mtime = stored[0].get("mtime")
    if stored_mtime is None:
        return False
    return abs(float(stored_mtime) - current_mtime) < 0.001


def _extract_tags(raw_content: str) -> list[str]:
    """Extract tags from a note's YAML frontmatter."""
    meta, _ = fm_parse(raw_content)
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    return tags if isinstance(tags, list) else []


def _tags_to_meta(tags: list[str]) -> str:
    """Serialize tags to a safely searchable string: ",tag1,tag2,"""
    return "," + ",".join(tags) + ","


def _index_note(path: str, content: str | None = None) -> bool:
    """Index a single note. Returns True if successful."""
    try:
        if content is None:
            raw = obsidian_client.get_note(path)
        else:
            raw = content

        tags = _extract_tags(raw)
        sanitized = _sanitize(raw)
        wc = _word_count(sanitized)
        if wc < SKIP_MIN_TOKENS:
            log.debug(f"Skipped (too short): {path} — {wc} words")
            return False
        chroma_store.delete_by_path(path)
        chunks = chunk_text(sanitized)
        title = os.path.splitext(os.path.basename(path))[0]
        mtime = _get_file_mtime(path)
        for i, chunk in enumerate(chunks):
            embedding = llm_client.embed(chunk)
            metadata: dict = {
                "path": path,
                "title": title,
                "chunk": i,
                "word_count": wc,
            }
            if tags:
                metadata["tags_str"] = _tags_to_meta(tags)
            if mtime is not None:
                metadata["mtime"] = mtime
            chroma_store.upsert(path=path, chunk_idx=i, embedding=embedding, metadata=metadata)
        log.info(f"Indexed: {path} ({len(chunks)} chunks, tags={tags})")
        return True
    except Exception as e:
        log_error(log, f"FAILED: {path}", exc=e)
        return False


def _delete_note(path: str) -> bool:
    """Delete a note from the index. Returns True if successful."""
    try:
        chroma_store.delete_by_path(path)
        log.info(f"Deleted from index: {path}")
        return True
    except Exception as e:
        log_error(log, f"DELETE FAILED: {path}", exc=e)
        return False


def run_index():
    notes = obsidian_client.list_all_notes()
    log.info(f"Starting index — {len(notes)} notes found")
    indexed = 0
    skipped = 0
    failed = 0
    mtime_skipped = 0
    for path in notes:
        try:
            if _should_skip_by_mtime(path):
                mtime_skipped += 1
                log.debug(f"Skipped (mtime unchanged): {path}")
                continue

            content = obsidian_client.get_note(path)
            wc = _word_count(_sanitize(content))
            if wc < SKIP_MIN_TOKENS:
                skipped += 1
                log.debug(f"Skipped (too short): {path} — {wc} words")
                continue

            if _index_note(path, content=content):
                indexed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            log_error(log, f"FAILED: {path}", exc=e)
    log.info(f"Done — Indexed: {indexed}, Skipped: {skipped}, Unchanged (mtime): {mtime_skipped}, Failed: {failed}")


def add_tags_to_note(path: str, tags: list[str]) -> None:
    """Add tags to a note's YAML frontmatter."""
    content = obsidian_client.get_note(path)
    new_content = fm_add_tags(content, tags)
    obsidian_client.put_note(path, new_content)


def watch():
    import threading

    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    vault_path = config.vault_path
    if not vault_path:
        log.error("VAULT_PATH not set — add VAULT_PATH=/path/to/obsidian/vault to your .env file")
        print("ERROR: VAULT_PATH not set. Add VAULT_PATH to your .env file.")
        return
    if not os.path.isdir(vault_path):
        log.error(f"VAULT_PATH does not exist or is not a directory: {vault_path}")
        print(f"ERROR: VAULT_PATH '{vault_path}' does not exist or is not a directory.")
        return
    vault_path_normalized = vault_path.replace("\\", "/").rstrip("/")
    log.info(f"Starting watcher on vault: {vault_path}")

    def _to_rel_path(abs_path: str) -> str:
        rel = abs_path.replace("\\", "/")
        if rel.startswith(vault_path_normalized + "/"):
            rel = rel[len(vault_path_normalized) + 1:]
        return rel

    class Handler(FileSystemEventHandler):
        def __init__(self):
            self._debounce = {}

        def _schedule(self, path: str, func):
            now = time.time()
            if path in self._debounce and now - self._debounce[path] < 2:
                return
            self._debounce[path] = now
            threading.Thread(target=func, args=(path,)).start()

        def on_created(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            self._schedule(_to_rel_path(event.src_path), lambda p: _index_note(p))

        def on_modified(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            self._schedule(_to_rel_path(event.src_path), lambda p: _index_note(p))

        def on_deleted(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            self._schedule(_to_rel_path(event.src_path), lambda p: _delete_note(p))

        def on_moved(self, event):
            if event.is_directory:
                return
            if event.src_path.endswith(".md"):
                _delete_note(_to_rel_path(event.src_path))
            if event.dest_path.endswith(".md"):
                _index_note(_to_rel_path(event.dest_path))

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
    if "--watch" in sys.argv:
        watch()
    else:
        start = time.time()
        run_index()
        print(f"Elapsed: {time.time() - start:.1f}s")
