import os
import re
import sys
import time
from . import obsidian_client
from . import llm_client
from . import chroma_store
from .frontmatter import add_tags as fm_add_tags
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


def _index_note(path: str, content: str | None = None) -> bool:
    """Index a single note. Returns True if successful."""
    try:
        if content is None:
            content = _sanitize(obsidian_client.get_note(path))
        else:
            content = _sanitize(content)
        wc = _word_count(content)
        if wc < SKIP_MIN_TOKENS:
            log.debug(f"Skipped (too short): {path} — {wc} words")
            return False
        chroma_store.delete_by_path(path)
        chunks = chunk_text(content)
        title = os.path.splitext(os.path.basename(path))[0]
        for i, chunk in enumerate(chunks):
            embedding = llm_client.embed(chunk)
            metadata = {
                "path": path,
                "title": title,
                "chunk": i,
                "word_count": wc,
            }
            chroma_store.upsert(path=path, chunk_idx=i, embedding=embedding, metadata=metadata)
        log.info(f"Indexed: {path} ({len(chunks)} chunks)")
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
    notes = obsidian_client.list_notes()
    log.info(f"Starting index — {len(notes)} notes found")
    indexed = 0
    skipped = 0
    failed = 0
    for path in notes:
        try:
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
    log.info(f"Done — Indexed: {indexed}, Skipped: {skipped}, Failed: {failed}")


def add_tags_to_note(path: str, tags: list[str]) -> None:
    """Add tags to a note's YAML frontmatter."""
    content = obsidian_client.get_note(path)
    new_content = fm_add_tags(content, tags)
    obsidian_client.put_note(path, new_content)


def watch():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    import threading

    vault_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    log.info(f"Starting watcher on vault")

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
            rel = event.src_path.replace("\\", "/")
            self._schedule(rel, lambda p: _index_note(p))

        def on_modified(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            rel = event.src_path.replace("\\", "/")
            self._schedule(rel, lambda p: _index_note(p))

        def on_deleted(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            rel = event.src_path.replace("\\", "/")
            self._schedule(rel, lambda p: _delete_note(p))

        def on_moved(self, event):
            if event.is_directory:
                return
            if event.src_path.endswith(".md"):
                old_rel = event.src_path.replace("\\", "/")
                _delete_note(old_rel)
            if event.dest_path.endswith(".md"):
                new_rel = event.dest_path.replace("\\", "/")
                _index_note(new_rel)

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
