import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time

from . import chroma_store, config, entity_store, graph_store, llm_client, obsidian_client, pipelines

from .frontmatter import add_tags as fm_add_tags, parse as fm_parse
from .logger import get_logger, log_error
from .wiki_links import extract_wiki_links

log = get_logger(__name__, log_file="indexer.log")

HASH_MAP_PATH = os.path.join(config.data_dir, "content_hashes.json")
ENTITY_CACHE_PATH = os.path.join(config.data_dir, "entity_cache.json")
SUMMARY_CACHE_PATH = os.path.join(config.data_dir, "summary_cache.json")

# Persistent entity cache: content_hash -> list[dict] (shared across threads).
_entity_cache: dict[str, list[dict]] = {}
_entity_cache_lock = threading.Lock()

# Persistent summary cache: content_hash -> str
_summary_cache: dict[str, str] = {}
_summary_cache_lock = threading.Lock()

# Limit concurrent Ollama chat calls (entity extraction + summary) to 1
# to prevent overwhelming Ollama during parallel indexing. Embeddings
# (POST /api/embeddings) are unaffected and still run in parallel.
_llm_chat_lock = threading.Semaphore(1)

# When True, skip LLM-based entity extraction for faster indexing.
# Existing entity data is preserved (not cleared) so entity search still works.
SKIP_ENTITIES = False

# When True, skip LLM-based summary generation for faster indexing.
SKIP_SUMMARIES = False

SUMMARY_SYSTEM = (
    "You are a note summarizer. Given a note from an Obsidian vault, "
    "produce a concise 1-2 sentence summary capturing the key information. "
    "Be factual, specific, and use the same language as the original note. "
    "Return ONLY the summary text — no preamble, no labels."
)


def _compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _load_hash_map() -> dict[str, str]:
    try:
        if os.path.isfile(HASH_MAP_PATH):
            with open(HASH_MAP_PATH, "r", encoding="utf-8") as f:
                return dict(json.load(f))
    except Exception as e:
        log.warning(f"Failed to load content hash map: {e}")
    return {}


def _save_hash_map(hash_map: dict[str, str]) -> None:
    try:
        os.makedirs(os.path.dirname(HASH_MAP_PATH), exist_ok=True)
        with open(HASH_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(hash_map, f, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Failed to save content hash map: {e}")


def _load_entity_cache() -> dict[str, list[dict]]:
    try:
        if os.path.isfile(ENTITY_CACHE_PATH):
            with open(ENTITY_CACHE_PATH, "r", encoding="utf-8") as f:
                return dict(json.load(f))
    except Exception as e:
        log.warning(f"Failed to load entity cache: {e}")
    return {}


def _save_entity_cache(cache: dict[str, list[dict]]) -> None:
    try:
        os.makedirs(os.path.dirname(ENTITY_CACHE_PATH), exist_ok=True)
        with open(ENTITY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Failed to save entity cache: {e}")


def _load_summary_cache() -> dict[str, str]:
    try:
        if os.path.isfile(SUMMARY_CACHE_PATH):
            with open(SUMMARY_CACHE_PATH, "r", encoding="utf-8") as f:
                return dict(json.load(f))
    except Exception as e:
        log.warning(f"Failed to load summary cache: {e}")
    return {}


def _save_summary_cache(cache: dict[str, str]) -> None:
    try:
        os.makedirs(os.path.dirname(SUMMARY_CACHE_PATH), exist_ok=True)
        with open(SUMMARY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Failed to save summary cache: {e}")


def _generate_summary_cached(sanitized: str, content_hash: str | None) -> str:
    """Generate a 1-2 sentence summary with a persistent cache by content hash. Thread-safe."""
    if content_hash and not SKIP_SUMMARIES:
        with _summary_cache_lock:
            cached = _summary_cache.get(content_hash)
        if cached is not None:
            return cached

    if SKIP_SUMMARIES:
        return ""

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"Note content:\n\n{sanitized[:3000]}"},
    ]

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            summary = llm_client.chat(messages, think=False).strip()
            if content_hash:
                with _summary_cache_lock:
                    _summary_cache[content_hash] = summary
            return summary
        except Exception as e:
            last_exc = e
            if attempt < 2:
                wait = 2 ** attempt
                log.warning(f"Summary generation attempt {attempt + 1} failed, retrying in {wait}s: {e}")
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _extract_entities_cached(sanitized: str, path: str, content_hash: str | None) -> list[dict]:
    """Extract entities with a persistent cache by content hash. Thread-safe."""
    if content_hash:
        with _entity_cache_lock:
            cached = _entity_cache.get(content_hash)
        if cached is not None:
            return cached

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            entities = pipelines.extract_entities(sanitized, path=path)
            if content_hash:
                with _entity_cache_lock:
                    _entity_cache[content_hash] = entities
            return entities
        except Exception as e:
            last_exc = e
            if attempt < 2:
                wait = 2 ** attempt
                log.warning(f"Entity extraction attempt {attempt + 1} failed for {path}, retrying in {wait}s: {e}")
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _embed_workers_for_current_machine() -> int:
    """Pick a safe embedding concurrency level.

    Default: up to 4 workers, bounded by config.embed_worker_ceil.
    """
    try:
        cpu = os.cpu_count() or 1
        workers = min(config.embed_worker_ceil, max(config.embed_worker_floor, cpu // 2))
        return int(workers)
    except Exception:
        return config.embed_worker_floor


def _word_count(text: str) -> int:

    return len(text.split())


def _sanitize(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str, size: int = config.chunk_size, overlap: int = config.chunk_overlap) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks


_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split text into (heading_path, body) sections based on Markdown headings.

    Returns list of tuples where heading_path accumulates parent headings
    (e.g., "# Setup > ## Config") and body is the section text.
    Text before the first heading has an empty heading_path.
    """
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [("", text)]

    # Text before first heading
    if matches[0].start() > 0:
        sections.append(("", text[:matches[0].start()].strip()))

    # Build heading hierarchy
    heading_stack: list[tuple[int, str]] = []  # (level, text)

    for match in matches:
        level = len(match.group(1))
        heading_text = match.group(2).strip()

        # Pop headings deeper than current level
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()

        heading_stack.append((level, heading_text))

        # Build heading path
        path = " > ".join(f"{'#' * h_level} {h_text}" for h_level, h_text in heading_stack)

        # Find section end
        start = match.end()
        end = matches[matches.index(match) + 1].start() if matches.index(match) + 1 < len(matches) else len(text)
        sections.append((path, text[start:end].strip()))

    return sections


def chunk_text_heading_aware(text: str, size: int = config.chunk_size, overlap: int = config.chunk_overlap) -> list[tuple[str, str]]:
    """Split text into chunks respecting Markdown heading structure.

    Returns list of (heading_path, chunk_content) tuples.
    Each chunk is prefixed with its heading path for structural context.
    Sections smaller than `size` words are kept intact; larger sections
    fall back to word-boundary chunking.
    """
    sections = split_by_headings(text)
    chunks: list[tuple[str, str]] = []

    for heading_path, body in sections:
        if not body:
            continue

        words = body.split()
        prefix = f"{heading_path}\n\n" if heading_path else ""

        if len(words) <= size:
            # Section fits in one chunk
            chunks.append((heading_path, f"{prefix}{body}"))
        else:
            # Fall back to word-boundary chunking within this section
            sub_chunks = chunk_text(body, size=size, overlap=overlap)
            for sub_chunk in sub_chunks:
                chunks.append((heading_path, f"{prefix}{sub_chunk}"))

    return chunks


def _extract_frontmatter_fields(raw_content: str) -> dict:
    """Extract frontmatter fields beyond tags for ChromaDB metadata.

    Returns dict with any of: created, modified, aliases_str, cssclasses_str, fm_title.
    """
    meta, _ = fm_parse(raw_content)
    fields: dict = {}

    if "created" in meta and meta["created"] is not None:
        fields["created"] = str(meta["created"])
    if "modified" in meta and meta["modified"] is not None:
        fields["modified"] = str(meta["modified"])
    if "aliases" in meta and meta["aliases"]:
        aliases = meta["aliases"]
        if isinstance(aliases, str):
            aliases = [aliases]
        if isinstance(aliases, list):
            fields["aliases_str"] = _links_to_meta([str(a) for a in aliases])
    if "cssclasses" in meta and meta["cssclasses"]:
        cssclasses = meta["cssclasses"]
        if isinstance(cssclasses, str):
            cssclasses = [cssclasses]
        if isinstance(cssclasses, list):
            fields["cssclasses_str"] = _links_to_meta([str(c) for c in cssclasses])
    if "title" in meta and meta["title"]:
        fields["fm_title"] = str(meta["title"])

    return fields


def _get_file_mtime(path: str) -> float | None:
    """Get the filesystem mtime for a vault-relative path, or None if unavailable."""
    if not config.vault_path:
        return None
    abs_path = os.path.join(config.vault_path, path)
    try:
        return os.path.getmtime(abs_path)
    except OSError:
        return None


def _should_skip_by_mtime(path: str, stored_mtime_map: dict[str, float]) -> bool:
    """Check if a note's mtime is unchanged since last index. Returns True to skip."""
    current_mtime = _get_file_mtime(path)
    if current_mtime is None:
        return False
    stored_mtime = stored_mtime_map.get(path)
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


def _links_to_meta(links: list[str]) -> str:
    """Serialize wiki-link targets to a safely searchable string: ",link1,link2,"""
    return "," + ",".join(links) + ","


def _build_metadata(
    path: str, title: str, chunk_idx: int, word_count: int, heading: str,
    tags: list[str], links: list[str], mtime: float | None,
    entities_str: str, fm_fields: dict, summary: str = "",
) -> dict:
    metadata: dict = {
        "path": path,
        "title": title,
        "chunk": chunk_idx,
        "word_count": word_count,
        "heading": heading,
    }
    if tags:
        metadata["tags_str"] = _tags_to_meta(tags)
    if links:
        metadata["links_str"] = _links_to_meta(links)
    if mtime is not None:
        metadata["mtime"] = mtime
    if entities_str:
        metadata["entities_str"] = entities_str
    if summary:
        metadata["summary"] = summary
    metadata.update(fm_fields)
    return metadata


def _index_note(path: str, content: str | None = None, *, embed_workers: int = 1,
                _sanitized: str | None = None, _wc: int | None = None,
                _content_hash: str | None = None) -> bool:
    """Index a single note. Returns True if successful."""
    try:

        if content is None:
            raw = obsidian_client.get_note(path)
        else:
            raw = content

        tags = _extract_tags(raw)
        links = extract_wiki_links(raw)
        fm_fields = _extract_frontmatter_fields(raw)
        if _sanitized is not None:
            sanitized = _sanitized
        else:
            sanitized = _sanitize(raw)
        wc = _wc if _wc is not None else _word_count(sanitized)
        if wc < config.skip_min_tokens:
            log.debug(f"Skipped (too short): {path} — {wc} words")
            return False
        chroma_store.delete_by_path(path)
        heading_chunks = chunk_text_heading_aware(sanitized)

        # Entity extraction (skippable via SKIP_ENTITIES flag)
        entities = []
        if not SKIP_ENTITIES:
            try:
                with _llm_chat_lock:
                    entities = _extract_entities_cached(sanitized, path, _content_hash)
            except Exception as e:
                log.warning(f"Entity extraction failed for {path}: {e}")
        entities_str = ""

        # Summary generation (skippable via SKIP_SUMMARIES flag)
        summary = ""
        if not SKIP_SUMMARIES:
            try:
                with _llm_chat_lock:
                    summary = _generate_summary_cached(sanitized, _content_hash)
            except Exception as e:
                log.warning(f"Summary generation failed for {path}: {e}")
        if entities:
            serialised = ",".join(f"{e['type']}:{e['name']}" for e in entities)
            entities_str = f",{serialised},"
            for ent in entities:
                entity_store.add(
                    name=ent["name"],
                    type=ent["type"],
                    confidence=ent["confidence"],
                    path=path,
                    chunk_idx=0,
                    context=sanitized[:200],
                )

        # Optional: reuse embeddings for identical chunk text within this run.
        # (Keeps memory bounded; only scoped to a single note.)
        local_embedding_cache: dict[str, list[float]] = {}
        _cache_lock = threading.Lock()

        title = fm_fields.pop("fm_title", None) or os.path.splitext(os.path.basename(path))[0]
        mtime = _get_file_mtime(path)
        def _embed_one(args: tuple[int, str, str]) -> tuple[int, str, list[float]]:
            i_, heading_, chunk_ = args
            with _cache_lock:
                cached = local_embedding_cache.get(chunk_)
            if cached is not None:
                return i_, heading_, cached
            emb = llm_client.embed(chunk_)
            with _cache_lock:
                local_embedding_cache[chunk_] = emb
            return i_, heading_, emb


        if embed_workers <= 1 or len(heading_chunks) <= 1:
            for i, (heading, chunk) in enumerate(heading_chunks):
                embedding = llm_client.embed(chunk)
                metadata = _build_metadata(path, title, i, wc, heading, tags, links, mtime, entities_str, fm_fields, summary=summary)
                chroma_store.upsert(path=path, chunk_idx=i, embedding=embedding, metadata=metadata, document=chunk)
        else:
            embed_workers = max(2, embed_workers)
            # Embed chunks in parallel, then upsert serially (Chroma writes).
            with concurrent.futures.ThreadPoolExecutor(max_workers=embed_workers) as ex:
                futures = [ex.submit(_embed_one, (i, heading, chunk)) for i, (heading, chunk) in enumerate(heading_chunks)]
                for fut in concurrent.futures.as_completed(futures):
                    i, heading, embedding = fut.result()
                    chunk = heading_chunks[i][1]
                    metadata = _build_metadata(path, title, i, wc, heading, tags, links, mtime, entities_str, fm_fields, summary=summary)
                    chroma_store.upsert(path=path, chunk_idx=i, embedding=embedding, metadata=metadata, document=chunk)

        log.info(f"Indexed: {path} ({len(heading_chunks)} chunks, tags={tags})")
        return True
    except Exception as e:
        log_error(log, f"FAILED: {path}", exc=e)
        return False


def _delete_note(path: str) -> bool:
    """Delete a note from the index and graph. Returns True if successful."""
    try:
        chroma_store.delete_by_path(path)
        graph_store.remove_node(path)
        graph_store.save()
        log.info(f"Deleted from index: {path}")
        return True
    except Exception as e:
        log_error(log, f"DELETE FAILED: {path}", exc=e)
        return False


def _build_stored_mtime_map(timeout: float = 15) -> dict[str, float]:
    """Build a {path: stored_mtime} map from Chroma metadata once per run.
    
    Uses a timeout to avoid hanging if ChromaDB is unresponsive.
    Returns empty dict on timeout/error (re-indexes everything).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(chroma_store.get_all_documents)
        try:
            _, _, metadatas = fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log.warning("ChromaDB get_all_documents timed out — will re-index all notes")
            return {}
        except Exception:
            return {}

    m: dict[str, float] = {}
    for meta in metadatas:
        path = meta.get("path")
        mt = meta.get("mtime")
        if not path or mt is None:
            continue
        # All chunks for the same note share the same mtime; keep the first.
        if path not in m:
            try:
                m[path] = float(mt)
            except (TypeError, ValueError):
                continue
    return m


def run_index():
    notes = obsidian_client.list_all_notes()

    log.info(f"Starting index — {len(notes)} notes found")

    # Reset entity store (skip if user opted out of entity extraction)
    if not SKIP_ENTITIES:
        entity_store.clear()

    # Load stored content hashes for incremental skip
    hash_map = _load_hash_map()
    log.info(f"Loaded {len(hash_map)} content hashes from disk")

    # Load persistent entity cache
    _entity_cache.clear()
    _entity_cache.update(_load_entity_cache())
    log.info(f"Loaded {len(_entity_cache)} entity cache entries")

    # Load persistent summary cache
    _summary_cache.clear()
    _summary_cache.update(_load_summary_cache())
    log.info(f"Loaded {len(_summary_cache)} summary cache entries")

    indexed = 0
    skipped = 0
    unchanged = 0
    failed = 0
    interrupted = False

    try:
        # Read all notes, compute hashes, detect changes
        log.info("Reading notes...")
        all_contents: dict[str, str] = {}
        content_hashes: dict[str, str] = {}
        changed_count = 0
        unchanged_count = 0
        total_notes = len(notes)
        _read_log_interval = max(1, total_notes // 4)
        for i, path in enumerate(notes):
            try:
                raw = obsidian_client.get_note(path)
                all_contents[path] = raw
                current_hash = _compute_hash(raw)
                content_hashes[path] = current_hash
                stored_hash = hash_map.get(path)
                if stored_hash and stored_hash == current_hash:
                    unchanged_count += 1
                else:
                    changed_count += 1
            except Exception:
                pass
            if (i + 1) % _read_log_interval == 0 or i == total_notes - 1:
                log.info(f"Read {i + 1}/{total_notes} notes")

        # Detect deleted notes (in hash map but no longer in vault)
        deleted = [p for p in hash_map if p not in notes]
        log.info(f"Read {len(notes)} notes — {changed_count} changed, {unchanged_count} unchanged, {len(deleted)} deleted")

        # Incremental graph update (instead of full rebuild)
        log.info("Updating graph...")
        graph_total = len(deleted) + sum(1 for p in notes if content_hashes.get(p) != hash_map.get(p) and all_contents.get(p) is not None)
        graph_done = 0
        _graph_log_interval = max(1, graph_total // 4)
        for path in deleted:
            log.debug(f"Graph: removing deleted note {path}")
            graph_store.remove_node(path)
            hash_map.pop(path, None)
            graph_done += 1
            if graph_done % _graph_log_interval == 0 or graph_done == graph_total:
                log.info(f"Graph update: {graph_done}/{graph_total}")
        for path in notes:
            content = all_contents.get(path)
            if content is None:
                continue
            stored_hash = hash_map.get(path)
            current_hash = content_hashes.get(path)
            if stored_hash and current_hash and stored_hash == current_hash:
                continue
            log.debug(f"Graph: updating edges for {path}")
            graph_store.remove_node(path)
            graph_store.register_title(path)
            links = extract_wiki_links(content)
            for link in links:
                resolved = graph_store.resolve_link(link)
                if resolved and resolved != path:
                    graph_store.add_edge(path, resolved)
            graph_done += 1
            if graph_done % _graph_log_interval == 0 or graph_done == graph_total:
                log.info(f"Graph update: {graph_done}/{graph_total}")
        graph_store.save()
        log.info(f"Graph updated: {graph_store.node_count()} nodes")

        stored_mtime_map = _build_stored_mtime_map()

        # Pre-filter: find notes that actually need processing
        changed_paths: list[str] = []
        for path in notes:
            stored_hash = hash_map.get(path)
            current_hash = content_hashes.get(path)
            if stored_hash and current_hash and stored_hash == current_hash:
                unchanged += 1
                log.debug(f"Skipped (hash unchanged): {path}")
                continue

            if _should_skip_by_mtime(path, stored_mtime_map):
                unchanged += 1
                log.debug(f"Skipped (mtime unchanged): {path}")
                continue

            changed_paths.append(path)

        # Parallel note processing
        max_workers = _embed_workers_for_current_machine()
        hash_map_lock = threading.Lock()
        log.info(f"Processing {len(changed_paths)} changed notes with {max_workers} workers")

        def _process_one(path: str) -> str:
            """Process a single changed note. Returns 'indexed', 'skipped', or 'failed'."""
            try:
                content = all_contents.get(path)
                if content is None:
                    content = obsidian_client.get_note(path)

                sanitized = _sanitize(content)
                wc = _word_count(sanitized)
                if wc < config.skip_min_tokens:
                    log.debug(f"Skipped (too short): {path} — {wc} words")
                    return "skipped"

                current_hash = content_hashes.get(path) or _compute_hash(content)
                success = _index_note(path, content=content, embed_workers=1,
                                      _sanitized=sanitized, _wc=wc, _content_hash=current_hash)
                if success:
                    with hash_map_lock:
                        hash_map[path] = current_hash
                    return "indexed"
                return "failed"
            except Exception as e:
                log_error(log, f"FAILED: {path}", exc=e)
                return "failed"

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_path = {ex.submit(_process_one, p): p for p in changed_paths}
            total_changed = len(changed_paths)
            completed = 0
            _proc_log_interval = max(1, total_changed // 10)
            _next_proc_log = _proc_log_interval
            for fut in concurrent.futures.as_completed(fut_to_path):
                status = fut.result()
                if status == "indexed":
                    indexed += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                completed += 1
                if completed >= _next_proc_log or completed == total_changed:
                    log.info(f"Progress: {completed}/{total_changed} — {indexed} indexed, {skipped} skipped, {failed} failed")
                    _next_proc_log += _proc_log_interval

    except KeyboardInterrupt:
        interrupted = True
        log.warning("Interrupted — finishing in-progress notes and saving state...")

    _save_hash_map(hash_map)
    entity_store.save()
    log.info(f"Entity store saved — {entity_store.stats()['total_entities']} entities")
    _save_entity_cache(_entity_cache)
    log.info(f"Entity cache saved — {len(_entity_cache)} entries")
    _save_summary_cache(_summary_cache)
    log.info(f"Summary cache saved — {len(_summary_cache)} entries")
    if interrupted:
        log.info(f"Partial state saved — re-run to continue ({indexed} indexed, {skipped} skipped, {failed} failed)")
    else:
        log.info(f"Done — Indexed: {indexed}, Skipped: {skipped}, Unchanged (hash/mtime): {unchanged}, Failed: {failed}")


def add_tags_to_note(path: str, tags: list[str]) -> None:
    """Add tags to a note's YAML frontmatter."""
    content = obsidian_client.get_note(path)
    new_content = fm_add_tags(content, tags)
    obsidian_client.put_note(path, new_content)


def watch():
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

    # Coalescing queue: single worker thread, events merged by path.
    _pending: dict[str, tuple[str, float]] = {}  # path -> (action, timestamp)
    _pending_lock = threading.Lock()
    _wake = threading.Event()

    _DEBOUNCE_SECS = 2.0

    def _queue_event(path: str, action: str) -> None:
        """Enqueue an index/delete/rename action, coalescing duplicates."""
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
                    if now - ts >= _DEBOUNCE_SECS
                ]
                for path, _ in ready:
                    del _pending[path]
            for path, action in ready:
                try:
                    if action == "index":
                        _index_note(path)
                        graph_store.register_title(path)
                        # Update graph edges for this note
                        content = obsidian_client.get_note(path)
                        for link in extract_wiki_links(content):
                            resolved = graph_store.resolve_link(link)
                            if resolved and resolved != path:
                                graph_store.add_edge(path, resolved)
                        graph_store.save()
                    elif action == "delete":
                        _delete_note(path)
                    elif action.startswith("rename:"):
                        old_path, new_path = action.split(":", 1)[1].split("->")
                        graph_store.rename_node(old_path, new_path)
                        graph_store.save()
                except Exception as e:
                    log_error(log, f"Worker failed ({action}): {path}", exc=e)

    threading.Thread(target=_worker, daemon=True).start()

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
