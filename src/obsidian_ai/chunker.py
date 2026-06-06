"""Phase 1: chunk + embed (nomic) + store in ChromaDB. No LLM chat calls."""

import concurrent.futures
import os
import threading
import time

from . import _index_utils as utils
from . import chroma_store, config, graph_store, llm_client, obsidian_client
from .logger import get_logger, log_error
from .wiki_links import extract_wiki_links

log = get_logger(__name__, log_file="indexer.log")


def prepare_note_data(
    path: str, content: str | None = None, *,
    _sanitized: str | None = None, _wc: int | None = None,
    _content_hash: str | None = None,
    _links: list[str] | None = None,
    _is_new: bool = False,
) -> dict | None:
    """Prepare note data — chunking + delta analysis. No LLM calls.

    Returns a dict with all data needed for ChromaDB upsert, or None if skipped.
    Entities/summary fields are empty — Phase 2 (extractor) fills them in.
    """
    try:
        raw = obsidian_client.get_note(path) if content is None else content
        tags = utils._extract_tags(raw)
        links = _links if _links is not None else extract_wiki_links(raw)
        fm_fields = utils._extract_frontmatter_fields(raw)
        sanitized = _sanitized if _sanitized is not None else utils._sanitize(raw)
        wc = _wc if _wc is not None else utils._word_count(sanitized)
        if wc < config.skip_min_tokens:
            log.debug(f"Skipped (too short): {path} — {wc} words")
            return None

        title = fm_fields.pop("fm_title", None) or os.path.splitext(os.path.basename(path))[0]
        mtime = utils._get_file_mtime(path)

        # Chunk-level delta analysis
        heading_chunks = utils.chunk_text_heading_aware(sanitized)
        new_chunks_text = [chunk for _, chunk in heading_chunks]
        new_hashes = [utils.compute_hash(c) for c in new_chunks_text]

        old_hash_to_id: dict[str, str] = {}
        old_hash_to_idx: dict[str, int] = {}
        if not _is_new:
            try:
                old_ids, old_metas, old_docs = chroma_store.get_chunks_by_path(path)
                for oid, ometa, odoc in zip(old_ids, old_metas, old_docs, strict=False):
                    if odoc:
                        oh = utils.compute_hash(odoc)
                        old_hash_to_id[oh] = oid
                        old_hash_to_idx[oh] = ometa.get("chunk", 0)
            except Exception:
                pass

        chunks_to_embed_indices = []
        embed_indices_set = set()
        for i, h in enumerate(new_hashes):
            if h not in old_hash_to_id:
                chunks_to_embed_indices.append(i)
                embed_indices_set.add(i)

        old_hashes_set = set(old_hash_to_id)
        new_hashes_set = set(new_hashes)
        stale_hashes = old_hashes_set - new_hashes_set
        stale_ids = [old_hash_to_id[h] for h in stale_hashes if h in old_hash_to_id]

        return {
            "path": path,
            "title": title,
            "mtime": mtime,
            "tags": tags,
            "links": links,
            "fm_fields": fm_fields,
            "wc": wc,
            "entities_str": "",
            "summary": "",
            "heading_chunks": heading_chunks,
            "new_chunks_text": new_chunks_text,
            "new_hashes": new_hashes,
            "chunks_to_embed_indices": chunks_to_embed_indices,
            "embed_indices_set": embed_indices_set,
            "stale_ids": stale_ids,
            "old_hash_to_id": old_hash_to_id,
            "texts_to_embed": [new_chunks_text[i] for i in chunks_to_embed_indices],
        }
    except Exception as e:
        log_error(log, f"FAILED: {path}", exc=e)
        return None


def finalize_note(prepared: dict, embeddings: list[list[float]], chunk_indices: list[int]) -> bool:
    """Upsert a prepared note's chunks into ChromaDB with pre-computed embeddings."""
    try:
        path = prepared["path"]
        embed_iter = iter(embeddings)
        embed_indices_set = set(chunk_indices)
        stale_ids = prepared["stale_ids"]
        heading_chunks = prepared["heading_chunks"]
        stale_fallback = False

        for i, (heading, chunk) in enumerate(heading_chunks):
            metadata = utils._build_metadata(
                path, prepared["title"], i, prepared["wc"], heading,
                prepared["tags"], prepared["links"], prepared["mtime"],
                prepared["entities_str"], prepared["fm_fields"],
                summary=prepared["summary"],
            )
            if i in embed_indices_set:
                embedding = next(embed_iter)
                chroma_store.upsert(path=path, chunk_idx=i, embedding=embedding, metadata=metadata, document=chunk)

        if stale_ids:
            try:
                chroma_store._ensure_init()
                chroma_store._collection.delete(ids=stale_ids)
                log.debug(f"Delta: removed {len(stale_ids)} stale chunks for {path}")
            except Exception:
                stale_fallback = True

        if stale_fallback:
            chroma_store.delete_by_path(path)
            for i, (heading, chunk) in enumerate(heading_chunks):
                metadata = utils._build_metadata(
                    path, prepared["title"], i, prepared["wc"], heading,
                    prepared["tags"], prepared["links"], prepared["mtime"],
                    prepared["entities_str"], prepared["fm_fields"],
                    summary=prepared["summary"],
                )
                chroma_store.upsert(path=path, chunk_idx=i, embedding=embeddings[i] if i in embed_indices_set else None,
                                   metadata=metadata, document=chunk)

        if not prepared["texts_to_embed"] and not stale_ids:
            log.debug(f"Delta: no changes detected for {path}")

        log.info(f"Chunked: {path} ({len(heading_chunks)} chunks, tags={prepared['tags']})")
        return True
    except Exception as e:
        log_error(log, f"FAILED: {path}", exc=e)
        return False


def index_note(path: str, content: str | None = None, *,
               _sanitized: str | None = None, _wc: int | None = None,
               _content_hash: str | None = None,
               _links: list[str] | None = None,
               _is_new: bool = False) -> bool:
    """Chunk and embed a single note. No LLM calls."""
    prepared = prepare_note_data(
        path, content=content,
        _sanitized=_sanitized, _wc=_wc,
        _content_hash=_content_hash,
        _links=_links, _is_new=_is_new,
    )
    if prepared is None:
        return False

    texts = prepared["texts_to_embed"]
    indices = prepared["chunks_to_embed_indices"]
    embeddings = llm_client.batch_embed(texts) if texts else []
    return finalize_note(prepared, embeddings, indices)


def delete_note(path: str) -> bool:
    try:
        chroma_store.delete_by_path(path)
        graph_store.remove_node(path)
        graph_store.save()
        log.info(f"Deleted from index: {path}")
        return True
    except Exception as e:
        log_error(log, f"DELETE FAILED: {path}", exc=e)
        return False


def run_chunking(folder: str | None = None):
    """Phase 1: chunk + embed + store all notes (or a folder). No LLM calls."""
    all_notes = obsidian_client.list_all_notes()
    notes = all_notes
    if folder:
        folder_prefix = folder.strip("/").rstrip("/") + "/"
        notes = [n for n in all_notes if n == folder or n.startswith(folder_prefix)]
        log.info(f"Filtered to folder '{folder}' — {len(notes)}/{len(all_notes)} notes")

    log.info(f"Starting chunking — {len(notes)} notes found")

    hash_map = utils.load_hash_map()
    log.info(f"Loaded {len(hash_map)} content hashes from disk")

    mtime_map = utils.load_mtime_map()
    if not mtime_map:
        mtime_map = utils._build_stored_mtime_map()
        log.info(f"Built mtime map from ChromaDB with {len(mtime_map)} entries")
    else:
        log.info(f"Loaded {len(mtime_map)} mtime entries from disk")

    # When folder-scoped, create filtered views for change detection
    if folder:
        folder_prefix = folder.strip("/").rstrip("/") + "/"
        _hash_map = {k: v for k, v in hash_map.items() if k.startswith(folder_prefix)}
        _mtime_map = {k: v for k, v in mtime_map.items() if k.startswith(folder_prefix)}
    else:
        _hash_map = hash_map
        _mtime_map = mtime_map

    indexed = 0
    skipped = 0
    unchanged = 0
    failed = 0
    interrupted = False

    try:
        log.info("Reading notes...")
        all_contents: dict[str, str] = {}
        content_hashes: dict[str, str] = {}
        changed_count = 0
        unchanged_count = 0
        total_notes = len(notes)
        _read_log_interval = max(1, total_notes // 4)
        _read_lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.read_workers) as ex:
            fut_to_path = {ex.submit(utils._read_one_note, p): p for p in notes}
            for completed_idx, fut in enumerate(concurrent.futures.as_completed(fut_to_path), 1):
                result = fut.result()
                if result is not None:
                    path, raw, current_hash = result
                    with _read_lock:
                        all_contents[path] = raw
                        content_hashes[path] = current_hash
                        stored_hash = _hash_map.get(path)
                        if stored_hash and stored_hash == current_hash:
                            unchanged_count += 1
                        else:
                            changed_count += 1
                if completed_idx % _read_log_interval == 0 or completed_idx == total_notes:
                    log.info(f"Read {completed_idx}/{total_notes} notes")

        deleted = [p for p in _hash_map if p not in notes]
        log.info(f"Read {len(notes)} notes — {changed_count} changed, "
                 f"{unchanged_count} unchanged, {len(deleted)} deleted")

        log.info("Updating graph...")
        links_cache: dict[str, list[str]] = {}
        graph_total = len(deleted) + sum(
            1 for p in notes
            if content_hashes.get(p) != _hash_map.get(p) and all_contents.get(p) is not None
        )
        graph_done = 0
        _graph_log_interval = max(1, graph_total // 4)
        for path in deleted:
            graph_store.remove_node(path)
            hash_map.pop(path, None)
            graph_done += 1
            if graph_done % _graph_log_interval == 0 or graph_done == graph_total:
                log.info(f"Graph update: {graph_done}/{graph_total}")
        for path in notes:
            content = all_contents.get(path)
            if content is None:
                continue
            stored_hash = _hash_map.get(path)
            current_hash = content_hashes.get(path)
            if stored_hash and current_hash and stored_hash == current_hash:
                continue
            graph_store.remove_node(path)
            graph_store.register_title(path)
            links = extract_wiki_links(content)
            links_cache[path] = links
            for link in links:
                resolved = graph_store.resolve_link(link)
                if resolved and resolved != path:
                    graph_store.add_edge(path, resolved)
            graph_done += 1
            if graph_done % _graph_log_interval == 0 or graph_done == graph_total:
                log.info(f"Graph update: {graph_done}/{graph_total}")
        graph_store.save()
        log.info(f"Graph updated: {graph_store.node_count()} nodes")

        changed_paths: list[str] = []
        new_paths: set[str] = set()
        for path in notes:
            stored_hash = _hash_map.get(path)
            current_hash = content_hashes.get(path)
            is_new = stored_hash is None
            if stored_hash and current_hash and stored_hash == current_hash:
                unchanged += 1
                log.debug(f"Skipped (hash unchanged): {path}")
                continue
            if utils._should_skip_by_mtime(path, _mtime_map):
                unchanged += 1
                log.debug(f"Skipped (mtime unchanged): {path}")
                continue
            changed_paths.append(path)
            if is_new:
                new_paths.add(path)

        max_workers = utils._embed_workers_for_current_machine()
        hash_map_lock = threading.Lock()
        batch_size = config.index_batch_size
        total_batches = (len(changed_paths) + batch_size - 1) // batch_size
        log.info(f"Processing {len(changed_paths)} changed notes in {total_batches} batch(es) "
                 f"of up to {batch_size} with {max_workers} workers")

        def _prepare_one(path: str, _all_contents: dict, _links_cache: dict,
                         _prepare_lock: threading.Lock, _all_prepared: list,
                         _new_paths: set) -> str:
            try:
                content = _all_contents.get(path)
                if content is None:
                    content = obsidian_client.get_note(path)
                sanitized = utils._sanitize(content)

                wc = utils._word_count(sanitized)
                if wc < config.skip_min_tokens:
                    log.debug(f"Skipped (too short): {path} — {wc} words")
                    return "skipped"

                current_hash = content_hashes.get(path) or utils.compute_hash(content)
                prepared = prepare_note_data(
                    path, content=content,
                    _sanitized=sanitized, _wc=wc, _content_hash=current_hash,
                    _links=_links_cache.get(path),
                    _is_new=path in _new_paths,
                )
                if prepared is not None:
                    prepared["_hash"] = current_hash
                    prepared["_is_new"] = path in _new_paths
                    with _prepare_lock:
                        _all_prepared.append(prepared)
                    return "prepared"
                return "failed"
            except Exception as e:
                log_error(log, f"FAILED: {path}", exc=e)
                return "failed"

        for batch_num, batch_start in enumerate(
            range(0, len(changed_paths), batch_size), 1
        ):
            batch_paths = changed_paths[batch_start:batch_start + batch_size]
            log.info(f"Batch {batch_num}/{total_batches}: {len(batch_paths)} notes")

            all_prepared: list[dict] = []
            prepare_lock = threading.Lock()

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_to_path = {
                    ex.submit(
                        _prepare_one, p, all_contents, links_cache,
                        prepare_lock, all_prepared, new_paths,
                    ): p for p in batch_paths
                }
                batch_changed = len(batch_paths)
                _proc_log_interval = max(1, batch_changed // 10)
                _next_proc_log = _proc_log_interval
                for completed_idx, fut in enumerate(
                    concurrent.futures.as_completed(fut_to_path), 1
                ):
                    status = fut.result()
                    if status == "prepared":
                        indexed += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                    if completed_idx >= _next_proc_log or completed_idx == batch_changed:
                        log.info(f"  Prepare: {completed_idx}/{batch_changed} — "
                                 f"{indexed}/{len(changed_paths)} total prepared")
                        _next_proc_log += _proc_log_interval

            if all_prepared:
                all_texts: list[str] = []
                text_map: list[tuple[int, int]] = []
                for note_idx, pn in enumerate(all_prepared):
                    for ci in pn["chunks_to_embed_indices"]:
                        all_texts.append(pn["new_chunks_text"][ci])
                        text_map.append((note_idx, ci))

                log.info(f"  Batch-embedding {len(all_texts)} chunks ({len(all_prepared)} notes)")
                all_embeddings = llm_client.batch_embed(all_texts) if all_texts else []

                note_embeddings: dict[int, tuple[list[list[float]], list[int]]] = {}
                for map_idx, (note_idx, ci) in enumerate(text_map):
                    if note_idx not in note_embeddings:
                        note_embeddings[note_idx] = ([], [])
                    note_embeddings[note_idx][0].append(all_embeddings[map_idx])
                    note_embeddings[note_idx][1].append(ci)

                finalized = 0
                skipped_final = 0
                for note_idx, pn in enumerate(all_prepared):
                    emb_list, emb_indices = note_embeddings.get(note_idx, ([], []))
                    success = finalize_note(pn, emb_list, emb_indices)
                    if success:
                        with hash_map_lock:
                            hash_map[pn["path"]] = pn["_hash"]
                            current_mtime = utils._get_file_mtime(pn["path"])
                            if current_mtime is not None:
                                mtime_map[pn["path"]] = current_mtime
                        finalized += 1
                    else:
                        skipped_final += 1

                log.info(f"  Batch {batch_num} finalized: {finalized} indexed, "
                         f"{skipped_final} failed")
            else:
                log.info(f"  Batch {batch_num}: no notes to embed")

            if batch_num < total_batches:
                cooldown = config.llm_call_delay * 3
                log.info(f"  Cooling down for {cooldown:.1f}s before next batch...")
                time.sleep(cooldown)

    except KeyboardInterrupt:
        interrupted = True
        log.warning("Interrupted — saving state...")

    if folder:
        # Partial index: only GC stale entries within the folder scope
        stale = [p for p in _hash_map if p not in all_contents]
    else:
        known_paths = set(all_contents.keys())
        stale = [p for p in hash_map if p not in known_paths]
    if stale:
        for p in stale:
            del hash_map[p]
        log.info(f"GC removed {len(stale)} stale content hash entries")
    utils.save_hash_map(hash_map)
    utils.save_mtime_map(mtime_map)
    log.info(f"Mtime map saved — {len(mtime_map)} entries")
    llm_client.save_embed_cache()
    log.info("Embed cache saved to disk")
    if interrupted:
        log.info(f"Partial state saved — re-run to continue "
                 f"({indexed} indexed, {skipped} skipped, {failed} failed)")
    else:
        log.info(f"Chunking done — Indexed: {indexed}, Skipped: {skipped}, "
                 f"Unchanged: {unchanged}, Failed: {failed}")

    return len([p for p in notes if utils.compute_hash(obsidian_client.get_note(p)) != hash_map.get(p)])


def watch_chunking():
    """Watch mode for Phase 1 — chunk+embed on file changes. No LLM calls."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

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
    log.info(f"Starting chunker watcher on vault: {vault_path}")

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
                        index_note(path, content=content, _content_hash=current_hash)
                        graph_store.register_title(path)
                        for link in extract_wiki_links(content):
                            resolved = graph_store.resolve_link(link)
                            if resolved and resolved != path:
                                graph_store.add_edge(path, resolved)
                        graph_store.save()
                        with _pending_lock:
                            hash_map[path] = current_hash
                        utils.save_hash_map(hash_map)
                    elif action == "delete":
                        delete_note(path)
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
    log.info("Chunker watcher started — press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start = time.time()
    run_chunking()
    print(f"Chunking elapsed: {time.time() - start:.1f}s")
