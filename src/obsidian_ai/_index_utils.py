import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

from . import chroma_store, config, obsidian_client
from .frontmatter import parse as fm_parse
from .logger import get_logger
from .wiki_links import extract_wiki_links

log = get_logger(__name__)

HASH_MAP_PATH = os.path.join(config.data_dir, "content_hashes.json")
MTIME_MAP_PATH = os.path.join(config.data_dir, "mtime_map.json")

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_hash_map() -> dict[str, str]:
    try:
        if os.path.isfile(HASH_MAP_PATH):
            with open(HASH_MAP_PATH, encoding="utf-8") as f:
                return dict(json.load(f))
    except Exception as e:
        log.warning(f"Failed to load content hash map: {e}")
    return {}


def save_hash_map(hash_map: dict[str, str]) -> None:
    try:
        os.makedirs(os.path.dirname(HASH_MAP_PATH), exist_ok=True)
        with open(HASH_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(hash_map, f, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Failed to save content hash map: {e}")


def load_mtime_map() -> dict[str, float]:
    try:
        if os.path.isfile(MTIME_MAP_PATH):
            with open(MTIME_MAP_PATH, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {k: float(v) for k, v in data.items()}
    except Exception as e:
        log.warning(f"Failed to load mtime map: {e}")
    return {}


def save_mtime_map(m: dict[str, float]) -> None:
    try:
        os.makedirs(os.path.dirname(MTIME_MAP_PATH), exist_ok=True)
        with open(MTIME_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Failed to save mtime map: {e}")


def _word_count(text: str) -> int:
    return len(text.split())


def _sanitize(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    return re.sub(r'[^\S\r\n]+', ' ', text).strip()


def chunk_text(text: str, size: int = config.chunk_size, overlap: int = config.chunk_overlap) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks


def split_by_headings(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [("", text)]

    if matches[0].start() > 0:
        sections.append(("", text[:matches[0].start()].strip()))

    heading_stack: list[tuple[int, str]] = []

    for match in matches:
        level = len(match.group(1))
        heading_text = match.group(2).strip()

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()

        heading_stack.append((level, heading_text))

        path = " > ".join(f"{'#' * h_level} {h_text}" for h_level, h_text in heading_stack)

        start = match.end()
        end = matches[matches.index(match) + 1].start() if matches.index(match) + 1 < len(matches) else len(text)
        sections.append((path, text[start:end].strip()))

    return sections


def chunk_text_heading_aware(text: str, size: int = config.chunk_size, overlap: int = config.chunk_overlap) -> list[tuple[str, str]]:
    sections = split_by_headings(text)
    chunks: list[tuple[str, str]] = []

    for heading_path, body in sections:
        if not body:
            continue

        words = body.split()
        prefix = f"{heading_path}\n\n" if heading_path else ""

        if len(words) <= size:
            chunks.append((heading_path, f"{prefix}{body}"))
        else:
            sub_chunks = chunk_text(body, size=size, overlap=overlap)
            for sub_chunk in sub_chunks:
                chunks.append((heading_path, f"{prefix}{sub_chunk}"))

    return chunks


def _extract_frontmatter_fields(raw_content: str) -> dict:
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
    if not config.vault_path:
        return None
    abs_path = os.path.join(config.vault_path, path)
    try:
        return os.path.getmtime(abs_path)
    except OSError:
        return None


def _should_skip_by_mtime(path: str, stored_mtime_map: dict[str, float]) -> bool:
    current_mtime = _get_file_mtime(path)
    if current_mtime is None:
        return False
    stored_mtime = stored_mtime_map.get(path)
    if stored_mtime is None:
        return False
    return abs(float(stored_mtime) - current_mtime) < 0.001


def _extract_tags(raw_content: str) -> list[str]:
    meta, _ = fm_parse(raw_content)
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    return tags if isinstance(tags, list) else []


def _tags_to_meta(tags: list[str]) -> str:
    return "," + ",".join(tags) + ","


def _links_to_meta(links: list[str]) -> str:
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


def _build_stored_mtime_map(timeout: float = 15) -> dict[str, float]:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(chroma_store.get_all_documents)
        try:
            _, _, metadatas = fut.result(timeout=timeout)
        except TimeoutError:
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
        if path not in m:
            try:
                m[path] = float(mt)
            except (TypeError, ValueError):
                continue
    return m


def _embed_workers_for_current_machine() -> int:
    try:
        cpu = os.cpu_count() or 1
        workers = min(config.embed_worker_ceil, max(config.embed_worker_floor, cpu // 2))
        return int(workers)
    except Exception:
        return config.embed_worker_floor


def _read_one_note(path: str) -> tuple[str, str, str] | None:
    try:
        raw = obsidian_client.get_note(path)
        return path, raw, compute_hash(raw)
    except Exception:
        return None
