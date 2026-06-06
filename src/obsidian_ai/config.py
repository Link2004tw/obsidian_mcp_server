import os
from pathlib import Path

from dotenv import load_dotenv

from .logger import get_logger

log = get_logger("obsidian_ai.config")

# Load .env relative to this file's location, not the CWD
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path, override=True)

obsidian_host = os.getenv("OBSIDIAN_HOST", "localhost")
obsidian_port = int(os.getenv("OBSIDIAN_PORT", "27123"))


class _RedactedString(str):
    """String subclass that hides its value in repr to prevent accidental logging."""

    def __repr__(self) -> str:
        if self:
            return f"'***{self[-4:]}'" if len(self) > 4 else "'****'"
        return "''"


obsidian_api_key: str = _RedactedString(os.getenv("OBSIDIAN_API_KEY", ""))

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b")

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.getenv("DATA_DIR", os.path.join(_project_root, "data"))
chroma_path = os.getenv("CHROMA_PATH", os.path.join(data_dir, "chroma_db"))
vault_path = os.getenv("VAULT_PATH", "")

EXCLUDE_PATTERNS = ["_gsdata_", ".gsbak", ".git", "__pycache__", "node_modules", ".excalidraw.md", ".github"]
entity_aliases_file = os.getenv("ENTITY_ALIASES_FILE", "")
todo_file = os.getenv("TODO_FILE", "todos.md")

# Indexer tuning constants
skip_min_tokens = 20
chunk_size = 500
chunk_overlap = 100
embed_worker_floor = int(os.getenv("EMBED_WORKER_FLOOR", "1"))
embed_worker_ceil = int(os.getenv("EMBED_WORKER_CEIL", "2"))
read_workers = int(os.getenv("READ_WORKERS", "2"))
llm_chat_concurrency = int(os.getenv("LLM_CHAT_CONCURRENCY", "1"))
index_batch_size = int(os.getenv("INDEX_BATCH_SIZE", "50"))
llm_call_delay = float(os.getenv("LLM_CALL_DELAY", "0.5"))
llm_call_hard_timeout = int(os.getenv("LLM_CALL_HARD_TIMEOUT", "30"))
gpu_temp_limit = int(os.getenv("GPU_TEMP_LIMIT", "85"))
gpu_vram_limit = int(os.getenv("GPU_VRAM_LIMIT", "80"))
disk_temp_limit = int(os.getenv("DISK_TEMP_LIMIT", "80"))
disk_temp_check_interval = int(os.getenv("DISK_TEMP_CHECK_INTERVAL", "30"))
expand_cache_ttl = int(os.getenv("EXPAND_CACHE_TTL", "3600"))

# Ranking weights for the unified Ranker (semantic, entity, graph, keyword)
# Sum does not need to be 1.0 — the Ranker normalises them internally.
ranking_weights = {
    "semantic": float(os.getenv("RANKING_SEMANTIC", "0.40")),
    "entity": float(os.getenv("RANKING_ENTITY", "0.30")),
    "graph": float(os.getenv("RANKING_GRAPH", "0.20")),
    "keyword": float(os.getenv("RANKING_KEYWORD", "0.10")),
}


def validate(verbose: bool = True) -> list[str]:
    """Validate the configuration and environment. Returns a list of warnings (empty if all good)."""
    import requests

    warnings: list[str] = []

    # Check API key
    if not obsidian_api_key:
        warnings.append("OBSIDIAN_API_KEY is not set — Obsidian REST API calls will fail")

    # Check vault path
    if not vault_path:
        warnings.append("VAULT_PATH is not set — file-watching and mtime checks disabled")
    elif not os.path.isdir(vault_path):
        warnings.append(f"VAULT_PATH '{vault_path}' is not a valid directory")

    # Check Ollama connectivity
    try:
        resp = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if verbose:
            log.info("Ollama at %s — %d models available", ollama_base_url, len(models))

        # Check chat model
        chat_models = [m for m in models if ollama_chat_model in m]
        if not chat_models:
            suggestions = [m for m in models if "qwen" in m or "llama" in m or "mistral" in m]
            hint = f" Available chat models: {suggestions[:5]}" if suggestions else ""
            warnings.append(f"Chat model '{ollama_chat_model}' not found in Ollama.{hint}")
        elif verbose:
            matched = f" (matched: {chat_models[0]})" if chat_models[0] != ollama_chat_model else ""
            log.info("Chat model: %s%s", ollama_chat_model, matched)

        # Check embed model
        embed_models = [m for m in models if ollama_embed_model in m]
        if not embed_models:
            suggestions = [m for m in models if "embed" in m or "nomic" in m]
            hint = f" Available embed models: {suggestions[:5]}" if suggestions else ""
            warnings.append(f"Embed model '{ollama_embed_model}' not found in Ollama.{hint}")
        elif verbose:
            matched = f" (matched: {embed_models[0]})" if embed_models[0] != ollama_embed_model else ""
            log.info("Embed model: %s%s", ollama_embed_model, matched)

    except requests.exceptions.ConnectionError:
        warnings.append(f"Cannot connect to Ollama at {ollama_base_url} — is it running?")
    except Exception as e:
        warnings.append(f"Ollama check failed: {e}")

    # Check data directory
    try:
        os.makedirs(data_dir, exist_ok=True)
    except Exception as e:
        warnings.append(f"Cannot create data directory '{data_dir}': {e}")

    if warnings:
        for w in warnings:
            log.warning(w)
    elif verbose:
        log.info("All config checks passed")

    return warnings
