import os
from pathlib import Path

from dotenv import load_dotenv

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
ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.getenv("DATA_DIR", os.path.join(_project_root, "data"))
chroma_path = os.getenv("CHROMA_PATH", os.path.join(data_dir, "chroma_db"))
vault_path = os.getenv("VAULT_PATH", "")

EXCLUDE_PATTERNS = ["_gsdata_", ".gsbak", ".git", "__pycache__", "node_modules", ".excalidraw.md"]
todo_file = os.getenv("TODO_FILE", "todos.md")

# Indexer tuning constants
skip_min_tokens = 20
chunk_size = 500
chunk_overlap = 100
embed_worker_floor = 2
embed_worker_ceil = 6
