import os
from dotenv import load_dotenv

load_dotenv()

obsidian_host = os.getenv("OBSIDIAN_HOST", "localhost")
obsidian_port = int(os.getenv("OBSIDIAN_PORT", "27123"))
obsidian_api_key = os.getenv("OBSIDIAN_API_KEY", "")

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
chroma_path = os.getenv("CHROMA_PATH", os.path.join(_project_root, "data", "chroma_db"))

EXCLUDE_PATTERNS = ["_gsdata_", ".gsbak", ".git", "__pycache__", "node_modules", ".excalidraw.md"]
