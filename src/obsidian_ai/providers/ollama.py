import time

import requests

from .. import config
from .base import BaseLLMProvider

REQUEST_TIMEOUT = int(__import__("os").getenv("OLLAMA_CHAT_TIMEOUT", "45"))
EMBED_TIMEOUT = 60
MAX_RETRIES = 2
INITIAL_BACKOFF = 2

RETRYABLE_STATUSES = {429, 502, 503}


class OllamaProvider(BaseLLMProvider):
    _AVAILABLE: bool | None = None
    _AVAILABLE_LAST_CHECK: float = 0
    _AVAILABLE_CACHE_SECONDS = 30

    @property
    def name(self) -> str:
        return "ollama"

    def _request_with_retry(self, method, url, *, timeout, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.request(method, url, timeout=timeout, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.exceptions.ReadTimeout:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = INITIAL_BACKOFF * (2 ** attempt)
                time.sleep(wait)
            except requests.exceptions.ConnectionError:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = INITIAL_BACKOFF * (2 ** attempt)
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in RETRYABLE_STATUSES and attempt < MAX_RETRIES - 1:
                    wait = INITIAL_BACKOFF * (2 ** attempt)
                    time.sleep(wait)
                else:
                    raise

    def is_available(self) -> bool:
        now = time.time()
        if self._AVAILABLE is not None and (now - self._AVAILABLE_LAST_CHECK) < self._AVAILABLE_CACHE_SECONDS:
            return self._AVAILABLE
        try:
            resp = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            has_chat = any(config.ollama_chat_model in m for m in models)
            has_embed = any(config.ollama_embed_model in m for m in models)
            self._AVAILABLE = bool(models)
            self._AVAILABLE_LAST_CHECK = now
            if not has_chat:
                from ..logger import get_logger
                get_logger(__name__).warning(f"Chat model '{config.ollama_chat_model}' not found in Ollama")
            if not has_embed:
                from ..logger import get_logger
                get_logger(__name__).warning(f"Embed model '{config.ollama_embed_model}' not found in Ollama")
            return self._AVAILABLE
        except Exception:
            self._AVAILABLE = False
            self._AVAILABLE_LAST_CHECK = now
            return False

    def check_health(self) -> dict:
        ollama_ok = self.is_available()
        status = {
            "ollama": {
                "available": ollama_ok,
                "url": config.ollama_base_url,
                "chat_model": config.ollama_chat_model,
                "embed_model": config.ollama_embed_model,
            },
            "overall": "healthy" if ollama_ok else "degraded",
        }
        if ollama_ok and isinstance(ollama_info := status.get("ollama"), dict):
            try:
                resp = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
                models: list[str] = [m["name"] for m in resp.json().get("models", [])]
                ollama_info["models"] = models
            except Exception:
                ollama_info["models"] = []
        return status

    def embed(self, text: str) -> list[float]:
        resp = self._request_with_retry(
            "POST",
            f"{config.ollama_base_url}/api/embeddings",
            json={"model": config.ollama_embed_model, "prompt": text},
            timeout=EMBED_TIMEOUT,
        )
        data = resp.json()
        embedding = data["embedding"]
        assert isinstance(embedding, list)
        return embedding

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = self._request_with_retry(
                "POST",
                f"{config.ollama_base_url}/api/embed",
                json={"model": config.ollama_embed_model, "input": texts},
                timeout=EMBED_TIMEOUT,
            )
            data = resp.json()
            embeddings: list[list[float]] = data["embeddings"]
            return embeddings
        except Exception:
            return [self.embed(t) for t in texts]

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        model = model or config.ollama_chat_model
        msgs = list(messages)
        payload = {
            "model": model,
            "messages": msgs,
            "stream": False,
            "keep_alive": "5m",
        }
        resp = self._request_with_retry(
            "POST",
            f"{config.ollama_base_url}/api/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        data = resp.json()
        content = data["message"]["content"]
        assert isinstance(content, str)
        return content

    def chat_safe(self, messages: list[dict], model: str | None = None) -> str | None:
        if not self.is_available():
            return None
        try:
            return self.chat(messages, model=model)
        except Exception:
            return None

    def switch_embed_model(self, model_name: str) -> None:
        config.ollama_embed_model = model_name

    def clear_embed_cache(self) -> None:
        pass

    def embed_cache_info(self) -> dict:
        return {}
