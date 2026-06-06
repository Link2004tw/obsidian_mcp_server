import time

from .. import config
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    _AVAILABLE: bool | None = None
    _AVAILABLE_LAST_CHECK: float = 0
    _AVAILABLE_CACHE_SECONDS = 30

    def __init__(self) -> None:
        self._client = None
        self._embed_model = config.openai_embed_model
        self._chat_model = config.openai_chat_model
        self._api_key = config.openai_api_key
        self._base_url = config.openai_base_url

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install obsidian-ai[openai]"
            ) from err
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        return self._client

    def is_available(self) -> bool:
        now = time.time()
        if self._AVAILABLE is not None and (now - self._AVAILABLE_LAST_CHECK) < self._AVAILABLE_CACHE_SECONDS:
            return self._AVAILABLE
        try:
            if not self._api_key and not self._base_url.startswith("http://localhost"):
                self._AVAILABLE = False
                self._AVAILABLE_LAST_CHECK = now
                return False
            client = self._get_client()
            client.models.list()
            self._AVAILABLE = True
            self._AVAILABLE_LAST_CHECK = now
            return True
        except Exception:
            self._AVAILABLE = False
            self._AVAILABLE_LAST_CHECK = now
            return False

    def check_health(self) -> dict:
        ok = self.is_available()
        base_url = self._base_url
        return {
            "openai": {
                "available": ok,
                "url": base_url,
                "chat_model": self._chat_model,
                "embed_model": self._embed_model,
            },
            "overall": "healthy" if ok else "degraded",
        }

    def embed(self, text: str) -> list[float]:
        client = self._get_client()
        resp = client.embeddings.create(
            input=text,
            model=self._embed_model,
        )
        result: list[float] = resp.data[0].embedding
        return result

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        resp = client.embeddings.create(
            input=texts,
            model=self._embed_model,
        )
        indexed = {i: e for i, e in enumerate(resp.data)}
        return [indexed[i].embedding for i in sorted(indexed)]

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        client = self._get_client()
        model = model or self._chat_model
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = resp.choices[0].message.content
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
        self._embed_model = model_name

    def clear_embed_cache(self) -> None:
        pass

    def embed_cache_info(self) -> dict:
        return {}
