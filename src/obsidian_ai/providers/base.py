from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def chat(self, messages: list[dict], model: str | None = None) -> str:
        ...

    @abstractmethod
    def chat_safe(self, messages: list[dict], model: str | None = None) -> str | None:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def check_health(self) -> dict:
        ...

    @abstractmethod
    def switch_embed_model(self, model_name: str) -> None:
        ...

    @abstractmethod
    def clear_embed_cache(self) -> None:
        ...

    @abstractmethod
    def embed_cache_info(self) -> dict:
        ...
