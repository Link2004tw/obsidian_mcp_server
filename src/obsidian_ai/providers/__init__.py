from ..logger import get_logger
from .base import BaseLLMProvider
from .ollama import OllamaProvider
from .openai_provider import OpenAIProvider

PROVIDER_MAP: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}

_INSTANCES: dict[str, BaseLLMProvider] = {}


def get_provider(name: str) -> BaseLLMProvider:
    if name in _INSTANCES:
        return _INSTANCES[name]

    cls = PROVIDER_MAP.get(name)
    if cls is None:
        available = ", ".join(PROVIDER_MAP)
        get_logger(__name__).warning(
            f"Unknown provider '{name}', falling back to 'ollama'. "
            f"Available: {available}"
        )
        cls = OllamaProvider

    instance = cls()
    _INSTANCES[name] = instance
    return instance
