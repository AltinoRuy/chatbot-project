from llm.provider_factory import get_provider
from llm.providers.ollama_provider import OllamaProvider


def test_get_provider_returns_ollama_provider() -> None:
    provider = get_provider()

    assert isinstance(provider, OllamaProvider)
