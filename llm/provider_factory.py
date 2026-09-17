from llm.providers.base_provider import BaseProvider
from llm.providers.ollama_provider import OllamaProvider


def get_provider() -> BaseProvider:
    return OllamaProvider()
