from llm.providers.base_provider import BaseProvider
from llm.providers.ollama_provider import OllamaProvider


def get_provider() -> BaseProvider:
    """Returns the configured language model provider.

    Returns:
        An instance of the configured language model provider.
    """
    return OllamaProvider()
