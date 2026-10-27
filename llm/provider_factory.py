"""Factory for configuring language model providers."""

import os
from collections.abc import Callable

from llm.providers.base_provider import BaseProvider
from llm.providers.fallback_provider import FallbackProvider

ProviderFactory = Callable[[], BaseProvider]


def _create_ollama_provider() -> BaseProvider:
    """Create the local Ollama provider on demand."""
    from llm.providers.ollama_provider import OllamaProvider

    return OllamaProvider(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    )


def _create_gemini_provider() -> BaseProvider:
    """Create the Gemini provider on demand."""
    from llm.providers.gemini_provider import GeminiProvider

    return GeminiProvider(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    )


def _create_openai_provider() -> BaseProvider:
    """Create the OpenAI provider on demand."""
    from llm.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    )


def get_provider() -> BaseProvider:
    """Return a configurable provider fallback chain.

    The CHATBOT_PROVIDERS environment variable defines the provider
    order, separated by commas. If it is not set, the default order is
    Ollama, Gemini, then OpenAI.

    Returns:
        A fallback provider containing the selected providers.

    Raises:
        ValueError: If the configuration is empty or contains an
            unsupported or duplicated provider.
    """
    configured = os.getenv(
        "CHATBOT_PROVIDERS",
        "ollama,gemini,openai",
    )

    provider_names = [
        name.strip().lower() for name in configured.split(",") if name.strip()
    ]

    if not provider_names:
        raise ValueError("CHATBOT_PROVIDERS must contain at least one provider.")

    if len(provider_names) != len(set(provider_names)):
        raise ValueError("CHATBOT_PROVIDERS cannot contain duplicate providers.")

    factories: dict[str, ProviderFactory] = {
        "ollama": _create_ollama_provider,
        "gemini": _create_gemini_provider,
        "openai": _create_openai_provider,
    }

    unsupported = [name for name in provider_names if name not in factories]

    if unsupported:
        raise ValueError(
            f"Unsupported provider(s): {', '.join(unsupported)}. "
            f"Supported providers: {', '.join(factories)}."
        )

    return FallbackProvider(
        [(name.title(), factories[name]) for name in provider_names]
    )
