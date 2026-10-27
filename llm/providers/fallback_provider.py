"""Fallback orchestration for language model providers."""

import logging
from collections.abc import Callable

import httpx
import ollama
from google.genai import errors as gemini_errors
from openai import APIError

from llm.providers.base_provider import BaseProvider
from llm.providers.errors import ProviderConfigurationError

ProviderFactory = Callable[[], BaseProvider]

logger = logging.getLogger(__name__)

OPERATIONAL_ERRORS = (
    TimeoutError,
    ConnectionError,
    httpx.HTTPError,
    ollama.ResponseError,
    APIError,
    gemini_errors.APIError,
)


class ProviderFallbackError(RuntimeError):
    """Raised when every configured language model provider fails."""


class FallbackProvider(BaseProvider):
    """Try configured providers in order until one returns a valid response."""

    def __init__(
        self,
        providers: list[tuple[str, ProviderFactory]],
    ) -> None:
        """Initialize the fallback chain.

        Args:
            providers: Ordered pairs containing a provider name and
                a callable that creates that provider.

        Raises:
            ValueError: If no providers are configured.
        """
        if not providers:
            raise ValueError("At least one language model provider must be configured.")

        self.providers = tuple(providers)

    def generate(self, prompt: str) -> str:
        """Generate a response using the first provider that succeeds.

        Args:
            prompt: Input prompt sent to each provider.

        Returns:
            The first non-empty response returned by a provider.

        Raises:
            ProviderFallbackError: If all providers fail or return
                an empty response.
        """
        failures: list[str] = []

        for provider_name, provider_factory in self.providers:
            try:
                provider = provider_factory()
                response = provider.generate(prompt)
            except OPERATIONAL_ERRORS + (ProviderConfigurationError,) as error:
                failures.append(f"{provider_name} ({type(error).__name__})")
                logger.warning(
                    "Provider %s failed with %s; trying the next provider.",
                    provider_name,
                    type(error).__name__,
                )
                continue

            if isinstance(response, str) and response.strip():
                return response

            failures.append(f"{provider_name} (empty response)")
            logger.warning(
                "Provider %s returned an empty or invalid response.",
                provider_name,
            )

        failure_summary = ", ".join(failures)
        raise ProviderFallbackError(
            f"All language model providers failed: {failure_summary}."
        )
