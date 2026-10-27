"""Fallback orchestration for language model providers."""

from collections.abc import Callable

from llm.providers.base_provider import BaseProvider

ProviderFactory = Callable[[], BaseProvider]


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
            except Exception as error:
                failures.append(f"{provider_name} ({type(error).__name__})")
                continue

            if isinstance(response, str) and response.strip():
                return response

            failures.append(f"{provider_name} (empty response)")

        failure_summary = ", ".join(failures)
        raise ProviderFallbackError(
            f"All language model providers failed: {failure_summary}."
        )
