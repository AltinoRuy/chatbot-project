"""OpenAI language model provider."""

import os

from openai import OpenAI

from llm.providers.base_provider import BaseProvider
from llm.providers.errors import ProviderConfigurationError


class OpenAIProvider(BaseProvider):
    """Provider for generating responses through the OpenAI API."""

    def __init__(
        self,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            model: OpenAI model identifier.
            timeout: Request timeout in seconds.

        Raises:
            ProviderConfigurationError: If the API key is not configured.
        """
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY environment variable is not configured."
            )

        self.model = model
        self.client = OpenAI(timeout=timeout)

    def generate(self, prompt: str) -> str:
        """Generate a response from the OpenAI model.

        Args:
            prompt: Input prompt sent to the model.

        Returns:
            Generated text response.
        """
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text
