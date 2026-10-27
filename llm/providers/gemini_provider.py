"""Google Gemini language model provider."""

import os

from google import genai

from llm.providers.base_provider import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider for generating responses through the Gemini API."""

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        timeout: float = 120.0,
    ) -> None:
        """Initialize the Gemini provider.

        Args:
            model: Gemini model identifier.
            timeout: Request timeout in seconds.

        Raises:
            RuntimeError: If the Gemini API key is not configured.
        """
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")

        self.model = model
        self.timeout = timeout
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """Generate a response through the Gemini API.

        Args:
            prompt: Input prompt sent to the Gemini provider.

        Returns:
            Generated text response, or an empty string if no text is returned.
        """
        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
            timeout=self.timeout,
        )

        return interaction.output_text or ""
