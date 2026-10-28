"""Ollama language model provider."""

import ollama

from llm.providers.base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider for generating responses through Ollama."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        timeout: float = 120.0,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            model: Ollama model identifier.
            timeout: HTTP inactivity timeout in seconds.
        """
        self.model = model
        self.client = ollama.Client(timeout=timeout)

    def generate(self, prompt: str) -> str:
        """Generate a response from the Ollama model.

        Args:
            prompt: Input prompt sent to the Ollama provider.

        Returns:
            Generated text response.
        """
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            keep_alive="30m",
        )

        return response.message.content
