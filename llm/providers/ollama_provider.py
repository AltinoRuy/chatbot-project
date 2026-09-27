import ollama

from llm.providers.base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider for generating responses through Ollama."""

    def __init__(self, model: str = "llama3.2:3b") -> None:
        """Initializes the Ollama provider.

        Args:
            model: Ollama model identifier.
        """
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generates a response from the Ollama model.

        Args:
            prompt: Input prompt sent to the Ollama model.

        Returns:
            Generated text response.
        """
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.message.content
