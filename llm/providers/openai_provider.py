from openai import OpenAI

from llm.providers.base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for generating responses through the OpenAI API."""

    def __init__(self, model: str) -> None:
        """Initializes the OpenAI provider.

        Args:
            model: OpenAI model identifier.
        """
        self.model = model
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        """Generates a response from the OpenAI model.

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
