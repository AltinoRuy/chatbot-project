from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Defines the interface for language model providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a response from the language model.

        Args:
            prompt: Input prompt sent to the language model.

        Returns:
            Generated text response.
        """
        pass
