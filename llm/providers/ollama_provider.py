import ollama

from llm.providers.base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    def __init__(self, model: str = "llama3.2:3b") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
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
