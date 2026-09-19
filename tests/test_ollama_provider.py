from unittest.mock import Mock, patch

from llm.providers.ollama_provider import OllamaProvider


def test_provider_initializes_with_default_model() -> None:
    provider = OllamaProvider()

    assert provider.model == "llama3.2:3b"


def test_generate_returns_ollama_response() -> None:
    provider = OllamaProvider()

    response = Mock()
    response.message.content = "Hello from Ollama."

    with patch(
        "llm.providers.ollama_provider.ollama.chat",
        return_value=response,
    ) as mock_chat:
        result = provider.generate("Hello")

    assert result == "Hello from Ollama."
    mock_chat.assert_called_once_with(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
    )
