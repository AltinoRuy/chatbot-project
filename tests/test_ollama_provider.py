from unittest.mock import Mock, patch

from llm.providers.ollama_provider import OllamaProvider


@patch("llm.providers.ollama_provider.ollama.Client")
def test_provider_initializes_with_default_model(
    mock_client_class: Mock,
) -> None:
    provider = OllamaProvider()

    assert provider.model == "llama3.2:3b"
    mock_client_class.assert_called_once_with(timeout=120.0)


@patch("llm.providers.ollama_provider.ollama.Client")
def test_provider_initializes_with_custom_model_and_timeout(
    mock_client_class: Mock,
) -> None:
    provider = OllamaProvider(model="custom-model", timeout=30.0)

    assert provider.model == "custom-model"
    mock_client_class.assert_called_once_with(timeout=30.0)


@patch("llm.providers.ollama_provider.ollama.Client")
def test_generate_returns_ollama_response(
    mock_client_class: Mock,
) -> None:
    response = Mock()
    response.message.content = "Hello from Ollama."

    mock_client = mock_client_class.return_value
    mock_client.chat.return_value = response

    provider = OllamaProvider()
    result = provider.generate("Hello")

    assert result == "Hello from Ollama."
    mock_client.chat.assert_called_once_with(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        keep_alive="30m",
    )
