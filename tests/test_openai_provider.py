from unittest.mock import Mock, patch

from llm.providers.openai_provider import OpenAIProvider


@patch("llm.providers.openai_provider.OpenAI")
def test_openai_provider_generates_response(mock_openai: Mock) -> None:
    """Test that OpenAIProvider returns the generated response text."""
    mock_client = Mock()
    mock_response = Mock()

    mock_response.output_text = "Hello from OpenAI."
    mock_client.responses.create.return_value = mock_response
    mock_openai.return_value = mock_client

    provider = OpenAIProvider(model="gpt-5-mini")

    result = provider.generate("Say hello.")

    assert result == "Hello from OpenAI."

    mock_openai.assert_called_once_with(timeout=120.0)
    mock_client.responses.create.assert_called_once_with(
        model="gpt-5-mini",
        input="Say hello.",
    )


@patch("llm.providers.openai_provider.OpenAI")
def test_openai_provider_accepts_custom_timeout(
    mock_openai: Mock,
) -> None:
    """Test that OpenAIProvider accepts a custom request timeout."""
    OpenAIProvider(model="gpt-5-mini", timeout=30.0)

    mock_openai.assert_called_once_with(timeout=30.0)
