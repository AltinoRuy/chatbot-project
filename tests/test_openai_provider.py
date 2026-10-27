from unittest.mock import Mock, patch

import pytest

from llm.providers.errors import ProviderConfigurationError
from llm.providers.openai_provider import OpenAIProvider


def test_openai_provider_generates_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that OpenAIProvider returns the generated response text."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    with patch("llm.providers.openai_provider.OpenAI") as mock_openai:
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


def test_openai_provider_accepts_custom_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that OpenAIProvider accepts a custom request timeout."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    with patch("llm.providers.openai_provider.OpenAI") as mock_openai:
        OpenAIProvider(model="gpt-5-mini", timeout=30.0)

    mock_openai.assert_called_once_with(timeout=30.0)


def test_openai_provider_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a typed configuration error when the API key is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("llm.providers.openai_provider.OpenAI") as mock_openai:
        with pytest.raises(
            ProviderConfigurationError,
            match="OPENAI_API_KEY environment variable is not configured",
        ):
            OpenAIProvider(model="gpt-5-mini")

    mock_openai.assert_not_called()
