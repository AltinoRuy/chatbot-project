from unittest.mock import Mock, patch

import pytest

from llm.providers.errors import ProviderConfigurationError
from llm.providers.gemini_provider import GeminiProvider


def test_provider_initializes_with_default_model() -> None:
    with (
        patch(
            "llm.providers.gemini_provider.os.getenv",
            return_value="test-api-key",
        ),
        patch(
            "llm.providers.gemini_provider.genai.Client",
        ) as mock_client_class,
    ):
        provider = GeminiProvider()

    assert provider.model == "gemini-3.8-flash"
    assert provider.timeout == 120.0
    mock_client_class.assert_called_once_with(
        api_key="test-api-key",
    )


def test_provider_accepts_custom_model_and_timeout() -> None:
    with (
        patch(
            "llm.providers.gemini_provider.os.getenv",
            return_value="test-api-key",
        ),
        patch(
            "llm.providers.gemini_provider.genai.Client",
        ),
    ):
        provider = GeminiProvider(
            model="custom-model",
            timeout=30.0,
        )

    assert provider.model == "custom-model"
    assert provider.timeout == 30.0


def test_generate_returns_gemini_response() -> None:
    with (
        patch(
            "llm.providers.gemini_provider.os.getenv",
            return_value="test-api-key",
        ),
        patch(
            "llm.providers.gemini_provider.genai.Client",
        ) as mock_client_class,
    ):
        mock_client = mock_client_class.return_value
        mock_response = Mock()
        mock_response.output_text = "Hello from Gemini."

        mock_client.interactions.create.return_value = mock_response

        provider = GeminiProvider()
        result = provider.generate("Say hello.")

    assert result == "Hello from Gemini."
    mock_client.interactions.create.assert_called_once_with(
        model="gemini-3.8-flash",
        input="Say hello.",
        timeout=120.0,
    )


def test_generate_returns_empty_string_when_output_is_empty() -> None:
    with (
        patch(
            "llm.providers.gemini_provider.os.getenv",
            return_value="test-api-key",
        ),
        patch(
            "llm.providers.gemini_provider.genai.Client",
        ) as mock_client_class,
    ):
        mock_client = mock_client_class.return_value
        mock_response = Mock()
        mock_response.output_text = None
        mock_client.interactions.create.return_value = mock_response

        provider = GeminiProvider()
        result = provider.generate("Say hello.")

    assert result == ""


def test_provider_requires_api_key() -> None:
    with (
        patch(
            "llm.providers.gemini_provider.os.getenv",
            return_value=None,
        ),
        patch(
            "llm.providers.gemini_provider.genai.Client",
        ) as mock_client_class,
    ):
        with pytest.raises(
            ProviderConfigurationError,
            match="GEMINI_API_KEY environment variable is not configured",
        ):
            GeminiProvider()

    mock_client_class.assert_not_called()
