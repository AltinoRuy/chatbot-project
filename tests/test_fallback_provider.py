from unittest.mock import Mock

import pytest

from llm.providers.base_provider import BaseProvider
from llm.providers.errors import ProviderConfigurationError
from llm.providers.fallback_provider import (
    FallbackProvider,
    ProviderFallbackError,
)


def test_returns_first_successful_provider_response() -> None:
    ollama = Mock(spec=BaseProvider)
    ollama.generate.return_value = "Response from Ollama."

    ollama_factory = Mock(return_value=ollama)
    gemini_factory = Mock()

    fallback = FallbackProvider(
        [
            ("Ollama", ollama_factory),
            ("Gemini", gemini_factory),
        ]
    )

    result = fallback.generate("Hello")

    assert result == "Response from Ollama."
    ollama.generate.assert_called_once_with("Hello")
    gemini_factory.assert_not_called()


def test_tries_next_provider_when_first_factory_times_out() -> None:
    gemini = Mock(spec=BaseProvider)
    gemini.generate.return_value = "Response from Gemini."

    ollama_factory = Mock(side_effect=TimeoutError("Ollama timed out"))
    gemini_factory = Mock(return_value=gemini)

    fallback = FallbackProvider(
        [
            ("Ollama", ollama_factory),
            ("Gemini", gemini_factory),
        ]
    )

    result = fallback.generate("Hello")

    assert result == "Response from Gemini."
    ollama_factory.assert_called_once_with()
    gemini_factory.assert_called_once_with()
    gemini.generate.assert_called_once_with("Hello")


@pytest.mark.parametrize(
    "error",
    [
        TypeError("Unexpected type"),
        AttributeError("Unexpected missing attribute"),
    ],
)
def test_propagates_programming_errors_from_factory(
    error: Exception,
) -> None:
    ollama_factory = Mock(side_effect=error)
    gemini_factory = Mock()

    fallback = FallbackProvider(
        [
            ("Ollama", ollama_factory),
            ("Gemini", gemini_factory),
        ]
    )

    with pytest.raises(type(error), match=str(error)):
        fallback.generate("Hello")

    gemini_factory.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        TypeError("Unexpected type"),
        AttributeError("Unexpected missing attribute"),
    ],
)
def test_propagates_programming_errors_from_generation(
    error: Exception,
) -> None:
    ollama = Mock(spec=BaseProvider)
    ollama.generate.side_effect = error
    gemini_factory = Mock()

    fallback = FallbackProvider(
        [
            ("Ollama", Mock(return_value=ollama)),
            ("Gemini", gemini_factory),
        ]
    )

    with pytest.raises(type(error), match=str(error)):
        fallback.generate("Hello")

    gemini_factory.assert_not_called()


def test_tries_next_provider_when_generation_times_out() -> None:
    ollama = Mock(spec=BaseProvider)
    ollama.generate.side_effect = TimeoutError("Request timed out")

    gemini = Mock(spec=BaseProvider)
    gemini.generate.return_value = "Response from Gemini."

    fallback = FallbackProvider(
        [
            ("Ollama", Mock(return_value=ollama)),
            ("Gemini", Mock(return_value=gemini)),
        ]
    )

    result = fallback.generate("Hello")

    assert result == "Response from Gemini."
    ollama.generate.assert_called_once_with("Hello")
    gemini.generate.assert_called_once_with("Hello")


def test_tries_next_provider_when_response_is_empty() -> None:
    ollama = Mock(spec=BaseProvider)
    ollama.generate.return_value = "   "

    gemini = Mock(spec=BaseProvider)
    gemini.generate.return_value = "Response from Gemini."

    fallback = FallbackProvider(
        [
            ("Ollama", Mock(return_value=ollama)),
            ("Gemini", Mock(return_value=gemini)),
        ]
    )

    result = fallback.generate("Hello")

    assert result == "Response from Gemini."
    ollama.generate.assert_called_once_with("Hello")
    gemini.generate.assert_called_once_with("Hello")


def test_raises_error_when_all_providers_fail() -> None:
    ollama = Mock(spec=BaseProvider)
    ollama.generate.side_effect = TimeoutError("Request timed out")

    gemini = Mock(spec=BaseProvider)
    gemini.generate.return_value = ""

    fallback = FallbackProvider(
        [
            ("Ollama", Mock(return_value=ollama)),
            ("Gemini", Mock(return_value=gemini)),
        ]
    )

    with pytest.raises(
        ProviderFallbackError,
        match="All language model providers failed",
    ) as error:
        fallback.generate("Hello")

    assert "Ollama (TimeoutError)" in str(error.value)
    assert "Gemini (empty response)" in str(error.value)


def test_rejects_empty_provider_list() -> None:
    with pytest.raises(
        ValueError,
        match="At least one language model provider",
    ):
        FallbackProvider([])


def test_returns_response_without_modifying_its_content() -> None:
    ollama = Mock(spec=BaseProvider)
    ollama.generate.return_value = "  Original response.  "

    fallback = FallbackProvider([("Ollama", Mock(return_value=ollama))])

    result = fallback.generate("Hello")

    assert result == "  Original response.  "


def test_tries_next_provider_when_factory_has_configuration_error() -> None:
    """Try the next provider when a configured provider lacks credentials."""
    ollama_factory = Mock(
        side_effect=ProviderConfigurationError("Ollama credentials are missing")
    )

    gemini = Mock(spec=BaseProvider)
    gemini.generate.return_value = "Response from Gemini."
    gemini_factory = Mock(return_value=gemini)

    fallback = FallbackProvider(
        [
            ("Ollama", ollama_factory),
            ("Gemini", gemini_factory),
        ]
    )

    result = fallback.generate("Hello")

    assert result == "Response from Gemini."
    ollama_factory.assert_called_once_with()
    gemini_factory.assert_called_once_with()
    gemini.generate.assert_called_once_with("Hello")
