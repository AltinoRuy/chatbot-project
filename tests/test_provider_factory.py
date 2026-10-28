from unittest.mock import patch

import pytest

from llm.provider_factory import (
    _create_gemini_provider,
    _create_ollama_provider,
    _create_openai_provider,
    get_provider,
)
from llm.providers.fallback_provider import (
    FallbackProvider,
    ProviderFallbackError,
)


def test_get_provider_uses_local_provider_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHATBOT_PROVIDERS", raising=False)

    provider = get_provider()

    assert isinstance(provider, FallbackProvider)
    assert [name.casefold() for name, _ in provider.providers] == ["ollama"]


def test_default_configuration_never_falls_back_to_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHATBOT_PROVIDERS", raising=False)

    with (
        patch("llm.provider_factory._create_ollama_provider") as ollama_factory,
        patch("llm.provider_factory._create_gemini_provider") as gemini_factory,
        patch("llm.provider_factory._create_openai_provider") as openai_factory,
    ):
        ollama_factory.return_value.generate.side_effect = TimeoutError(
            "Local provider timed out"
        )

        provider = get_provider()

        with pytest.raises(ProviderFallbackError):
            provider.generate("Test prompt")

    ollama_factory.assert_called_once_with()
    gemini_factory.assert_not_called()
    openai_factory.assert_not_called()


def test_get_provider_supports_cloud_only_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDERS", "gemini,openai")

    provider = get_provider()

    assert isinstance(provider, FallbackProvider)
    assert [name.casefold() for name, _ in provider.providers] == [
        "gemini",
        "openai",
    ]


def test_get_provider_preserves_custom_provider_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDERS", "openai,gemini")

    provider = get_provider()

    assert [name.casefold() for name, _ in provider.providers] == [
        "openai",
        "gemini",
    ]


def test_get_provider_supports_single_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDERS", "ollama")

    provider = get_provider()

    assert [name.casefold() for name, _ in provider.providers] == ["ollama"]


def test_get_provider_rejects_empty_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDERS", " , ")

    with pytest.raises(
        ValueError,
        match="must contain at least one provider",
    ):
        get_provider()


def test_get_provider_rejects_duplicate_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDERS", "gemini,gemini")

    with pytest.raises(
        ValueError,
        match="cannot contain duplicate providers",
    ):
        get_provider()


def test_get_provider_rejects_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDERS", "gemini,claude")

    with pytest.raises(
        ValueError,
        match="Unsupported provider",
    ):
        get_provider()


@pytest.mark.parametrize(
    ("factory", "module", "class_name", "env_var", "default_model"),
    [
        (
            _create_ollama_provider,
            "llm.providers.ollama_provider",
            "OllamaProvider",
            "OLLAMA_MODEL",
            "llama3.2:3b",
        ),
        (
            _create_gemini_provider,
            "llm.providers.gemini_provider",
            "GeminiProvider",
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        ),
        (
            _create_openai_provider,
            "llm.providers.openai_provider",
            "OpenAIProvider",
            "OPENAI_MODEL",
            "gpt-5.6-terra",
        ),
    ],
)
def test_provider_factory_uses_default_model(
    monkeypatch: pytest.MonkeyPatch,
    factory: object,
    module: str,
    class_name: str,
    env_var: str,
    default_model: str,
) -> None:
    monkeypatch.delenv(env_var, raising=False)

    with patch(f"{module}.{class_name}") as mock_provider:
        provider_factory = factory
        provider_factory()

    mock_provider.assert_called_once_with(model=default_model)


@pytest.mark.parametrize(
    ("factory", "module", "class_name", "env_var"),
    [
        (
            _create_ollama_provider,
            "llm.providers.ollama_provider",
            "OllamaProvider",
            "OLLAMA_MODEL",
        ),
        (
            _create_gemini_provider,
            "llm.providers.gemini_provider",
            "GeminiProvider",
            "GEMINI_MODEL",
        ),
        (
            _create_openai_provider,
            "llm.providers.openai_provider",
            "OpenAIProvider",
            "OPENAI_MODEL",
        ),
    ],
)
def test_provider_factory_uses_custom_model(
    monkeypatch: pytest.MonkeyPatch,
    factory: object,
    module: str,
    class_name: str,
    env_var: str,
) -> None:
    monkeypatch.setenv(env_var, "custom-test-model")

    with patch(f"{module}.{class_name}") as mock_provider:
        provider_factory = factory
        provider_factory()

    mock_provider.assert_called_once_with(model="custom-test-model")


def test_get_provider_falls_back_when_gemini_api_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDERS", "gemini,ollama")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with patch("llm.provider_factory._create_ollama_provider") as ollama_factory:
        ollama_factory.return_value.generate.return_value = "Ollama response."

        provider = get_provider()
        response = provider.generate("Hello")

    assert response == "Ollama response."
    ollama_factory.assert_called_once_with()
    ollama_factory.return_value.generate.assert_called_once_with("Hello")


def test_get_provider_falls_back_when_openai_api_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDERS", "openai,ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("llm.provider_factory._create_ollama_provider") as ollama_factory:
        ollama_factory.return_value.generate.return_value = "Ollama response."

        provider = get_provider()
        response = provider.generate("Hello")

    assert response == "Ollama response."
    ollama_factory.assert_called_once_with()
    ollama_factory.return_value.generate.assert_called_once_with("Hello")


@pytest.mark.parametrize(
    ("configuration", "expected_message"),
    [
        (" , ", "must contain at least one provider"),
        ("gemini,gemini", "cannot contain duplicate providers"),
        ("gemini,claude", "Unsupported provider"),
    ],
)
def test_invalid_global_configuration_fails_before_provider_initialization(
    monkeypatch: pytest.MonkeyPatch,
    configuration: str,
    expected_message: str,
) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDERS", configuration)

    with (
        patch("llm.provider_factory._create_ollama_provider") as ollama_factory,
        patch("llm.provider_factory._create_gemini_provider") as gemini_factory,
        patch("llm.provider_factory._create_openai_provider") as openai_factory,
    ):
        with pytest.raises(ValueError, match=expected_message):
            get_provider()

    ollama_factory.assert_not_called()
    gemini_factory.assert_not_called()
    openai_factory.assert_not_called()
