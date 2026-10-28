from unittest.mock import Mock, patch

import pytest

from agents.writer_agent import _detect_language, fallback_response, write_response
from llm.providers.fallback_provider import ProviderFallbackError


def test_write_response_returns_provider_response() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_writer_sends_authoritative_result_to_provider() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        write_response("5 + 4", 9.0)

    provider.generate.assert_called_once()
    prompt = provider.generate.call_args.args[0]

    assert "The authoritative result is 9." in prompt


def test_writer_prompt_forbids_calculations() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        write_response("5 + 4", 9.0)

    prompt = provider.generate.call_args.args[0]

    assert "Do not calculate anything." in prompt
    assert "Do not introduce any additional numbers." in prompt


def test_writer_prompt_protects_authoritative_result() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        write_response("5 + 4", 9.0)

    prompt = provider.generate.call_args.args[0]

    assert "The authoritative result is provided by the application." in prompt
    assert "Do not calculate anything." in prompt
    assert (
        "Do not interpret, reinterpret, or modify the mathematical operation." in prompt
    )
    assert "The sentence MUST contain the exact placeholder {{RESULT}}." in prompt


def test_write_response_formats_integer_result() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_formats_decimal_result() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("10 / 2", 5.0)

    assert result == "The result is 5."


def test_write_response_uses_fallback_when_result_placeholder_is_missing() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is 9."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_uses_fallback_when_response_has_extra_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}, or 10."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_handles_known_error_deterministically() -> None:
    provider = Mock()
    provider.generate.return_value = (
        "It is not possible to complete the operation: {{ERROR}}."
    )

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "10 / 0",
            error="Cannot divide by zero",
        )

    assert result == "It is not possible to divide by zero."
    provider.generate.assert_not_called()


def test_writer_does_not_send_known_error_to_provider() -> None:
    provider = Mock()

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("10 / 0", error="Cannot divide by zero")

    assert result == "It is not possible to divide by zero."
    provider.generate.assert_not_called()


def test_write_response_uses_deterministic_response_for_known_error() -> None:
    provider = Mock()
    provider.generate.return_value = "The answer is 10."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "10 / 0",
            error="Cannot divide by zero",
        )

    assert result == "It is not possible to divide by zero."
    provider.generate.assert_not_called()


def test_write_response_handles_context_requirement() -> None:
    provider = Mock()
    provider.generate.return_value = "There is no previous result."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("divide 2", needs_context=True)

    assert result == "There is no previous result."


def test_write_response_uses_fallback_for_blank_context_response() -> None:
    provider = Mock()
    provider.generate.return_value = "   "

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("divide 2", needs_context=True)

    assert result == "There is no previous mathematical result to use."


def test_write_response_uses_fallback_for_context_response_with_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "Your previous result was 10."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("divide 2", needs_context=True)

    assert result == "There is no previous mathematical result to use."


def test_write_response_handles_non_math_response() -> None:
    provider = Mock()
    provider.generate.return_value = "Hello! How can I help you?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Hello")

    assert result == "Hello! How can I help you?"


def test_write_response_uses_fallback_for_non_math_response_with_new_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "The answer is 42."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Hello")

    assert result == "I can help with that, but I will not perform calculations."


def test_write_response_handles_ambiguous_response() -> None:
    provider = Mock()
    provider.generate.return_value = "Could you clarify what you want to calculate?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="ambiguous",
        )

    assert result == "Could you clarify what you want to calculate?"


def test_write_response_uses_fallback_for_ambiguous_response_with_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "Do you mean 5 or 10?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="ambiguous",
        )

    assert result == "Could you clarify what you want to calculate?"


def test_write_response_handles_interpretation_failure() -> None:
    provider = Mock()
    provider.generate.return_value = (
        "I could not confidently interpret the mathematical expression."
    )

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="interpretation_failed",
        )

    assert result == "I could not confidently interpret the mathematical expression."


def test_interpretation_failure_falls_back_when_response_has_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "Try calculating 10 / 2."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="interpretation_failed",
        )

    assert result == "Could you rephrase your mathematical request?"


def test_write_response_uses_portuguese_for_portuguese_message() -> None:
    provider = Mock()
    provider.generate.return_value = "O resultado é {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("quanto é 2 + 2", 4.0)

    assert result == "O resultado é 4."


def test_write_response_uses_english_for_english_message() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("what is 2 + 2", 4.0)

    assert result == "The result is 4."


def test_write_response_rejects_digits_embedded_in_words() -> None:
    provider = Mock()
    provider.generate.return_value = "The answer is abc123 {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_uses_fallback_for_whitespace_non_math_response() -> None:
    provider = Mock()
    provider.generate.return_value = "   "

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Hello")

    assert result == "I can help with that, but I will not perform calculations."


def test_write_response_detects_portuguese_greeting() -> None:
    provider = Mock()
    provider.generate.return_value = "Olá! Como posso ajudar?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Bom dia, tudo bem?")

    assert result == "Olá! Como posso ajudar?"


@pytest.mark.parametrize("provider_response", [None, 42, ["unexpected"]])
def test_write_response_uses_fallback_for_non_string_result_response(
    provider_response: object,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


@pytest.mark.parametrize("provider_response", [None, 42, ["unexpected"]])
def test_write_response_uses_fallback_for_non_string_generic_error_response(
    provider_response: object,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate an expression",
            error="Invalid expression",
        )

    assert (
        result == "It was not possible to complete the operation: Invalid expression."
    )
    provider.generate.assert_called_once()


@pytest.mark.parametrize("provider_response", [None, 42, ["unexpected"]])
def test_write_response_uses_fallback_for_non_string_context_response(
    provider_response: object,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("divide 2", needs_context=True)

    assert result == "There is no previous mathematical result to use."


@pytest.mark.parametrize("provider_response", [None, 42, ["unexpected"]])
def test_write_response_uses_fallback_for_non_string_ambiguous_response(
    provider_response: object,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="ambiguous",
        )

    assert result == "Could you clarify what you want to calculate?"


def test_write_response_uses_portuguese_result_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Resultado inválido 123."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("quanto é 5 + 4", 9.0)

    assert result == "O resultado é 9."


def test_write_response_uses_portuguese_generic_error_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Resposta inválida 123."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calcular uma expressão",
            error="Invalid expression",
        )

    assert result == ("Não foi possível concluir a operação: Invalid expression.")


def test_write_response_uses_english_generic_error_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Invalid response 123."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate an expression",
            error="Invalid expression",
        )

    assert (
        result == "It was not possible to complete the operation: Invalid expression."
    )


def test_write_response_uses_portuguese_ambiguous_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Você quer 2 ou 3?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "poderia calcular isso",
            response_type="ambiguous",
        )

    assert result == "Poderia esclarecer o que deseja calcular?"


def test_write_response_uses_portuguese_interpretation_failure_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Tente calcular 10 / 2."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "poderia calcular isso",
            response_type="interpretation_failed",
        )

    assert result == "Poderia reformular seu pedido matemático?"


def test_write_response_uses_portuguese_non_math_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "A resposta é 42."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Olá, tudo bem?")

    assert result == "Posso ajudar com isso, mas não vou realizar cálculos."


def test_writer_uses_portuguese_when_language_markers_are_tied() -> None:
    provider = Mock()
    provider.generate.return_value = "Olá!"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("hello olá")

    assert result == "Olá!"
    prompt = provider.generate.call_args.args[0]
    assert "Respond only in Portuguese." in prompt


@pytest.mark.parametrize(
    (
        "message",
        "kwargs",
        "expected",
    ),
    [
        (
            "5 + 4",
            {"result": 9.0},
            "The result is 9.",
        ),
        (
            "10 / 0",
            {"error": "Cannot divide by zero"},
            "It is not possible to divide by zero.",
        ),
        (
            "divide 2",
            {"needs_context": True},
            "There is no previous mathematical result to use.",
        ),
        (
            "calculate something",
            {"response_type": "ambiguous"},
            "Could you clarify what you want to calculate?",
        ),
        (
            "calculate something",
            {"response_type": "interpretation_failed"},
            "Could you rephrase your mathematical request?",
        ),
        (
            "Hello",
            {},
            "I can help with that, but I will not perform calculations.",
        ),
    ],
)
def test_write_response_uses_fallback_when_provider_raises_exception(
    message: str,
    kwargs: dict[str, object],
    expected: str,
) -> None:
    provider = Mock()
    provider.generate.side_effect = ProviderFallbackError("Provider unavailable")

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(message, **kwargs)

    assert result == expected

    if kwargs.get("error") == "Cannot divide by zero":
        provider.generate.assert_not_called()
    else:
        provider.generate.assert_called_once()


@pytest.mark.parametrize(
    ("message", "kwargs", "expected"),
    [
        (
            "5 + 4",
            {"result": 9.0},
            "The result is 9.",
        ),
        (
            "divide 2",
            {"needs_context": True},
            "There is no previous mathematical result to use.",
        ),
        (
            "calculate something",
            {"response_type": "ambiguous"},
            "Could you clarify what you want to calculate?",
        ),
        (
            "calculate something",
            {"response_type": "interpretation_failed"},
            "Could you rephrase your mathematical request?",
        ),
        (
            "Hello",
            {},
            "I can help with that, but I will not perform calculations.",
        ),
        (
            "calculate an expression",
            {"error": "Invalid expression"},
            "It was not possible to complete the operation: Invalid expression.",
        ),
    ],
)
def test_write_response_uses_fallback_when_provider_initialization_fails(
    message: str,
    kwargs: dict[str, object],
    expected: str,
) -> None:
    with patch(
        "agents.writer_agent.get_provider",
        side_effect=ProviderFallbackError("Provider unavailable"),
    ):
        result = write_response(message, **kwargs)

    assert result == expected


@pytest.mark.parametrize(
    ("message", "provider_response", "expected"),
    [
        (
            "5 + 4",
            "The result is nine, not {{RESULT}}.",
            "The result is 9.",
        ),
        (
            "5 + 4",
            "The result is {{RESULT}}, or ten.",
            "The result is 9.",
        ),
        (
            "5 + 4",
            "The result is {{RESULT}}, or twenty-five.",
            "The result is 9.",
        ),
        (
            "Quanto é 5 + 4?",
            "O resultado é {{RESULT}}, ou dez.",
            "O resultado é 9.",
        ),
        (
            "Quanto é 5 + 4?",
            "O resultado é {{RESULT}}, ou vinte e cinco.",
            "O resultado é 9.",
        ),
    ],
)
def test_write_response_rejects_additional_numbers_written_as_words(
    message: str,
    provider_response: str,
    expected: str,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(message, 9.0)

    assert result == expected


@pytest.mark.parametrize(
    ("message", "provider_response"),
    [
        (
            "Quanto é 5 + 4?",
            "O resultado é {{RESULT}}, uma informação importante.",
        ),
        (
            "What is 5 + 4?",
            "The result is {{RESULT}}, one important detail.",
        ),
    ],
)
def test_writer_does_not_treat_number_words_as_extra_results(
    message: str,
    provider_response: str,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(message, 9.0)

    assert result == (
        "O resultado é 9, uma informação importante."
        if message.startswith("Quanto")
        else "The result is 9, one important detail."
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "O que é uma equação?",
            "O serviço de IA está indisponível no momento. "
            "Tente novamente mais tarde.",
        ),
        (
            "What is an equation?",
            "The AI service is currently unavailable. " "Please try again later.",
        ),
        (
            "¿Qué es una ecuación?",
            "El servicio de IA no está disponible en este momento. "
            "Inténtalo de nuevo más tarde.",
        ),
    ],
)
def test_write_response_handles_provider_unavailable_deterministically(
    message: str,
    expected: str,
) -> None:
    with patch("agents.writer_agent.get_provider") as mock_get_provider:
        result = write_response(
            message,
            response_type="provider_unavailable",
        )

    assert result == expected
    mock_get_provider.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        "subtraia 2",
        "multiplique por 3",
        "divida por 0",
    ],
)
def test_detect_language_identifies_short_portuguese_commands(
    message: str,
) -> None:
    """Identify short Portuguese math commands as Portuguese."""
    assert _detect_language(message) == "Portuguese"


@pytest.mark.parametrize(
    "unexpected_error",
    [TypeError, AttributeError],
)
def test_writer_propagates_unexpected_provider_errors(
    unexpected_error: type[Exception],
) -> None:
    """Propagate unexpected errors raised while generating a response."""
    provider = Mock()
    provider.generate.side_effect = unexpected_error("Unexpected programming error")

    with patch("agents.writer_agent.get_provider", return_value=provider):
        with pytest.raises(unexpected_error):
            write_response("What is 5 + 4?", 9.0)


@pytest.mark.parametrize(
    "unexpected_error",
    [TypeError, AttributeError],
)
def test_writer_propagates_unexpected_provider_initialization_errors(
    unexpected_error: type[Exception],
) -> None:
    """Propagate unexpected errors raised while initializing the provider."""
    with patch(
        "agents.writer_agent.get_provider",
        side_effect=unexpected_error("Unexpected programming error"),
    ):
        with pytest.raises(unexpected_error):
            write_response("What is 5 + 4?", 9.0)


@pytest.mark.parametrize(
    ("message", "error_type", "error", "expected"),
    [
        (
            "poderia calcular 5 % 2",
            "unsupported_operator",
            "Unsupported mathematical operator.",
            "O operador matem\u00e1tico solicitado n\u00e3o \u00e9 suportado.",
        ),
        (
            "\u00bfPuedes calcular esta operaci\u00f3n?",
            "unsupported_operator",
            "Unsupported mathematical operator.",
            "El operador matem\u00e1tico solicitado no es compatible.",
        ),
        (
            "Could you calculate this operation?",
            "unsupported_operator",
            "Unsupported mathematical operator.",
            "The requested mathematical operator is not supported.",
        ),
        (
            "poderia calcular um valor muito grande",
            "non_finite_result",
            "Mathematical result must be finite.",
            "O resultado matem\u00e1tico n\u00e3o \u00e9 um n\u00famero finito.",
        ),
        (
            "\u00bfPuedes calcular un valor demasiado grande?",
            "non_finite_result",
            "Mathematical result must be finite.",
            "El resultado matem\u00e1tico no es un n\u00famero finito.",
        ),
        (
            "Could you calculate a huge value?",
            "non_finite_result",
            "Mathematical result must be finite.",
            "The mathematical result is not a finite number.",
        ),
    ],
)
def test_writer_uses_typed_math_error_fallbacks(
    message: str,
    error_type: str,
    error: str,
    expected: str,
) -> None:
    """Use deterministic localized fallbacks for explicitly typed math errors."""
    with patch("agents.writer_agent.get_provider") as mock_get_provider:
        response = write_response(
            message,
            error=error,
            error_type=error_type,
        )

    assert response == expected
    mock_get_provider.assert_not_called()

    fallback = fallback_response(
        message,
        error=error,
        error_type=error_type,
    )
    assert fallback == expected


@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize(
    ("message", "error", "expected"),
    [
        (
            "Quanto \u00e9 10 / 0?",
            "Cannot divide by zero",
            "N\u00e3o \u00e9 poss\u00edvel dividir por zero.",
        ),
        (
            "\u00bfPuedes reescribir esta expresi\u00f3n matem\u00e1tica?",
            "Invalid mathematical expression.",
            "No pude interpretar esa expresi\u00f3n matem\u00e1tica. "
            "\u00bfPodr\u00edas reformularla?",
        ),
    ],
)
def test_legacy_math_errors_remain_deterministic(
    use_fallback: bool,
    message: str,
    error: str,
    expected: str,
) -> None:
    """Legacy error-only calls must retain localized deterministic fallbacks."""
    with patch("agents.writer_agent.get_provider") as mock_get_provider:
        if use_fallback:
            response = fallback_response(message, error=error)
        else:
            response = write_response(message, error=error)

    assert response == expected
    mock_get_provider.assert_not_called()


def test_writer_replaces_a_valid_error_placeholder() -> None:
    """Replace a valid error placeholder with the application's error."""
    provider = Mock()
    provider.generate.return_value = (
        "It was not possible to complete the operation: {{ERROR}}."
    )

    with patch("agents.writer_agent.get_provider", return_value=provider):
        response = write_response(
            "calculate an expression",
            error="Invalid expression",
        )

    assert (
        response == "It was not possible to complete the operation: Invalid expression."
    )
    provider.generate.assert_called_once()


def test_legacy_generic_error_fallback_is_deterministic() -> None:
    """Unknown legacy errors use a safe unexpected-error response."""
    response = fallback_response(
        "calculate an expression",
        error="Invalid expression",
    )

    assert response == (
        "An unexpected error occurred while processing your message. "
        "Please try again."
    )


def test_fallback_response_preserves_authoritative_result() -> None:
    """The deterministic fallback must use the supplied result unchanged."""
    response = fallback_response("What is 5 + 4?", result=9.0)

    assert response == "The result is 9."
