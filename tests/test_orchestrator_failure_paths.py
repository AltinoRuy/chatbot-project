"""Regression tests for orchestrator failure paths."""

import logging
from unittest.mock import patch

import pytest

from agents.math_agent import (
    InvalidMathematicalExpressionError,
    MathDomainError,
    MathResult,
    NonFiniteMathResultError,
    UnsupportedMathOperatorError,
)
from core.tools import DivisionByZeroError
from services.memory import Memory
from services.orchestrator import process_message

GENERIC_ERROR_RESPONSES = {
    "Portuguese": (
        "Ocorreu um erro inesperado ao processar sua mensagem. " "Tente novamente."
    ),
    "Spanish": (
        "Se produjo un error inesperado al procesar tu mensaje. "
        "Int\u00e9ntalo de nuevo."
    ),
    "English": (
        "An unexpected error occurred while processing your message. "
        "Please try again."
    ),
}


@pytest.mark.parametrize(
    ("error_type", "message", "language"),
    [
        (ValueError, "Ol\u00e1, tudo bem?", "Portuguese"),
        (ValueError, "\u00bfPuedes ayudarme?", "Spanish"),
        (ValueError, "Hello, how are you?", "English"),
        (RuntimeError, "Ol\u00e1, tudo bem?", "Portuguese"),
        (RuntimeError, "\u00bfPuedes ayudarme?", "Spanish"),
        (RuntimeError, "Hello, how are you?", "English"),
    ],
)
def test_unexpected_math_errors_are_safe_and_localized(
    error_type: type[Exception],
    message: str,
    language: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected math errors must not reach the Writer as domain errors."""
    memory = Memory()
    memory.save_result(17.0)
    secret = "api_key=DO_NOT_EXPOSE"

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=error_type(secret),
        ),
        patch("services.orchestrator.write_response") as mock_writer,
        caplog.at_level(logging.ERROR, logger="services.orchestrator"),
    ):
        response = process_message(message, memory)

    assert response == GENERIC_ERROR_RESPONSES[language]
    mock_writer.assert_not_called()
    assert memory.get_last_result() == 17.0
    assert secret not in response
    assert secret not in caplog.text
    assert error_type.__name__ in caplog.text
    assert "math" in caplog.text.casefold()


def _math_result_for(scenario: str) -> MathResult:
    """Return a representative result for a non-exceptional math path."""
    states = {
        "non_math": MathResult(handled=False),
        "needs_context": MathResult(handled=True, needs_context=True),
        "ambiguous": MathResult(handled=True, needs_clarification=True),
        "interpretation_failed": MathResult(
            handled=True,
            interpretation_failed=True,
        ),
        "provider_unavailable": MathResult(
            handled=True,
            provider_unavailable=True,
        ),
    }
    return states[scenario]


@pytest.mark.parametrize(
    ("scenario", "message", "expected"),
    [
        (
            "non_math",
            "Ol\u00e1, tudo bem?",
            "Posso ajudar com isso, mas n\u00e3o vou realizar c\u00e1lculos.",
        ),
        (
            "non_math",
            "\u00bfPuedes ayudarme?",
            "Puedo ayudarte con eso, pero no realizar\u00e9 c\u00e1lculos.",
        ),
        (
            "non_math",
            "Hello, how are you?",
            "I can help with that, but I will not perform calculations.",
        ),
        (
            "needs_context",
            "divida por 2",
            "N\u00e3o h\u00e1 um resultado matem\u00e1tico anterior para usar.",
        ),
        (
            "needs_context",
            "\u00bfPuedes dividir por 2?",
            "No hay un resultado matem\u00e1tico anterior que se pueda usar.",
        ),
        (
            "needs_context",
            "divide 2",
            "There is no previous mathematical result to use.",
        ),
        (
            "ambiguous",
            "Poderia calcular isso?",
            "Poderia esclarecer o que deseja calcular?",
        ),
        (
            "ambiguous",
            "\u00bfPuedes calcular esto?",
            "\u00bfPodr\u00edas aclarar qu\u00e9 deseas calcular?",
        ),
        (
            "ambiguous",
            "Calculate something",
            "Could you clarify what you want to calculate?",
        ),
        (
            "interpretation_failed",
            "Preciso reformular o pedido.",
            "Poderia reformular seu pedido matem\u00e1tico?",
        ),
        (
            "interpretation_failed",
            "Necesito reformular mi solicitud matem\u00e1tica.",
            "\u00bfPodr\u00edas reformular tu solicitud matem\u00e1tica?",
        ),
        (
            "interpretation_failed",
            "Please rephrase my math request.",
            "Could you rephrase your mathematical request?",
        ),
        (
            "provider_unavailable",
            "N\u00e3o consigo calcular isso.",
            "O servi\u00e7o de IA est\u00e1 indispon\u00edvel no momento. "
            "Tente novamente mais tarde.",
        ),
        (
            "provider_unavailable",
            "No puedes calcular esto.",
            "El servicio de IA no est\u00e1 disponible en este momento. "
            "Int\u00e9ntalo de nuevo m\u00e1s tarde.",
        ),
        (
            "provider_unavailable",
            "The AI service is unavailable.",
            "The AI service is currently unavailable. " "Please try again later.",
        ),
        (
            "division_by_zero",
            "Quanto \u00e9 10 / 0?",
            "N\u00e3o \u00e9 poss\u00edvel dividir por zero.",
        ),
        (
            "division_by_zero",
            "\u00bfCu\u00e1nto es 10 / 0?",
            "No es posible dividir por cero.",
        ),
        (
            "division_by_zero",
            "What is 10 / 0?",
            "It is not possible to divide by zero.",
        ),
        (
            "invalid_expression",
            "Poderia reformular a express\u00e3o matem\u00e1tica?",
            "N\u00e3o consegui interpretar essa express\u00e3o matem\u00e1tica. "
            "Pode reformul\u00e1-la?",
        ),
        (
            "invalid_expression",
            "\u00bfPuedes reescribir esta expresi\u00f3n matem\u00e1tica?",
            "No pude interpretar esa expresi\u00f3n matem\u00e1tica. "
            "\u00bfPodr\u00edas reformularla?",
        ),
        (
            "invalid_expression",
            "Could you rephrase this invalid expression?",
            "I couldn't interpret that mathematical expression. "
            "Could you rephrase it?",
        ),
        (
            "unsupported_operator",
            "Poderia calcular a opera\u00e7\u00e3o 5 % 2?",
            "O operador matem\u00e1tico solicitado n\u00e3o \u00e9 suportado.",
        ),
        (
            "unsupported_operator",
            "\u00bfPuedes calcular esta operaci\u00f3n?",
            "El operador matem\u00e1tico solicitado no es compatible.",
        ),
        (
            "unsupported_operator",
            "Please calculate this operation.",
            "The requested mathematical operator is not supported.",
        ),
        (
            "non_finite_result",
            "Poderia calcular um valor muito grande?",
            "O resultado matem\u00e1tico n\u00e3o \u00e9 um n\u00famero finito.",
        ),
        (
            "non_finite_result",
            "\u00bfPuedes calcular un valor demasiado grande?",
            "El resultado matem\u00e1tico no es un n\u00famero finito.",
        ),
        (
            "non_finite_result",
            "Could you calculate a huge value?",
            "The mathematical result is not a finite number.",
        ),
    ],
)
def test_writer_failure_uses_state_specific_localized_fallback(
    scenario: str,
    message: str,
    expected: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Writer failure must preserve the deterministic fallback for each state."""
    memory = Memory()
    memory.save_result(17.0)
    secret = "writer_exception_DO_NOT_LOG"

    if scenario == "division_by_zero":
        math_patch = patch(
            "services.orchestrator.process_math_message",
            side_effect=DivisionByZeroError("Cannot divide by zero"),
        )
    elif scenario == "invalid_expression":
        math_patch = patch(
            "services.orchestrator.process_math_message",
            side_effect=InvalidMathematicalExpressionError(
                "Invalid mathematical expression."
            ),
        )
    elif scenario == "unsupported_operator":
        math_patch = patch(
            "services.orchestrator.process_math_message",
            side_effect=UnsupportedMathOperatorError("Unsupported operator: %"),
        )
    elif scenario == "non_finite_result":
        math_patch = patch(
            "services.orchestrator.process_math_message",
            side_effect=NonFiniteMathResultError("Mathematical result must be finite."),
        )
    else:
        math_patch = patch(
            "services.orchestrator.process_math_message",
            return_value=_math_result_for(scenario),
        )

    with (
        math_patch,
        patch(
            "services.orchestrator.write_response",
            side_effect=RuntimeError(secret),
        ) as mock_writer,
        caplog.at_level(logging.ERROR, logger="services.orchestrator"),
    ):
        response = process_message(message, memory)

    assert response == expected
    mock_writer.assert_called_once()
    assert memory.get_last_result() == 17.0
    assert secret not in response
    assert secret not in caplog.text
    assert "RuntimeError" in caplog.text
    assert "writer" in caplog.text.casefold()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Quanto \u00e9 5 + 4?", "O resultado \u00e9 9."),
        ("\u00bfCu\u00e1nto es 5 + 4?", "El resultado es 9."),
        ("What is 5 + 4?", "The result is 9."),
    ],
)
def test_writer_failure_preserves_valid_result_in_supported_languages(
    message: str,
    expected: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The deterministic result fallback must preserve the validated result."""
    memory = Memory()
    secret = "writer_exception_DO_NOT_LOG"

    def save_valid_result(
        _message: str,
        target_memory: Memory,
    ) -> MathResult:
        target_memory.save_result(9.0)
        return MathResult(handled=True, result=9.0)

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=save_valid_result,
        ),
        patch(
            "services.orchestrator.write_response",
            side_effect=RuntimeError(secret),
        ),
        caplog.at_level(logging.ERROR, logger="services.orchestrator"),
    ):
        response = process_message(message, memory)

    assert response == expected
    assert memory.get_last_result() == 9.0
    assert secret not in response
    assert secret not in caplog.text
    assert "RuntimeError" in caplog.text
    assert "writer" in caplog.text.casefold()


@pytest.mark.parametrize(
    ("message", "language"),
    [
        ("Quanto \u00e9 5 + 4?", "Portuguese"),
        ("\u00bfCu\u00e1nto es 5 + 4?", "Spanish"),
        ("What is 5 + 4?", "English"),
    ],
)
def test_failure_of_result_fallback_is_safe_and_localized(
    message: str,
    language: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failure of both response generation and its fallback remains contained."""
    memory = Memory()
    writer_secret = "writer_exception_DO_NOT_LOG"
    fallback_secret = "fallback_exception_DO_NOT_LOG"

    def save_valid_result(
        _message: str,
        target_memory: Memory,
    ) -> MathResult:
        target_memory.save_result(9.0)
        return MathResult(handled=True, result=9.0)

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=save_valid_result,
        ),
        patch(
            "services.orchestrator.write_response",
            side_effect=RuntimeError(writer_secret),
        ),
        patch(
            "services.orchestrator.fallback_result_response",
            side_effect=RuntimeError(fallback_secret),
        ),
        caplog.at_level(logging.ERROR, logger="services.orchestrator"),
    ):
        response = process_message(message, memory)

    assert response == GENERIC_ERROR_RESPONSES[language]
    assert memory.get_last_result() == 9.0
    assert writer_secret not in response
    assert fallback_secret not in response
    assert writer_secret not in caplog.text
    assert fallback_secret not in caplog.text
    assert "RuntimeError" in caplog.text
    assert "fallback" in caplog.text.casefold()


@pytest.mark.parametrize(
    ("message", "language"),
    [
        ("Ola, tudo bem?", "Portuguese"),
        ("\u00bfPuedes ayudarme?", "Spanish"),
        ("Hello, how are you?", "English"),
    ],
)
def test_unmapped_math_domain_error_is_safe_and_localized(
    message: str,
    language: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Do not forward an unmapped domain error or its details to the Writer."""
    memory = Memory()
    memory.save_result(17.0)
    secret = "internal_detail_DO_NOT_EXPOSE"

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=MathDomainError(f"Unsupported operator: {secret}"),
        ),
        patch("services.orchestrator.write_response") as mock_writer,
        caplog.at_level(logging.ERROR, logger="services.orchestrator"),
    ):
        response = process_message(message, memory)

    assert response == GENERIC_ERROR_RESPONSES[language]
    mock_writer.assert_not_called()
    assert memory.get_last_result() == 17.0
    assert secret not in response
    assert secret not in caplog.text
    assert "MathDomainError" in caplog.text
    assert "Math domain error" in caplog.text


def test_non_result_fallback_failure_is_safe_and_localized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Contain failures of both Writer and fallback for a context request."""
    memory = Memory()
    memory.save_result(17.0)
    writer_secret = "writer_exception_DO_NOT_LOG"
    fallback_secret = "fallback_exception_DO_NOT_LOG"

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=_math_result_for("needs_context"),
        ),
        patch(
            "services.orchestrator.write_response",
            side_effect=RuntimeError(writer_secret),
        ),
        patch(
            "services.orchestrator.fallback_response",
            side_effect=RuntimeError(fallback_secret),
        ),
        caplog.at_level(logging.ERROR, logger="services.orchestrator"),
    ):
        response = process_message("divide 2", memory)

    assert response == GENERIC_ERROR_RESPONSES["English"]
    assert memory.get_last_result() == 17.0
    assert writer_secret not in response
    assert fallback_secret not in response
    assert writer_secret not in caplog.text
    assert fallback_secret not in caplog.text
    assert "Writer fallback failed" in caplog.text


@pytest.mark.parametrize(
    ("message", "language"),
    [
        ("Ol\u00e1, tudo bem?", "Portuguese"),
        ("\u00bfPuedes ayudarme?", "Spanish"),
        ("Hello, how are you?", "English"),
    ],
)
def test_last_resort_response_survives_localized_fallback_failure(
    message: str,
    language: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Return a safe localized message if the main fallback also fails."""
    memory = Memory()
    math_secret = "math_exception_DO_NOT_LOG"
    fallback_secret = "fallback_exception_DO_NOT_LOG"

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=RuntimeError(math_secret),
        ),
        patch(
            "services.orchestrator.fallback_unexpected_response",
            side_effect=RuntimeError(fallback_secret),
        ),
        caplog.at_level(logging.ERROR, logger="services.orchestrator"),
    ):
        response = process_message(message, memory)

    assert response == GENERIC_ERROR_RESPONSES[language]
    assert math_secret not in response
    assert fallback_secret not in response
    assert math_secret not in caplog.text
    assert fallback_secret not in caplog.text
    assert "Unexpected-response fallback failed" in caplog.text
