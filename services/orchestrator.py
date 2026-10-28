import logging
import unicodedata
from typing import Literal

from agents.math_agent import (
    InvalidMathematicalExpressionError,
    MathDomainError,
    NonFiniteMathResultError,
    UnsupportedMathOperatorError,
    process_math_message,
)
from agents.writer_agent import (
    MathErrorType,
    ResponseType,
    fallback_response,
    fallback_result_response,
    fallback_unexpected_response,
    write_response,
)
from core.tools import DivisionByZeroError
from services.memory import Memory

logger = logging.getLogger(__name__)

_GENERIC_ERROR_RESPONSES = {
    "pt": ("Ocorreu um erro inesperado ao processar sua mensagem. " "Tente novamente."),
    "es": (
        "Se produjo un error inesperado al procesar tu mensaje. "
        "Int\u00e9ntalo de nuevo."
    ),
    "en": (
        "An unexpected error occurred while processing your message. "
        "Please try again."
    ),
}

_PORTUGUESE_FALLBACK_MARKERS = (
    "ola",
    "tudo bem",
    "voce",
    "nao",
    "quanto",
    "qual",
    "adicione",
    "subtraia",
    "multiplique",
    "divida",
    "calcule",
    "poderia",
    "expressao matematica",
    "o resultado",
    "resultado e",
    "bom dia",
    "boa tarde",
    "boa noite",
)

_SPANISH_FALLBACK_MARKERS = (
    "puedes",
    "necesito",
    "cuanto",
    "que es",
    "ayudame",
    "calcula",
    "suma",
    "resta",
    "multiplica",
    "resultado es",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
)

FallbackState = (
    Literal[
        "non_math",
        "context",
        "division_by_zero",
        "invalid_expression",
        "unsupported_operator",
        "non_finite_result",
    ]
    | ResponseType
)

_RESPONSE_TYPES = frozenset(
    {"ambiguous", "interpretation_failed", "provider_unavailable"}
)

_MATH_ERROR_MESSAGES = {
    "division_by_zero": "Cannot divide by zero",
    "invalid_expression": "Invalid mathematical expression.",
    "unsupported_operator": "Unsupported mathematical operator.",
    "non_finite_result": "Mathematical result must be finite.",
}

_MATH_ERROR_TYPES: dict[str, MathErrorType] = {
    "division_by_zero": "division_by_zero",
    "invalid_expression": "invalid_expression",
    "unsupported_operator": "unsupported_operator",
    "non_finite_result": "non_finite_result",
}

_VALID_FALLBACK_STATES = (
    _RESPONSE_TYPES
    | frozenset(_MATH_ERROR_MESSAGES)
    | frozenset({"context", "non_math"})
)


def process_message(
    message: str,
    memory: Memory,
) -> str:
    """Process a message while containing unexpected application errors.

    Args:
        message: Original user message.
        memory: Session memory for mathematical context.

    Returns:
        A response safe to display to the user.
    """
    try:
        return _process_message(message, memory)
    except Exception as error:
        logger.error(
            "Unexpected error at the message boundary (%s).",
            type(error).__name__,
        )
        return _safe_unexpected_response(message)


def _process_message(
    message: str,
    memory: Memory,
) -> str:
    """Process a user message through the chatbot workflow.

    Args:
        message: User message to process.
        memory: Memory instance used by the math workflow.

    Returns:
        A natural-language response.
    """
    try:
        math_result = process_math_message(message, memory)
    except DivisionByZeroError:
        return _write_with_fallback(message, "division_by_zero")
    except InvalidMathematicalExpressionError:
        return _write_with_fallback(message, "invalid_expression")
    except UnsupportedMathOperatorError as error:
        logger.warning(
            "Unsupported mathematical operator encountered (%s).",
            type(error).__name__,
        )
        return _write_with_fallback(message, "unsupported_operator")
    except NonFiniteMathResultError as error:
        logger.error(
            "Mathematical calculation produced a non-finite result (%s).",
            type(error).__name__,
        )
        return _write_with_fallback(message, "non_finite_result")
    except MathDomainError as error:
        logger.error(
            "Math domain error could not be mapped safely (%s).",
            type(error).__name__,
        )
        return _safe_unexpected_response(message)
    except Exception as error:
        logger.error(
            "Math processing failed unexpectedly (%s).",
            type(error).__name__,
        )
        return _safe_unexpected_response(message)

    if math_result.needs_clarification:
        return _write_with_fallback(message, "ambiguous")

    if math_result.interpretation_failed:
        return _write_with_fallback(message, "interpretation_failed")

    if math_result.provider_unavailable:
        return _write_with_fallback(message, "provider_unavailable")

    if not math_result.handled:
        return _write_with_fallback(message, "non_math")

    if math_result.needs_context:
        return _write_with_fallback(message, "context")

    if math_result.result is not None:
        return _write_result_response(message, math_result.result)

    return _write_with_fallback(message, "non_math")


def _write_with_fallback(
    message: str,
    state: FallbackState = "non_math",
) -> str:
    """Generate a response using an explicit, typed response state."""
    if state not in _VALID_FALLBACK_STATES:
        raise ValueError("Unsupported response fallback state.")

    try:
        if state in _RESPONSE_TYPES:
            return write_response(message, response_type=state)

        if state == "context":
            return write_response(message, None, True)

        if state in _MATH_ERROR_MESSAGES:
            return write_response(
                message,
                error=_MATH_ERROR_MESSAGES[state],
                error_type=_MATH_ERROR_TYPES[state],
            )

        return write_response(message)
    except Exception as writer_error:
        logger.error(
            "Writer failed while preparing a %s response (%s).",
            state,
            type(writer_error).__name__,
        )

    try:
        if state in _RESPONSE_TYPES:
            return fallback_response(message, response_type=state)

        if state == "context":
            return fallback_response(message, needs_context=True)

        if state in _MATH_ERROR_MESSAGES:
            return fallback_response(
                message,
                error=_MATH_ERROR_MESSAGES[state],
                error_type=_MATH_ERROR_TYPES[state],
            )

        return fallback_response(message)
    except Exception as fallback_error:
        logger.error(
            "Writer fallback failed for a %s response (%s).",
            state,
            type(fallback_error).__name__,
        )
        raise


def _write_result_response(message: str, result: float) -> str:
    """Preserve an authoritative result if response generation fails."""
    try:
        return write_response(message, result)
    except Exception as writer_error:
        logger.error(
            "Writer failed after a successful mathematical operation (%s).",
            type(writer_error).__name__,
        )

    try:
        return fallback_result_response(message, result)
    except Exception as fallback_error:
        logger.error(
            "Result fallback failed after Writer failure (%s).",
            type(fallback_error).__name__,
        )
        raise


def _last_resort_unexpected_response(message: str) -> str:
    """Choose a static localized response without using the Writer's detector."""
    try:
        normalized = unicodedata.normalize("NFKD", message.casefold())
        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
    except Exception:
        return _GENERIC_ERROR_RESPONSES["en"]

    if any(marker in normalized for marker in _PORTUGUESE_FALLBACK_MARKERS):
        return _GENERIC_ERROR_RESPONSES["pt"]

    if any(marker in normalized for marker in _SPANISH_FALLBACK_MARKERS):
        return _GENERIC_ERROR_RESPONSES["es"]

    return _GENERIC_ERROR_RESPONSES["en"]


def _safe_unexpected_response(message: str) -> str:
    """Return a localized error without exposing exception details."""
    try:
        return fallback_unexpected_response(message)
    except Exception as fallback_error:
        logger.error(
            "Unexpected-response fallback failed (%s).",
            type(fallback_error).__name__,
        )
        return _last_resort_unexpected_response(message)
