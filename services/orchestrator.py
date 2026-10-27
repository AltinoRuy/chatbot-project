import logging

from agents.math_agent import MathDomainError, process_math_message
from agents.writer_agent import (
    fallback_response,
    fallback_result_response,
    fallback_unexpected_response,
    write_response,
)
from core.tools import DivisionByZeroError
from services.memory import Memory

logger = logging.getLogger(__name__)

_GENERIC_ERROR_RESPONSE = (
    "An unexpected error occurred while processing your message. " "Please try again."
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
        return _write_with_fallback(
            message,
            error="Cannot divide by zero",
        )
    except MathDomainError as error:
        if str(error) == "Invalid mathematical expression.":
            return _write_with_fallback(
                message,
                error="Invalid mathematical expression.",
            )

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
        return _write_with_fallback(
            message,
            response_type="ambiguous",
        )

    if math_result.interpretation_failed:
        return _write_with_fallback(
            message,
            response_type="interpretation_failed",
        )

    if math_result.provider_unavailable:
        return _write_with_fallback(
            message,
            response_type="provider_unavailable",
        )

    if not math_result.handled:
        return _write_with_fallback(message)

    if math_result.needs_context:
        return _write_with_fallback(
            message,
            needs_context=True,
        )

    if math_result.result is not None:
        return _write_result_response(message, math_result.result)

    return _write_with_fallback(message)


def _write_with_fallback(
    message: str,
    result: float | None = None,
    needs_context: bool = False,
    error: str | None = None,
    response_type: str | None = None,
) -> str:
    """Generate a response and use the matching deterministic fallback."""
    if error is not None:
        state = "math-error"
    elif response_type is not None:
        state = response_type
    elif needs_context:
        state = "context"
    else:
        state = "non-math"

    try:
        if error is not None:
            return write_response(message, error=error)

        if response_type is not None:
            return write_response(message, response_type=response_type)

        if needs_context:
            return write_response(message, None, True)

        if result is not None:
            return write_response(message, result)

        return write_response(message)
    except Exception as writer_error:
        logger.error(
            "Writer failed while preparing a %s response (%s).",
            state,
            type(writer_error).__name__,
        )

    try:
        return fallback_response(
            message,
            result=result,
            needs_context=needs_context,
            error=error,
            response_type=response_type,
        )
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


def _safe_unexpected_response(message: str) -> str:
    """Return a localized error without exposing exception details."""
    try:
        return fallback_unexpected_response(message)
    except Exception as fallback_error:
        logger.error(
            "Unexpected-response fallback failed (%s).",
            type(fallback_error).__name__,
        )
        return _GENERIC_ERROR_RESPONSE
