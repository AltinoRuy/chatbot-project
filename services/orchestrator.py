import logging

from agents.math_agent import process_math_message
from agents.writer_agent import fallback_result_response, write_response
from core.tools import DivisionByZeroError
from services.memory import Memory

logger = logging.getLogger(__name__)


def process_message(
    message: str,
    memory: Memory,
) -> str:
    """Process a message and contain unexpected errors at the app boundary.

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
            "Unexpected error while processing a user message (%s).",
            type(error).__name__,
        )
        return (
            "An unexpected error occurred while processing your message. "
            "Please try again."
        )


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
    except DivisionByZeroError as error:
        return write_response(
            message,
            error=str(error),
        )
    except ValueError as error:
        return write_response(
            message,
            error=str(error),
        )

    if math_result.needs_clarification:
        return write_response(
            message,
            response_type="ambiguous",
        )

    if math_result.interpretation_failed:
        return write_response(
            message,
            response_type="interpretation_failed",
        )

    if math_result.provider_unavailable:
        return write_response(
            message,
            response_type="provider_unavailable",
        )

    if not math_result.handled:
        return write_response(message)

    if math_result.needs_context:
        return write_response(message, None, True)

    if math_result.result is not None:
        try:
            return write_response(message, math_result.result)
        except Exception as error:
            logger.error(
                "Writer failed after a successful mathematical operation (%s).",
                type(error).__name__,
            )
            return fallback_result_response(message, math_result.result)

    return write_response(message, math_result.result)
