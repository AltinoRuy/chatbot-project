from agents.math_agent import process_math_message
from agents.writer_agent import write_response
from core.tools import DivisionByZeroError
from services.memory import Memory


def process_message(
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

    return write_response(message, math_result.result)
