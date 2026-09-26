from agents.math_agent import process_math_message
from agents.writer_agent import write_response


def process_message(message: str) -> str:
    """Processes a user message through the chatbot workflow.

    Args:
        message: User message to process.

    Returns:
        A natural-language response.
    """
    try:
        math_result = process_math_message(message)
    except ValueError as error:
        return str(error)

    if not math_result.handled:
        return write_response(message)

    if math_result.needs_context:
        return write_response(message, None, True)

    return write_response(message, math_result.result)