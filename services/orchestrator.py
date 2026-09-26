from agents.math_agent import process_math_message
from agents.writer_agent import write_response
from services.memory import Memory


def process_message(
    message: str,
    memory: Memory,
) -> str:
    """Processes a user message through the chatbot workflow.

    Args:
        message: User message to process.
        memory: Memory instance used by the math workflow.

    Returns:
        A natural-language response.
    """
    try:
        math_result = process_math_message(message, memory)
    except ValueError as error:
        return str(error)

    if not math_result.handled:
        return write_response(message)

    if math_result.needs_context:
        return write_response(message, None, True)

    return write_response(message, math_result.result)
