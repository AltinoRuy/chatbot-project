from agents.math_agent import solve, solve_with_context
from agents.writer_agent import write_response
from core.context_detector import requires_context
from core.expression_extractor import (
    extract_expression,
    is_invalid_mathematical_expression,
)
from core.intent_parser import parse_intent
from services.memory import get_last_result, save_result


def process_message(message: str) -> str:
    """Processes a user message through the chatbot workflow.

    Args:
        message: User message to process.

    Returns:
        A natural-language response.
    """
    expression = extract_expression(message)

    if expression is not None:
        try:
            result = solve(message)
            save_result(result)
            return write_response(message, result)
        except ValueError as error:
            return str(error)

    if is_invalid_mathematical_expression(message):
        return "Invalid mathematical expression."

    intent = parse_intent(message)

    needs_context = requires_context(expression, intent)

    if needs_context:
        last_result = get_last_result()

        if last_result is None:
            return write_response(message, None, True)

        try:
            operation, value = intent
            result = solve_with_context(last_result, operation, value)
            save_result(result)

            return write_response(message, result)

        except ValueError as error:
            return str(error)

    if intent is None:
        return write_response(message)

    return write_response(message)