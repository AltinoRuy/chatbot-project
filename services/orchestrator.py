from agents.math_agent import solve, solve_with_context
from agents.writer_agent import write_response
from core.context_detector import requires_context
from core.expression_extractor import extract_expression
from core.intent_parser import parse_intent
from services.memory import get_last_result, save_result


def process_message(message: str) -> str:
    """Processes a user message through the chatbot workflow.

    Args:
        message: User message to process.

    Returns:
        A user-friendly response generated from the processed request.
    """
    expression = extract_expression(message)

    if expression is not None:
        result = solve(message)
        save_result(result)
        return write_response(message, result)

    intent = parse_intent(message)
    needs_context = requires_context(expression, intent)
    result = None

    if needs_context:
        last_result = get_last_result()

        if last_result is None:
            return write_response(message, None, True)

        operation, value = intent
        result = solve_with_context(last_result, operation, value)
        save_result(result)

    return write_response(message, result)
