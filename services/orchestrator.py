from agents.math_agent import calculate, solve
from agents.writer_agent import write_response
from core.expression_extractor import extract_expression
from core.intent_parser import parse_intent
from services.memory import get_last_result, save_result


def process_message(message: str) -> str:
    expression = extract_expression(message)

    if expression is not None:
        result = solve(message)
        save_result(result)
        return write_response(message, result)

    last_result = get_last_result()
    result = None

    intent = parse_intent(message)

    if intent is not None:
        operation, value = intent
        result = calculate(last_result, operation, value)
        save_result(result)

    return write_response(message, result)
