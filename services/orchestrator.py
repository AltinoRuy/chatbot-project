from agents.math_agent import solve
from agents.writer_agent import write_response
from core.expression_extractor import extract_expression


def process_message(message: str) -> str:
    expression = extract_expression(message)

    if expression is not None:
        result = solve(message)
        return write_response(message, result)

    return write_response(message, None)