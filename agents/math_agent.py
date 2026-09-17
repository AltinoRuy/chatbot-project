from core.expression_extractor import extract_expression
from core.tools import add, divide, multiply, subtract


def calculate(left: float, operator: str, right: float):
    if operator == "+":
        return add(left, right)

    elif operator == "-":
        return subtract(left, right)

    elif operator == "*":
        return multiply(left, right)

    elif operator == "/":
        return divide(left, right)

    else:
        return "Unsupported operator"


def solve_addition(a: float, b: float) -> float:
    return add(a, b)


def solve(message: str):
    expression = extract_expression(message)

    if expression is None:
        return "I couldn't understand the expression. Try something like: 5 + 4"

    left, operator, right = expression

    return calculate(left, operator, right)
