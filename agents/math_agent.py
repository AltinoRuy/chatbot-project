from dataclasses import dataclass

from core.context_detector import requires_context
from core.expression_extractor import (
    extract_expression,
    is_invalid_mathematical_expression,
)
from core.intent_parser import parse_intent
from core.tools import add, divide, multiply, subtract
from services.memory import Memory


@dataclass
class MathResult:
    """Structured result returned by the Math Agent."""

    handled: bool
    result: float | None = None
    needs_context: bool = False


def calculate(
    left: float,
    operator: str,
    right: float,
) -> float:
    """Calculates a mathematical operation using the appropriate tool.

    Args:
        left: First operand.
        operator: Mathematical operator.
        right: Second operand.

    Returns:
        The result of the mathematical operation.

    Raises:
        ValueError: If the operator is not supported.
    """
    if operator == "+":
        return add(left, right)

    if operator == "-":
        return subtract(left, right)

    if operator == "*":
        return multiply(left, right)

    if operator == "/":
        return divide(left, right)

    raise ValueError(f"Unsupported operator: {operator}")


def tokenize(expression: str) -> list[str]:
    """Tokenizes a mathematical expression.

    A minus sign is treated as a negative sign when it appears at the
    beginning of the expression or immediately after another operator.
    Otherwise, it is treated as the subtraction operator.

    Args:
        expression: Mathematical expression to tokenize.

    Returns:
        A list containing numbers and operators.

    Raises:
        ValueError: If the expression contains an invalid token.
    """
    expression = expression.replace(" ", "")

    tokens: list[str] = []
    i = 0

    while i < len(expression):
        character = expression[i]

        if character.isdigit() or character == ".":
            start = i

            while i < len(expression) and (
                expression[i].isdigit() or expression[i] == "."
            ):
                i += 1

            tokens.append(expression[start:i])
            continue

        if character == "-" and (not tokens or tokens[-1] in {"+", "-", "*", "/"}):
            i += 1
            start = i

            while i < len(expression) and (
                expression[i].isdigit() or expression[i] == "."
            ):
                i += 1

            if start == i:
                raise ValueError("Invalid mathematical expression.")

            tokens.append(f"-{expression[start:i]}")
            continue

        if character in {"+", "-", "*", "/"}:
            tokens.append(character)
            i += 1
            continue

        raise ValueError("Invalid mathematical expression.")

    return tokens


def evaluate_tokens(tokens: list[str]) -> float:
    """Evaluates tokenized mathematical expressions with operator precedence.

    Multiplication and division are evaluated before addition and subtraction.
    Operations with equal precedence are evaluated from left to right.

    Args:
        tokens: Tokenized mathematical expression.

    Returns:
        The calculated result.

    Raises:
        ValueError: If the token sequence is invalid.
    """
    values: list[float | str] = []

    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token in {"*", "/"}:
            if not values or i + 1 >= len(tokens):
                raise ValueError("Invalid mathematical expression.")

            left = float(values.pop())
            right = float(tokens[i + 1])

            values.append(calculate(left, token, right))
            i += 2
            continue

        values.append(token)
        i += 1

    if not values:
        raise ValueError("Invalid mathematical expression.")

    result = float(values[0])

    i = 1

    while i < len(values):
        operator = values[i]
        right = float(values[i + 1])

        result = calculate(result, operator, right)
        i += 2

    return result


def solve(expression: str) -> float:
    """Solves a normalized mathematical expression.

    Args:
        expression: Normalized mathematical expression.

    Returns:
        The calculated result.

    Raises:
        ValueError: If the expression is invalid.
    """
    tokens = tokenize(expression)

    if len(tokens) < 3:
        raise ValueError("Invalid mathematical expression.")

    return evaluate_tokens(tokens)


def solve_with_context(
    last_result: float,
    operation: str,
    value: float,
) -> float:
    """Calculates an operation using the previous result as the first operand.

    Args:
        last_result: Result from the previous calculation.
        operation: Mathematical operator.
        value: Number to use in the operation.

    Returns:
        The result of the mathematical operation.
    """
    return calculate(last_result, operation, value)


def process_math_message(
    message: str,
    memory: Memory,
) -> MathResult:
    """Processes a user message as a mathematical task.

    This is the public entry point of the Math Agent. It decides whether the
    message is a complete expression, a contextual operation, or not a
    mathematical request.

    Args:
        message: User message.
        memory: Memory instance used to store and retrieve mathematical results.

    Returns:
        A structured MathResult describing the outcome.

    Raises:
        ValueError: If the mathematical expression is invalid.
    """
    expression = extract_expression(message)

    if expression is not None:
        result = solve(expression)
        memory.save_result(result)

        return MathResult(
            handled=True,
            result=result,
        )

    if is_invalid_mathematical_expression(message):
        raise ValueError("Invalid mathematical expression.")

    intent = parse_intent(message)

    if not requires_context(expression, intent):
        return MathResult(handled=False)

    last_result = memory.get_last_result()

    if last_result is None:
        return MathResult(
            handled=True,
            result=None,
            needs_context=True,
        )

    operation, value = intent
    result = solve_with_context(last_result, operation, value)
    memory.save_result(result)

    return MathResult(
        handled=True,
        result=result,
    )
