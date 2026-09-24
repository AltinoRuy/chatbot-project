from core.expression_extractor import normalize_expression
from core.tools import add, divide, multiply, subtract


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


def solve(message: str) -> float:
    """Solves a complete mathematical expression.

    Args:
        message: User message containing a mathematical expression.

    Returns:
        The result of the mathematical expression.

    Raises:
        ValueError: If the message does not contain a valid expression.
    """
    expression = normalize_expression(message)
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
