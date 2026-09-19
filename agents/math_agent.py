from core.expression_extractor import extract_expression
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
    elif operator == "-":
        return subtract(left, right)
    elif operator == "*":
        return multiply(left, right)
    elif operator == "/":
        return divide(left, right)
    else:
        raise ValueError(f"Unsupported operator: {operator}")


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


def solve(message: str) -> float:
    """Solves a complete mathematical expression.

    Args:
        message: User message containing a mathematical expression.

    Returns:
        The result of the mathematical expression.

    Raises:
        ValueError: If the message does not contain a valid expression.
    """
    expression = extract_expression(message)

    if expression is None:
        raise ValueError(
            "I couldn't understand the expression. Try something like: 5 + 4"
        )

    left, operator, right = expression

    return calculate(left, operator, right)
