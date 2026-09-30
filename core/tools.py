class DivisionByZeroError(ValueError):
    """Raised when a division by zero is attempted."""


def add(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of a and b.
    """
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtracts one number from another.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The difference between a and b.
    """
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiplies two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The product of a and b.
    """
    return a * b


def divide(a: float, b: float) -> float:
    """Divides one number by another.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        The quotient of a divided by b.

    Raises:
        DivisionByZeroError: If b is zero.
    """
    if b == 0:
        raise DivisionByZeroError("Cannot divide by zero")

    return a / b
