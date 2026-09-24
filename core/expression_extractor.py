import re

NUMBER_WORDS = {
    "zero": "0",
    "um": "1",
    "uma": "1",
    "dois": "2",
    "duas": "2",
    "tres": "3",
    "três": "3",
    "quatro": "4",
    "cinco": "5",
    "seis": "6",
    "sete": "7",
    "oito": "8",
    "nove": "9",
    "dez": "10",
}


OPERATOR_WORDS = {
    "mais": "+",
    "menos": "-",
    "vezes": "*",
    "dividido por": "/",
}


def normalize_expression(message: str) -> str:
    """Normalizes mathematical words and spacing into symbols.

    Args:
        message: User message containing a mathematical expression.

    Returns:
        A normalized mathematical expression.
    """
    expression = message.lower().strip()

    for text, symbol in OPERATOR_WORDS.items():
        expression = expression.replace(text, symbol)

    for word, number in NUMBER_WORDS.items():
        expression = re.sub(rf"\b{word}\b", number, expression)

    expression = re.sub(r"\s+", "", expression)

    return expression


def extract_expression(message: str) -> str | None:
    """Extracts and validates a complete mathematical expression.

    Args:
        message: User message containing a mathematical expression.

    Returns:
        The normalized expression if valid, otherwise None.
    """
    expression = normalize_expression(message)

    number = r"-?\d+(?:\.\d+)?"
    operator = r"[+\-*/]"
    pattern = rf"^{number}(?:{operator}{number})+$"

    if re.fullmatch(pattern, expression):
        return expression

    return None


def is_context_operation(message: str) -> bool:
    """Checks whether a message represents a contextual operation.

    Contextual operations use the previous result as the first operand.

    Args:
        message: User message to inspect.

    Returns:
        True if the message represents a valid contextual operation.
    """
    expression = normalize_expression(message)
    number = r"-?\d+(?:\.\d+)?"

    pattern = rf"^[+\-*/]{number}$"

    return re.fullmatch(pattern, expression) is not None


def is_invalid_mathematical_expression(message: str) -> bool:
    """Checks whether a message is a malformed mathematical expression.

    Args:
        message: User message to inspect.

    Returns:
        True if the message is a malformed mathematical expression.
    """
    expression = normalize_expression(message)

    if not expression:
        return False

    if extract_expression(message) is not None:
        return False

    if is_context_operation(message):
        return False

    contains_number = bool(re.search(r"\d", expression))
    contains_operator = bool(re.search(r"[+\-*/]", expression))

    return contains_number and contains_operator


def is_mathematical_input(message: str) -> bool:
    """Checks whether a message appears to be a mathematical input.

    Args:
        message: User message to inspect.

    Returns:
        True when the message contains a number and a mathematical operator.
    """
    expression = normalize_expression(message)

    contains_number = bool(re.search(r"\d", expression))
    contains_operator = bool(re.search(r"[+\-*/]", expression))

    return contains_number and contains_operator
