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
    "cuatro": "4",
    "cinco": "5",
    "seis": "6",
    "sete": "7",
    "oito": "8",
    "nove": "9",
    "dez": "10",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


OPERATOR_WORDS = {
    "dividido por": "/",
    "divided by": "/",
    "dividido entre": "/",
    "divided into": "/",
    "multiplicado por": "*",
    "multiplied by": "*",
    "vezes": "*",
    "times": "*",
    "más": "+",
    "mas": "+",
    "mais": "+",
    "plus": "+",
    "menos": "-",
    "minus": "-",
    "subtract": "-",
    "subtraia": "-",
    "subtrair": "-",
}


def normalize_expression(message: str) -> str:
    """Normalizes mathematical words and symbols.

    The normalization supports common mathematical vocabulary in Portuguese,
    English, and Spanish.

    Args:
        message: Message containing a mathematical expression.

    Returns:
        A normalized message containing mathematical symbols.
    """
    expression = message.lower().strip()

    expression = re.sub(r"(\d),(\d)", r"\1.\2", expression)

    for text, symbol in sorted(
        OPERATOR_WORDS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        expression = re.sub(
            rf"\b{re.escape(text)}\b",
            symbol,
            expression,
        )

    for word, number in NUMBER_WORDS.items():
        expression = re.sub(
            rf"\b{re.escape(word)}\b",
            number,
            expression,
        )

    expression = re.sub(r"\s+", "", expression)

    return expression


def _extract_candidate(expression: str) -> str | None:
    """Extracts a mathematical expression from normalized text.

    Args:
        expression: Normalized expression candidate.

    Returns:
        A mathematical expression if one is found, otherwise None.
    """
    number = r"-?\d+(?:\.\d+)?"
    operator = r"[+\-*/]"

    pattern = rf"(?P<expression>{number}(?:{operator}{number})+)"

    matches = re.findall(pattern, expression)

    if not matches:
        return None

    return matches[0]


def extract_expression(message: str) -> str | None:
    """Extracts a complete mathematical expression from a message.

    The expression may appear by itself or inside a natural-language sentence.

    Args:
        message: User message containing a mathematical expression.

    Returns:
        The normalized mathematical expression if valid, otherwise None.
    """
    expression = normalize_expression(message)

    if not expression:
        return None

    candidate = _extract_candidate(expression)

    if candidate is None:
        return None

    return candidate


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


def _has_math_operator(expression: str) -> bool:
    """Checks whether an expression contains a mathematical operator.

    Args:
        expression: Normalized expression to inspect.

    Returns:
        True if a mathematical operator is present.
    """
    return bool(re.search(r"[+\-*/]", expression))


def _looks_like_math_expression(expression: str) -> bool:
    """Checks whether text has the structure of a mathematical expression.

    A mathematical expression is considered malformed when an operator
    separates two non-empty operands.

    Args:
        expression: Normalized expression to inspect.

    Returns:
        True if the text resembles a malformed mathematical expression.
    """
    return bool(
        re.search(
            r"[^\s+\-*/]+[+\-*/][^\s+\-*/]+",
            expression,
        )
    )


def is_invalid_mathematical_expression(message: str) -> bool:
    """Checks whether a message is a malformed mathematical expression.

    Args:
        message: User message to inspect.

    Returns:
        True if the message appears to be an invalid mathematical expression.
    """
    expression = normalize_expression(message)

    if not expression:
        return False

    if extract_expression(message) is not None:
        return False

    if is_context_operation(message):
        return False

    if not _has_math_operator(expression):
        return False

    if _looks_like_math_expression(expression):
        return True

    contains_number = bool(re.search(r"\d", expression))
    contains_operator = bool(re.search(r"[+\-*/]", expression))

    return contains_number and contains_operator
