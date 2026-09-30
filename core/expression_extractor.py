import re

from core.intent_parser import parse_intent

NUMBER_WORDS = {
    "zero": "0",
    "um": "1",
    "uma": "1",
    "dois": "2",
    "duas": "2",
    "tres": "3",
    "três": "3",
    "trÃªs": "3",
    "trãªs": "3",
    "trÃƒÂªs": "3",
    "trãƒâªs": "3",
    "quatro": "4",
    "cuatro": "4",
    "cinco": "5",
    "seis": "6",
    "sete": "7",
    "oito": "8",
    "nove": "9",
    "dez": "10",
    "vinte": "20",
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
    "twenty": "20",
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
    "mÃƒÂ¡s": "+",
    "mÃ¡s": "+",
    "mãƒâ¡s": "+",
    "mã¡s": "+",
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

    for word, number in sorted(
        NUMBER_WORDS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        expression = re.sub(
            rf"\b{re.escape(word)}\b",
            number,
            expression,
        )

    expression = re.sub(r"\s+", "", expression)

    return expression


def _is_valid_candidate(candidate: str) -> bool:
    """Validates the syntax of a mathematical expression candidate.

    Args:
        candidate: Candidate mathematical expression.

    Returns:
        True if the candidate contains valid operands, operators,
        and balanced parentheses.
    """
    if not candidate:
        return False

    if not re.fullmatch(r"[0-9.+\-*/()]+", candidate):
        return False

    depth = 0
    expect_operand = True
    index = 0

    while index < len(candidate):
        character = candidate[index]

        if character == "(":
            if not expect_operand:
                return False

            depth += 1
            index += 1
            continue

        if character == ")":
            if expect_operand:
                return False

            depth -= 1

            if depth < 0:
                return False

            expect_operand = False
            index += 1
            continue

        if character in {"+", "-", "*", "/"}:
            if expect_operand:
                if character != "-":
                    return False

                index += 1

                if index >= len(candidate):
                    return False

                if not candidate[index].isdigit():
                    return False

                start = index

                while index < len(candidate) and (
                    candidate[index].isdigit() or candidate[index] == "."
                ):
                    index += 1

                number = candidate[start:index]

                if number.count(".") > 1 or number == ".":
                    return False

                expect_operand = False
                continue

            expect_operand = True
            index += 1
            continue

        if character.isdigit():
            start = index

            while index < len(candidate) and (
                candidate[index].isdigit() or candidate[index] == "."
            ):
                index += 1

            number = candidate[start:index]

            if number.count(".") > 1 or number == ".":
                return False

            expect_operand = False
            continue

        return False

    return depth == 0 and not expect_operand


def _extract_candidate(expression: str) -> str | None:
    """Extracts a complete mathematical expression from normalized text.

    Args:
        expression: Normalized expression candidate.

    Returns:
        A complete mathematical expression if one is found, otherwise None.
    """
    allowed_characters = set("0123456789.+-*/()")

    candidates: list[str] = []
    current: list[str] = []

    for character in expression:
        if character in allowed_characters:
            current.append(character)
            continue

        if current:
            candidates.append("".join(current))
            current = []

    if current:
        candidates.append("".join(current))

    for candidate in candidates:
        if _is_valid_candidate(candidate) and _contains_operation(candidate):
            return candidate

    return None


def _contains_operation(expression: str) -> bool:
    """Checks whether an expression contains a mathematical operation.

    Args:
        expression: Mathematical expression.

    Returns:
        True if the expression contains at least one operator.
    """
    return bool(re.search(r"[+\-*/]", expression))


def extract_expression(message: str) -> str | None:
    """Extracts a complete mathematical expression from a message.

    Contextual mathematical operations are excluded so they can be handled
    by the intent parser using the previous mathematical result.

    Args:
        message: User message containing a mathematical expression.

    Returns:
        The normalized mathematical expression if valid, otherwise None.
    """
    if is_context_operation(message):
        return None

    expression = normalize_expression(message)

    if not expression:
        return None

    return _extract_candidate(expression)


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

    if re.fullmatch(pattern, expression):
        return True

    return parse_intent(message) is not None


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
