import re

from llm.provider_factory import get_provider

NUMBER_WORDS = {
    "zero": 0,
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "três": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
}


def _parse_number(value: str) -> float | None:
    """Parses a numeric value or a supported number word.

    Args:
        value: Value to parse.

    Returns:
        The parsed number, or None if the value is invalid.
    """
    value = value.lower().strip()

    if value in NUMBER_WORDS:
        return float(NUMBER_WORDS[value])

    try:
        return float(value)
    except ValueError:
        return None


def parse_intent(message: str) -> tuple[str, float] | None:
    """Parses a contextual mathematical intent from the user message.

    Deterministic patterns are evaluated before using the language model.
    This prevents the language model from interpreting mathematical
    operations that can be handled reliably by application logic.

    Args:
        message: User message.

    Returns:
        A tuple containing the mathematical operator and operand,
        or None when no mathematical intent is detected.
    """
    message = message.lower().strip()

    if re.match(r"^\d+(?:\.\d+)?\s*[+\-*/]\s*$", message):
        return None

    shorthand = re.match(
        r"^([+\-*/])\s*(-?\d+(?:\.\d+)?)$",
        message,
    )

    if shorthand:
        operator, value = shorthand.groups()
        return operator, float(value)

    patterns = [
        (r"^mais\s*(.+)$", "+"),
        (r"^menos\s*(.+)$", "-"),
        (r"^(?:vezes|multiplica(?:r)?\s*por)\s*(.+)$", "*"),
        (
            r"^(?:dividido\s*por|divide(?:\s*por)?|divida(?:\s*por)?)" r"\s*(.+)$",
            "/",
        ),
    ]

    for pattern, operator in patterns:
        match = re.match(pattern, message)

        if match:
            number = _parse_number(match.group(1))

            if number is not None:
                return operator, number

    prompt = (
        "Identify the user's mathematical intent.\n\n"
        "Return exactly one of these formats:\n"
        "+|5\n"
        "-|3\n"
        "*|2\n"
        "/|10\n\n"
        "Rules:\n"
        "- The number must always be positive.\n"
        "- Do not calculate.\n"
        '- If it is not a mathematical request, return "none".\n\n'
        f"User message: {message}"
    )

    provider = get_provider()
    response = provider.generate(prompt).strip()

    if response.lower() == "none":
        return None

    parts = response.split("|")

    if len(parts) != 2:
        return None

    operator, value = parts

    if operator not in {"+", "-", "*", "/"}:
        return None

    try:
        numeric_value = abs(float(value))
    except ValueError:
        return None

    return operator, numeric_value
