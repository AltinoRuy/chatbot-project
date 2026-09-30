import re

NUMBER_WORDS = {
    "zero": 0,
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "três": 3,
    "trãªs": 3,
    "trãƒâªs": 3,
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
    """Parses a contextual mathematical intent.

    Mathematical intent is detected using deterministic patterns only.
    Unrecognized messages return None instead of being interpreted by a
    language model.

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

    prefix = (
        r"(?:agora|ahora|now|then|então|entao|después|despues|depois|"
        r"por favor|please)?\s*"
    )

    result_reference = r"(?:(?:that|the\s+result|el\s+resultado|o\s+resultado)\s*)?"

    patterns = [
        (
            prefix
            + r"(?:mais|adicione|adicionar|some|somar|add|plus)\s*"
            + result_reference
            + r"(.+)$",
            "+",
        ),
        (
            prefix + r"(?:menos|subtraia(?:\s+por)?|subtrair(?:\s+por)?"
            r"|tire|tirar|subtract|minus)\s*" + result_reference + r"(.+)$",
            "-",
        ),
        (
            prefix + r"(?:vezes|multiplica(?:r)?\s*por|multiplique\s*por"
            r"|multiply(?:\s+(?:that|the\s+result))?(?:\s+by)?|times)\s*" + r"(.+)$",
            "*",
        ),
        (
            prefix + r"(?:dividido\s*por|divide(?:\s+(?:that|the\s+result))?"
            r"(?:\s*por|\s*by)?|divida(?:\s*por)?"
            r"|divided\s+by)\s*" + r"(.+)$",
            "/",
        ),
        (
            prefix + r"(?:suma|sumar)\s*(.+)$",
            "+",
        ),
        (
            prefix + r"(?:resta|restar)\s*(.+)$",
            "-",
        ),
        (
            prefix
            + r"(?:multiplica(?:\s+por)?|multiplicar\s+por)\s*"
            + result_reference
            + r"(.+)$",
            "*",
        ),
        (
            prefix + r"(?:divide(?:\s+(?:that|the\s+result|el\s+resultado|"
            r"o\s+resultado))?(?:\s+por|\s+by)?|dividir\s+por)\s*" + r"(.+)$",
            "/",
        ),
    ]

    for pattern, operator in patterns:
        match = re.match(pattern, message)

        if match:
            number = _parse_number(match.group(1))

            if number is not None:
                return operator, number

    return None
