import math
import re

NUMBER_WORDS = {
    # Portuguese
    "zero": 0,
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "dos": 2,
    "tres": 3,
    "três": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
    # English
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    # Spanish
    "uno": 1,
    "una": 1,
    "un": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "dieciséis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21,
    "veintiún": 21,
    "veintiun": 21,
    "veintiuna": 21,
    "veintidós": 22,
    "veintidos": 22,
    "veintitrés": 23,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiséis": 26,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}


def _parse_number(value: str) -> float | None:
    """Parse a numeric value or a supported number word.

    Args:
        value: Value to parse.

    Returns:
        The parsed finite number, or None if the value is invalid.
    """
    value = value.lower().strip()

    if value in NUMBER_WORDS:
        return float(NUMBER_WORDS[value])

    try:
        number = float(value)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def parse_intent(message: str) -> tuple[str, float] | None:
    """Parse a contextual mathematical intent.

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
            prefix + r"(?:sumar|suma)\s*(.+)$",
            "+",
        ),
        (
            prefix + r"(?:restar|resta)\s*(.+)$",
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
