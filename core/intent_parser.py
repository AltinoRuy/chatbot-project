import re
from typing import Optional

from llm.provider_factory import get_provider


def parse_intent(
    message: str,
) -> Optional[tuple[str, float]]:
    """Parses the user's message into a mathematical intent.

    Args:
        message: User message to analyze.

    Returns:
        A tuple containing the operation and numeric value,
        or None if no valid mathematical intent is found.
    """
    if re.match(r"^\s*\d+(?:\.\d+)?\s*[+\-*/]\s*$", message):
        return None

    prompt = (
        "Identify whether the user is requesting a mathematical operation "
        "using a number. "
        "Return exactly this format: OPERATION|NUMBER. "
        "Return none if the message is not a valid mathematical request. "
        "Only return a mathematical operation when the requested value is a number. "
        "Words are not numbers and must not be interpreted as numbers. "
        "Do not calculate the result. "
        "Do not explain anything. "
        "The operation must be exactly one of: +, -, *, /. "
        "The number must contain only numeric characters, with "
        "an optional decimal part. "
        "Examples: "
        "subtract 2 -> -|2. "
        "multiply 3 -> *|3. "
        "divide 2 -> /|2. "
        "add 5 -> +|5. "
        "banana + banana -> none. "
        "hello -> none.\n\n"
        f"User message: {message}"
    )

    provider = get_provider()
    result = provider.generate(prompt).strip()

    if result.lower() == "none":
        return None

    parts = result.split("|")

    if len(parts) != 2:
        return None

    operation, value = parts

    operation = operation.strip()
    value = value.strip()

    if operation not in ["+", "-", "*", "/"]:
        return None

    if value == "":
        return None

    try:
        numeric_value = float(value)
    except ValueError:
        return None

    return operation, numeric_value
