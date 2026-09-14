from typing import Optional, Tuple


def extract_expression(message: str) -> Optional[Tuple[float, str, float]]:
    parts = message.split()

    if len(parts) != 3:
        return None

    left, operator, right = parts

    try:
        return float(left), operator, float(right)

    except ValueError:
        return None