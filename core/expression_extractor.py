import re


def extract_expression(message: str):
    pattern = r"(\d+)\s*([+\-*/])\s*(\d+)"

    match = re.search(pattern, message)

    if match is None:
        return None

    left, operator, right = match.groups()

    return float(left), operator, float(right)