from typing import TypeAlias

MathIntent: TypeAlias = tuple[str, float]


def requires_context(
    expression: str | None,
    intent: MathIntent | None,
) -> bool:
    """Determines whether a mathematical request requires previous context.

    Args:
        expression: Complete mathematical expression extracted from the
            user message, if one was found.
        intent: Contextual mathematical operation and operand, if one was
            identified.

    Returns:
        True if no complete expression was found but a contextual operation
        was identified; otherwise, False.
    """
    return expression is None and intent is not None
