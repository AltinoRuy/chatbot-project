from llm.provider_factory import get_provider


def write_response(
    message: str,
    result: float | None = None,
    needs_context: bool = False,
) -> str:
    """Generates a natural-language response for the user.

    Args:
        message: Original user message.
        result: Authoritative result produced by the math system.
        needs_context: Whether the request requires a previous result.

    Returns:
        A natural-language response.
    """
    provider = get_provider()

    if result is not None:
        prompt = (
            "You are a response writer.\n"
            "Respond in the same language as the user.\n"
            "Your task is only to communicate an already calculated result.\n"
            "Do not calculate anything.\n"
            "Do not modify the result.\n"
            "Do not add, remove, or change numbers.\n"
            "Return exactly one short and natural sentence.\n"
            "The sentence MUST contain the exact placeholder {{RESULT}}.\n"
            "Do not replace, remove, or modify the placeholder.\n\n"
            f"User message: {message}\n"
            "Authoritative result: {{RESULT}}"
        )

        response = provider.generate(prompt).strip()

        if "{{RESULT}}" not in response:
            return f"The result is {result}."

        return response.replace("{{RESULT}}", _format_result(result))

    if needs_context:
        prompt = (
            "You are a response writer.\n"
            "Respond in the same language as the user.\n"
            "The user requested a mathematical operation that requires "
            "a previous result, but no previous result exists.\n"
            "Explain this briefly and naturally.\n"
            "Do not calculate anything.\n"
            "Return exactly one short sentence.\n\n"
            f"User message: {message}"
        )

        return provider.generate(prompt).strip()

    prompt = (
        "You are a response writer.\n"
        "Respond in the same language as the user.\n"
        "Respond naturally and briefly.\n"
        "Do not perform mathematical calculations.\n\n"
        f"User message: {message}"
    )

    return provider.generate(prompt).strip()


def _format_result(result: float) -> str:
    """Formats a numerical result for display.

    Args:
        result: Authoritative numerical result.

    Returns:
        A human-readable representation of the result.
    """
    if result.is_integer():
        return str(int(result))

    return str(result)
