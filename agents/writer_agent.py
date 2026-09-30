import re

from llm.provider_factory import get_provider

PORTUGUESE_MARKERS = {
    "a",
    "as",
    "ao",
    "aos",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "é",
    "em",
    "essa",
    "esse",
    "isso",
    "mais",
    "menos",
    "não",
    "o",
    "os",
    "para",
    "por",
    "quanto",
    "qual",
    "que",
    "resultado",
    "sobre",
    "uma",
    "um",
    "agora",
    "calcule",
    "calcular",
    "divida",
    "divide",
    "multiplique",
    "some",
    "subtraia",
    "vinte",
    "três",
    "tres",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
    "dez",
}

ENGLISH_MARKERS = {
    "a",
    "an",
    "and",
    "are",
    "calculate",
    "calculation",
    "by",
    "divide",
    "divided",
    "do",
    "for",
    "from",
    "how",
    "is",
    "it",
    "minus",
    "multiply",
    "now",
    "of",
    "plus",
    "result",
    "subtract",
    "the",
    "then",
    "this",
    "to",
    "what",
    "with",
    "you",
}


def write_response(
    message: str,
    result: float | None = None,
    needs_context: bool = False,
    error: str | None = None,
) -> str:
    """Generates a natural-language response for the user.

    Args:
        message: Original user message.
        result: Authoritative result produced by the math system.
        needs_context: Whether the request requires a previous result.
        error: Mathematical error that should be communicated to the user.

    Returns:
        A natural-language response.
    """
    provider = get_provider()
    language = _detect_language(message)

    if error is not None:
        prompt = (
            "You are a response writer.\n"
            f"The user's language has already been determined as {language}.\n"
            f"Respond only in {language}.\n"
            "A mathematical operation could not be completed.\n"
            "The application has provided the authoritative error message.\n"
            "Your task is ONLY to communicate this error naturally.\n"
            "Do not calculate anything.\n"
            "Do not modify the meaning of the error.\n"
            "Do not introduce any numerical values.\n"
            "Return exactly one short and natural sentence.\n"
            "The sentence MUST contain the exact placeholder {{ERROR}}.\n"
            "Do not replace, remove, or modify the placeholder.\n\n"
            f"User message: {message}\n"
            "Authoritative error: {{ERROR}}"
        )

        response = provider.generate(prompt).strip()

        if not _is_valid_error_response(response):
            return _fallback_error(message, error)

        return response.replace("{{ERROR}}", _format_error(error))

    if result is not None:
        prompt = (
            "You are a response writer.\n"
            f"The user's language has already been determined as {language}.\n"
            f"Respond only in {language}.\n"
            "The mathematical operation has already been calculated.\n"
            "The authoritative result is provided by the application.\n"
            "Your task is ONLY to communicate that result naturally.\n"
            "Do not calculate anything.\n"
            "Do not interpret, reinterpret, or modify the mathematical "
            "operation.\n"
            "Do not add any numbers from your own reasoning.\n"
            "Return exactly one short and natural sentence.\n"
            "The sentence MUST contain the exact placeholder {{RESULT}}.\n"
            "Do not replace, remove, or modify the placeholder.\n\n"
            f"User message: {message}\n"
            "Authoritative result: {{RESULT}}"
        )

        response = provider.generate(prompt).strip()

        if not _is_valid_result_response(response):
            return _fallback_result(message, result)

        return response.replace("{{RESULT}}", _format_result(result))

    if needs_context:
        prompt = (
            "You are a response writer.\n"
            f"The user's language has already been determined as {language}.\n"
            f"Respond only in {language}.\n"
            "The user requested a mathematical operation that requires "
            "a previous result, but no previous result exists.\n"
            "Explain this briefly and naturally.\n"
            "Do not calculate anything.\n"
            "Do not introduce or derive any numerical values.\n"
            "Return exactly one short sentence.\n\n"
            f"User message: {message}"
        )

        response = provider.generate(prompt).strip()

        if _contains_numbers(response):
            return _fallback_context(message)

        return response

    prompt = (
        "You are a response writer.\n"
        f"The user's language has already been determined as {language}.\n"
        f"Respond only in {language}.\n"
        "Respond naturally and briefly.\n"
        "Do not perform mathematical calculations.\n"
        "Do not add, subtract, multiply, divide, count, or compare "
        "numerical values.\n"
        "Do not derive a numerical result from numbers mentioned by the "
        "user.\n"
        "Do not introduce any new numerical values in your response.\n"
        "If the user mentions numbers, treat them only as information "
        "provided by the user and do not calculate with them.\n\n"
        f"User message: {message}"
    )

    response = provider.generate(prompt).strip()

    if _contains_new_numbers(message, response):
        return _fallback_non_math(message)

    return response


def _detect_language(message: str) -> str:
    """Determines the response language using deterministic word markers.

    Portuguese is selected when Portuguese markers are stronger than English
    markers. English is selected when English markers are stronger or when
    the message is ambiguous.

    Args:
        message: User message.

    Returns:
        The language name used by the Writer Agent.
    """
    words = set(re.findall(r"\b[\wÀ-ÿ]+\b", message.lower()))

    portuguese_score = len(words & PORTUGUESE_MARKERS)
    english_score = len(words & ENGLISH_MARKERS)

    if portuguese_score > english_score:
        return "Portuguese"

    if english_score > portuguese_score:
        return "English"

    return "English"


def _is_portuguese(message: str) -> bool:
    """Detects whether the user message is written in Portuguese."""
    return _detect_language(message) == "Portuguese"


def _fallback_result(message: str, result: float) -> str:
    """Returns a localized fallback containing the authoritative result."""
    formatted_result = _format_result(result)

    if _is_portuguese(message):
        return f"O resultado é {formatted_result}."

    return f"The result is {formatted_result}."


def _fallback_context(message: str) -> str:
    """Returns a localized fallback for missing mathematical context."""
    if _is_portuguese(message):
        return "Não há um resultado matemático anterior para usar."

    return "There is no previous mathematical result to use."


def _fallback_error(message: str, error: str) -> str:
    """Returns a localized fallback for a mathematical error."""
    if error == "Cannot divide by zero":
        if _is_portuguese(message):
            return "Não é possível dividir por zero."

        return "It is not possible to divide by zero."

    return error


def _fallback_non_math(message: str) -> str:
    """Returns a localized fallback when the Writer violates numeric guardrails."""
    if _is_portuguese(message):
        return "Posso ajudar com isso, mas não vou realizar cálculos."

    return "I can help with that, but I will not perform calculations."


def _is_valid_result_response(response: str) -> bool:
    """Validates a Writer response containing an authoritative result."""
    if response.count("{{RESULT}}") != 1:
        return False

    response_without_placeholder = response.replace("{{RESULT}}", "")
    return not re.search(r"\d", response_without_placeholder)


def _is_valid_error_response(response: str) -> bool:
    """Validates a Writer response containing an authoritative error."""
    if response.count("{{ERROR}}") != 1:
        return False

    response_without_placeholder = response.replace("{{ERROR}}", "")
    return not re.search(r"\d", response_without_placeholder)


def _contains_numbers(response: str) -> bool:
    """Checks whether a response contains numerical digits."""
    return bool(re.search(r"\d", response))


def _contains_new_numbers(message: str, response: str) -> bool:
    """Checks whether a response introduces numerical values not in the input."""
    input_numbers = re.findall(r"\d+(?:[.,]\d+)*", message)
    response_numbers = re.findall(r"\d+(?:[.,]\d+)*", response)

    return any(number not in input_numbers for number in response_numbers)


def _format_result(result: float) -> str:
    """Formats a numerical result for display."""
    if result.is_integer():
        return str(int(result))

    return str(result)


def _format_error(error: str) -> str:
    """Formats an authoritative mathematical error for display."""
    return error
