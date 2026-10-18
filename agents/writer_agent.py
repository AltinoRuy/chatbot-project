import re
from typing import Literal

from llm.provider_factory import get_provider

ResponseType = Literal["ambiguous", "interpretation_failed"]

PORTUGUESE_MARKERS = {
    "a",
    "ajuda",
    "bom",
    "boa",
    "calcular",
    "como",
    "conta",
    "da",
    "das",
    "de",
    "dia",
    "dividir",
    "e",
    "é",
    "em",
    "está",
    "estou",
    "mais",
    "não",
    "no",
    "o",
    "obrigado",
    "obrigada",
    "olá",
    "para",
    "por",
    "poderia",
    "preciso",
    "qual",
    "quando",
    "quanto",
    "que",
    "quero",
    "resultado",
    "reformular",
    "seu",
    "soma",
    "subtrair",
    "tudo",
    "um",
    "uma",
    "você",
    "vocês",
    "três",
}

ENGLISH_MARKERS = {
    "a",
    "about",
    "and",
    "can",
    "calculate",
    "could",
    "divide",
    "do",
    "for",
    "hello",
    "help",
    "how",
    "i",
    "is",
    "it",
    "me",
    "multiply",
    "of",
    "please",
    "result",
    "subtract",
    "the",
    "there",
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
    response_type: ResponseType | None = None,
) -> str:
    """Generate a response while preserving authoritative application data.

    Args:
        message: Original user message.
        result: Authoritative result produced by the math workflow.
        needs_context: Whether the request requires a previous result.
        error: Error message produced by the application.
        response_type: Type of clarification response to generate.

    Returns:
        A localized response that respects the application guardrails.
    """
    provider = get_provider()
    language = _detect_language(message)

    if response_type is not None:
        prompts = {
            "ambiguous": (
                "The mathematical request is ambiguous. "
                "Ask the user to clarify the intended operation or interpretation."
            ),
            "interpretation_failed": (
                "The mathematical request could not be interpreted reliably. "
                "Ask the user to rephrase the mathematical request."
            ),
        }

        prompt = _build_prompt(
            message,
            language,
            (
                "The request has not been interpreted with sufficient certainty. "
                f"{prompts[response_type]} "
                "Do not calculate anything. "
                "Do not guess the intended operation. "
                "Do not include numerical values. "
                "Respond with one short sentence."
            ),
        )

        response = _normalize_response(provider.generate(prompt))

        if not response or _contains_numbers(response):
            return _fallback_response_type(message, response_type)

        return response

    if error is not None:
        prompt = _build_prompt(
            message,
            language,
            (
                "The application has already identified an error. "
                "Explain the error naturally without changing its meaning. "
                "Do not calculate anything or introduce additional numbers. "
                "The sentence MUST contain the exact placeholder {{ERROR}}. "
                f"The authoritative error is: {error}"
            ),
        )

        response = _normalize_response(provider.generate(prompt))

        if not _is_valid_error_response(response):
            return _fallback_error(message, error)

        return response.replace("{{ERROR}}", error)

    if result is not None:
        formatted_result = _format_result(result)

        prompt = _build_prompt(
            message,
            language,
            (
                "The mathematical operation has already been calculated. "
                "The authoritative result is provided by the application. "
                "Do not calculate anything. "
                "Do not interpret, reinterpret, or modify the mathematical operation. "
                "Do not introduce any additional numbers. "
                "The sentence MUST contain the exact placeholder {{RESULT}}. "
                f"The authoritative result is {formatted_result}."
            ),
        )

        response = _normalize_response(provider.generate(prompt))

        if not _is_valid_result_response(response):
            return _fallback_result(message, formatted_result)

        return response.replace("{{RESULT}}", formatted_result)

    if needs_context:
        prompt = _build_prompt(
            message,
            language,
            (
                "The user requested an operation that requires a previous result, "
                "but no usable previous result is available. "
                "Explain this limitation briefly. "
                "Do not calculate anything or introduce numbers."
            ),
        )

        response = _normalize_response(provider.generate(prompt))

        if not response or _contains_numbers(response):
            return _fallback_context(message)

        return response

    prompt = _build_prompt(
        message,
        language,
        (
            "Respond naturally to the user's message. "
            "Do not perform mathematical calculations. "
            "Do not introduce numerical results."
        ),
    )

    response = _normalize_response(provider.generate(prompt))

    if not response or _contains_new_numbers(message, response):
        return _fallback_non_math(message)

    return response


def _normalize_response(response: str | None) -> str:
    """Normalize provider output and reject empty or whitespace-only content."""
    if not isinstance(response, str):
        return ""

    return response.strip()


def _build_prompt(
    message: str,
    language: str,
    instructions: str,
) -> str:
    """Build a prompt with explicit language and behavior constraints."""
    language_instruction = (
        "The user's language has already been determined as Portuguese. "
        "Respond only in Portuguese."
        if language == "Portuguese"
        else "The user's language has already been determined as English. "
        "Respond only in English."
    )

    return f"{language_instruction}\n\nUser message: {message}\n\n{instructions}"


def _detect_language(message: str) -> str:
    """Detect whether the message is Portuguese or English."""
    words = set(re.findall(r"\b\w+\b", message.casefold()))

    portuguese_score = len(words & PORTUGUESE_MARKERS)
    english_score = len(words & ENGLISH_MARKERS)

    if portuguese_score > english_score:
        return "Portuguese"

    if english_score > portuguese_score:
        return "English"

    if re.search(r"[áâãàçéêíóôõú]", message.casefold()):
        return "Portuguese"

    return "English"


def _is_portuguese(message: str) -> bool:
    """Return whether the message is classified as Portuguese."""
    return _detect_language(message) == "Portuguese"


def _contains_numbers(response: str) -> bool:
    """Return whether a response contains numeric digits."""
    return bool(re.search(r"\d", response))


def _contains_new_numbers(message: str, response: str) -> bool:
    """Check whether a response introduces numbers absent from the message."""
    number_pattern = r"\d+(?:[.,]\d+)*"

    input_numbers = set(re.findall(number_pattern, message))
    response_numbers = set(re.findall(number_pattern, response))

    return not response_numbers.issubset(input_numbers)


def _is_valid_result_response(response: str) -> bool:
    """Validate the result placeholder and prevent additional numbers."""
    if not response or response.count("{{RESULT}}") != 1:
        return False

    without_placeholder = response.replace("{{RESULT}}", "")

    return not _contains_numbers(without_placeholder)


def _is_valid_error_response(response: str) -> bool:
    """Validate the error placeholder and prevent additional numbers."""
    if not response or response.count("{{ERROR}}") != 1:
        return False

    without_placeholder = response.replace("{{ERROR}}", "")

    return not _contains_numbers(without_placeholder)


def _format_result(result: float) -> str:
    """Format a numeric result without unnecessary decimal zeros."""
    numeric_result = float(result)

    if numeric_result.is_integer():
        return str(int(numeric_result))

    return str(numeric_result)


def _fallback_result(message: str, formatted_result: str) -> str:
    """Return a deterministic response preserving the authoritative result."""
    if _is_portuguese(message):
        return f"O resultado é {formatted_result}."

    return f"The result is {formatted_result}."


def _fallback_context(message: str) -> str:
    """Explain that no previous mathematical result is available."""
    if _is_portuguese(message):
        return "Não há um resultado matemático anterior para usar."

    return "There is no previous mathematical result to use."


def _fallback_error(message: str, error: str) -> str:
    """Return a deterministic response for a mathematical error."""
    normalized_error = error.casefold()

    is_division_by_zero = "zero" in normalized_error and (
        "divid" in normalized_error or "division" in normalized_error
    )

    if is_division_by_zero:
        if _is_portuguese(message):
            return "Não é possível dividir por zero."

        return "It is not possible to divide by zero."

    if _is_portuguese(message):
        return f"Não foi possível concluir a operação: {error}."

    return f"It was not possible to complete the operation: {error}."


def _fallback_response_type(
    message: str,
    response_type: ResponseType,
) -> str:
    """Return a deterministic clarification or interpretation fallback."""
    if _is_portuguese(message):
        if response_type == "ambiguous":
            return "Poderia esclarecer o que deseja calcular?"

        return "Poderia reformular seu pedido matemático?"

    if response_type == "ambiguous":
        return "Could you clarify what you want to calculate?"

    return "Could you rephrase your mathematical request?"


def _fallback_non_math(message: str) -> str:
    """Return a safe fallback for non-mathematical requests."""
    if _is_portuguese(message):
        return "Posso ajudar com isso, mas não vou realizar cálculos."

    return "I can help with that, but I will not perform calculations."
