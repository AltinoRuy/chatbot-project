import re
from typing import Literal

from llm.provider_factory import get_provider
from llm.providers.base_provider import BaseProvider

ResponseType = Literal["ambiguous", "interpretation_failed"]

PORTUGUESE_MARKERS = {
    "ajuda",
    "bom",
    "boa",
    "calcular",
    "como",
    "conta",
    "das",
    "de",
    "dia",
    "dividir",
    "em",
    "está",
    "estou",
    "mais",
    "não",
    "obrigado",
    "obrigada",
    "olá",
    "para",
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
    "você",
    "vocês",
}

ENGLISH_MARKERS = {
    "about",
    "and",
    "calculate",
    "can",
    "could",
    "divide",
    "hello",
    "help",
    "how",
    "multiply",
    "please",
    "result",
    "subtract",
    "the",
    "there",
    "this",
    "what",
    "with",
    "you",
}

SPANISH_MARKERS = {
    "ayuda",
    "calcular",
    "cómo",
    "cuál",
    "cuánto",
    "cuándo",
    "dividir",
    "es",
    "está",
    "gracias",
    "hola",
    "hacer",
    "multiplicar",
    "necesito",
    "no",
    "operación",
    "para",
    "por",
    "puede",
    "puedes",
    "qué",
    "quiero",
    "resultado",
    "restar",
    "sumar",
    "tengo",
}

NUMBER_WORDS = {
    # English
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
    # Portuguese
    "um",
    "uma",
    "uns",
    "umas",
    "dois",
    "duas",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
    "dez",
    "onze",
    "doze",
    "treze",
    "quatorze",
    "catorze",
    "quinze",
    "dezesseis",
    "dezassete",
    "dezessete",
    "dezoito",
    "dezenove",
    "vinte",
    "trinta",
    "quarenta",
    "cinquenta",
    "sessenta",
    "setenta",
    "oitenta",
    "noventa",
    "cem",
    "cento",
    "duzentos",
    "duzentas",
    "trezentos",
    "trezentas",
    "quatrocentos",
    "quatrocentas",
    "quinhentos",
    "quinhentas",
    "seiscentos",
    "seiscentas",
    "setecentos",
    "setecentas",
    "oitocentos",
    "oitocentas",
    "novecentos",
    "novecentas",
    "mil",
    "milhão",
    "milhões",
    "bilhão",
    "bilhões",
    "trilhão",
    "trilhões",
    # Spanish
    "un",
    "una",
    "unos",
    "unas",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
    "once",
    "doce",
    "trece",
    "catorce",
    "quince",
    "dieciséis",
    "diecisiete",
    "dieciocho",
    "diecinueve",
    "veinte",
    "veintiuno",
    "veintidós",
    "veintitrés",
    "veinticuatro",
    "veinticinco",
    "treinta",
    "cuarenta",
    "cincuenta",
    "sesenta",
    "setenta",
    "ochenta",
    "noventa",
    "cien",
    "ciento",
    "doscientos",
    "doscientas",
    "trescientos",
    "trescientas",
    "cuatrocientos",
    "cuatrocientas",
    "quinientos",
    "quinientas",
    "seiscientos",
    "seiscientas",
    "setecientos",
    "setecientas",
    "ochocientos",
    "ochocientas",
    "novecientos",
    "novecientas",
    "mil",
    "millón",
    "millones",
    "billón",
    "billones",
}

# These words can also function as ordinary determiners.
AMBIGUOUS_NUMBER_WORDS = {
    "one",
    "um",
    "uma",
    "uns",
    "umas",
    "un",
    "una",
    "unos",
    "unas",
    "uno",
}

_NUMBER_WORD_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(word) for word in sorted(NUMBER_WORDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


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
        result: Authoritative mathematical result, if available.
        needs_context: Whether a previous mathematical result is required.
        error: Error message produced by the application.
        response_type: Type of clarification response to generate.

    Returns:
        A localized response that respects the application guardrails.
    """
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

        response = _generate_response_from_provider(prompt)

        if not response or _contains_numbers(response):
            return _fallback_response_type(message, response_type)

        return response

    if error is not None:
        localized_error = _localize_error(message, error)

        if _is_known_error(localized_error):
            return _fallback_error(message, localized_error)

        prompt = _build_prompt(
            message,
            language,
            (
                "The application has already identified an error. "
                "Explain the error naturally without changing its meaning. "
                "Do not calculate anything or introduce additional numbers. "
                "The sentence MUST contain the exact placeholder {{ERROR}}. "
                f"The authoritative error is: {localized_error}"
            ),
        )

        response = _generate_response_from_provider(prompt)

        if not _is_valid_error_response(response):
            return _fallback_error(message, localized_error)

        return response.replace("{{ERROR}}", localized_error)

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

        response = _generate_response_from_provider(prompt)

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

        response = _generate_response_from_provider(prompt)

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

    response = _generate_response_from_provider(prompt)

    if not response or _contains_new_numbers(message, response):
        return _fallback_non_math(message)

    return response


def _generate_response_from_provider(prompt: str) -> str:
    """Generate a response safely, including provider initialization."""
    try:
        provider = get_provider()
    except Exception:
        return ""

    return _generate_response(provider, prompt)


def _generate_response(provider: BaseProvider, prompt: str) -> str:
    """Generate a response safely when the provider raises an exception."""
    try:
        response = provider.generate(prompt)
    except Exception:
        return ""

    return _normalize_response(response)


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
    language_instructions = {
        "Portuguese": (
            "The user's language has been determined as Portuguese. "
            "Respond only in Portuguese."
        ),
        "Spanish": (
            "The user's language has been determined as Spanish. "
            "Respond only in Spanish."
        ),
        "English": (
            "The user's language has been determined as English. "
            "Respond only in English."
        ),
    }

    language_instruction = language_instructions.get(
        language,
        language_instructions["English"],
    )

    return (
        f"{language_instruction}\n\n" f"User message: {message}\n\n" f"{instructions}"
    )


def _detect_language(message: str) -> str:
    """Detect Portuguese, English, or Spanish using lexical markers."""
    normalized_message = message.casefold()
    words = set(re.findall(r"\b\w+\b", normalized_message))

    scores = {
        "Portuguese": len(words & PORTUGUESE_MARKERS),
        "English": len(words & ENGLISH_MARKERS),
        "Spanish": len(words & SPANISH_MARKERS),
    }

    # Distinctive characters provide additional evidence.
    if re.search(r"[ãõç]", normalized_message):
        scores["Portuguese"] += 2

    if re.search(r"[¿¡ñ]", normalized_message):
        scores["Spanish"] += 2

    if re.search(r"[áéíóú]", normalized_message):
        scores["Spanish"] += 1

    best_language = max(scores, key=scores.get)

    if scores[best_language] > 0:
        return best_language

    # Preserve the existing default for expressions and unknown messages.
    return "English"


def _is_portuguese(message: str) -> bool:
    """Return whether the message is classified as Portuguese."""
    return _detect_language(message) == "Portuguese"


def _contains_numbers(response: str) -> bool:
    """Return whether text contains digits or contextual numeric words."""
    return bool(re.search(r"\d", response) or _number_word_matches(response))


def _number_word_matches(text: str) -> set[str]:
    """Extract numeric words while avoiding common determiner false positives.

    Ambiguous words are ignored when followed by an ordinary non-numeric word.
    """
    matches = set()

    for match in _NUMBER_WORD_PATTERN.finditer(text):
        word = match.group(0).casefold()

        if word in AMBIGUOUS_NUMBER_WORDS:
            remaining_text = text[match.end() :]
            next_word_match = re.match(
                r"\s+([A-Za-zÀ-ÿ]+)",
                remaining_text,
            )

            if next_word_match:
                next_word = next_word_match.group(1).casefold()

                if next_word not in NUMBER_WORDS:
                    continue

        matches.add(word)

    return matches


def _contains_new_numbers(message: str, response: str) -> bool:
    """Check whether a response introduces numbers absent from the message."""
    number_pattern = r"\d+(?:[.,]\d+)*"

    input_numbers = set(re.findall(number_pattern, message))
    response_numbers = set(re.findall(number_pattern, response))

    if not response_numbers.issubset(input_numbers):
        return True

    input_number_words = _number_word_matches(message)
    response_number_words = _number_word_matches(response)

    return not response_number_words.issubset(input_number_words)


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


def _is_division_by_zero_error(error: str) -> bool:
    """Return whether an error describes division by zero."""
    normalized_error = error.casefold()

    zero_terms = ("zero", "cero")
    division_terms = ("divid", "division", "división")

    return any(term in normalized_error for term in zero_terms) and any(
        term in normalized_error for term in division_terms
    )


def _is_invalid_expression_error(error: str) -> bool:
    """Return whether an error describes an invalid mathematical expression."""
    normalized_error = error.casefold()

    return any(
        phrase in normalized_error
        for phrase in (
            "invalid mathematical expression",
            "expressão matemática inválida",
            "expresión matemática inválida",
        )
    )


def _is_known_error(error: str) -> bool:
    """Return whether an error has a deterministic user-facing response."""
    return _is_division_by_zero_error(error) or _is_invalid_expression_error(error)


def _localize_error(message: str, error: str) -> str:
    """Translate known application errors into the user's language."""
    language = _detect_language(message)

    if _is_division_by_zero_error(error):
        localized_errors = {
            "Portuguese": "não é possível dividir por zero",
            "Spanish": "no es posible dividir por cero",
            "English": "division by zero is not allowed",
        }
        return localized_errors[language]

    if _is_invalid_expression_error(error):
        localized_errors = {
            "Portuguese": "expressão matemática inválida",
            "Spanish": "expresión matemática inválida",
            "English": "invalid mathematical expression",
        }
        return localized_errors[language]

    return error


def _fallback_result(message: str, formatted_result: str) -> str:
    """Return a deterministic response preserving the authoritative result."""
    language = _detect_language(message)

    if language == "Portuguese":
        return f"O resultado é {formatted_result}."

    if language == "Spanish":
        return f"El resultado es {formatted_result}."

    return f"The result is {formatted_result}."


def _fallback_context(message: str) -> str:
    """Explain that no previous mathematical result is available."""
    language = _detect_language(message)

    if language == "Portuguese":
        return "Não há um resultado matemático anterior para usar."

    if language == "Spanish":
        return "No hay un resultado matemático anterior que se pueda usar."

    return "There is no previous mathematical result to use."


def _fallback_error(message: str, error: str) -> str:
    """Return a deterministic, localized response for a mathematical error."""
    language = _detect_language(message)

    if _is_division_by_zero_error(error):
        if language == "Portuguese":
            return "Não é possível dividir por zero."

        if language == "Spanish":
            return "No es posible dividir por cero."

        return "It is not possible to divide by zero."

    if _is_invalid_expression_error(error):
        if language == "Portuguese":
            return (
                "Não consegui interpretar essa expressão matemática. "
                "Pode reformulá-la?"
            )

        if language == "Spanish":
            return (
                "No pude interpretar esa expresión matemática. "
                "¿Podrías reformularla?"
            )

        return (
            "I couldn't interpret that mathematical expression. "
            "Could you rephrase it?"
        )

    if language == "Portuguese":
        return f"Não foi possível concluir a operação: {error}."

    if language == "Spanish":
        return f"No fue posible completar la operación: {error}."

    return f"It was not possible to complete the operation: {error}."


def _fallback_response_type(
    message: str,
    response_type: ResponseType,
) -> str:
    """Return a deterministic clarification or interpretation fallback."""
    language = _detect_language(message)

    if response_type == "ambiguous":
        if language == "Portuguese":
            return "Poderia esclarecer o que deseja calcular?"

        if language == "Spanish":
            return "¿Podrías aclarar qué deseas calcular?"

        return "Could you clarify what you want to calculate?"

    if language == "Portuguese":
        return "Poderia reformular seu pedido matemático?"

    if language == "Spanish":
        return "¿Podrías reformular tu solicitud matemática?"

    return "Could you rephrase your mathematical request?"


def _fallback_non_math(message: str) -> str:
    """Return a safe fallback for non-mathematical requests."""
    language = _detect_language(message)

    if language == "Portuguese":
        return "Posso ajudar com isso, mas não vou realizar cálculos."

    if language == "Spanish":
        return "Puedo ayudarte con eso, pero no realizaré cálculos."

    return "I can help with that, but I will not perform calculations."
