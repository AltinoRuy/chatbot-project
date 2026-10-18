import json
import math
import re
from dataclasses import dataclass
from typing import Any

from core.context_detector import requires_context
from core.expression_extractor import (
    extract_expression,
    is_invalid_mathematical_expression,
    normalize_expression,
)
from core.intent_parser import parse_intent
from core.tools import add, divide, multiply, subtract
from llm.provider_factory import get_provider
from services.memory import Memory

MATH_AGENT_PROMPT = """
You are the Math Agent responsible for interpreting mathematical requests.

Return exactly one valid JSON object without Markdown or extra text.

Supported intents:
- expression: {"intent": "expression", "expression": "5+4"}
- contextual_operation: {
    "intent": "contextual_operation",
    "operator": "/",
    "operand": 2
  }
- not_math: {"intent": "not_math"}
- ambiguous: {"intent": "ambiguous"}

Rules:
- Never calculate or provide a mathematical result.
- Never invent missing numbers or operations.
- Use only +, -, *, and / as operators.
- Use expression only when the complete expression is explicit.
- Use contextual_operation when the user refers to a previous result
  and specifies an operation and operand.
- Use not_math for ordinary non-mathematical messages.
- Use ambiguous when the mathematical request cannot be interpreted
  reliably.
- The application executes calculations using mathematical tools.
"""


@dataclass
class MathResult:
    """Structured result returned by the Math Agent.

    Attributes:
        handled: Whether the message was identified as a mathematical task.
        result: Authoritative mathematical result, if available.
        needs_context: Whether a previous mathematical result is required.
        needs_clarification: Whether the user must clarify the request.
        interpretation_failed: Whether LLM interpretation failed validation
            or could not be completed.
    """

    handled: bool
    result: float | None = None
    needs_context: bool = False
    needs_clarification: bool = False
    interpretation_failed: bool = False


def calculate(left: float, operator: str, right: float) -> float:
    """Execute a supported operation through the mathematical tools."""
    if operator == "+":
        return add(left, right)
    if operator == "-":
        return subtract(left, right)
    if operator == "*":
        return multiply(left, right)
    if operator == "/":
        return divide(left, right)

    raise ValueError(f"Unsupported operator: {operator}")


def tokenize(expression: str) -> list[str]:
    """Convert a mathematical expression into tokens."""
    expression = expression.replace(" ", "")
    tokens: list[str] = []
    i = 0

    while i < len(expression):
        character = expression[i]

        if character.isdigit() or character == ".":
            start = i
            while i < len(expression) and (
                expression[i].isdigit() or expression[i] == "."
            ):
                i += 1

            number = expression[start:i]
            if number.count(".") > 1 or number == ".":
                raise ValueError("Invalid mathematical expression.")

            tokens.append(number)
            continue

        if character == "-" and (not tokens or tokens[-1] in {"+", "-", "*", "/", "("}):
            i += 1
            start = i
            while i < len(expression) and (
                expression[i].isdigit() or expression[i] == "."
            ):
                i += 1

            if start == i:
                raise ValueError("Invalid mathematical expression.")

            number = expression[start:i]
            if number.count(".") > 1 or number == ".":
                raise ValueError("Invalid mathematical expression.")

            tokens.append(f"-{number}")
            continue

        if character in {"+", "-", "*", "/", "(", ")"}:
            tokens.append(character)
            i += 1
            continue

        raise ValueError("Invalid mathematical expression.")

    return tokens


class _ExpressionParser:
    """Parse expressions while delegating arithmetic to the tools."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0

    def parse(self) -> float:
        if not self.tokens:
            raise ValueError("Invalid mathematical expression.")

        result = self._parse_expression()
        if self.position != len(self.tokens):
            raise ValueError("Invalid mathematical expression.")

        return result

    def _parse_expression(self) -> float:
        result = self._parse_term()

        while self._match("+") or self._match("-"):
            operator = self.tokens[self.position - 1]
            right = self._parse_term()
            result = calculate(result, operator, right)

        return result

    def _parse_term(self) -> float:
        result = self._parse_factor()

        while self._match("*") or self._match("/"):
            operator = self.tokens[self.position - 1]
            right = self._parse_factor()
            result = calculate(result, operator, right)

        return result

    def _parse_factor(self) -> float:
        if self._match("("):
            result = self._parse_expression()
            if not self._match(")"):
                raise ValueError("Invalid mathematical expression.")
            return result

        if self.position >= len(self.tokens):
            raise ValueError("Invalid mathematical expression.")

        token = self.tokens[self.position]
        if token in {"+", "-", "*", "/", ")"}:
            raise ValueError("Invalid mathematical expression.")

        self.position += 1

        try:
            return float(token)
        except ValueError as error:
            raise ValueError("Invalid mathematical expression.") from error

    def _match(self, token: str) -> bool:
        if self.position >= len(self.tokens):
            return False

        if self.tokens[self.position] != token:
            return False

        self.position += 1
        return True


def evaluate_tokens(tokens: list[str]) -> float:
    """Evaluate tokens using the expression parser and mathematical tools."""
    return _ExpressionParser(tokens).parse()


def solve(expression: str) -> float:
    """Evaluate a complete mathematical expression."""
    tokens = tokenize(expression)
    if not tokens:
        raise ValueError("Invalid mathematical expression.")

    return evaluate_tokens(tokens)


def solve_with_context(
    last_result: float,
    operation: str,
    value: float,
) -> float:
    """Apply an operation to the previous result."""
    return calculate(last_result, operation, value)


def _failed_interpretation() -> dict[str, Any]:
    """Represent an unsuccessful LLM interpretation explicitly."""
    return {"intent": "interpretation_failed"}


def _expression_is_grounded(expression: str, message: str) -> bool:
    """Check whether the proposed expression matches explicit message content."""
    expression_pattern = (
        r"(?<![\w.])-?\d+(?:[.,]\d+)?"
        r"(?:\s*[+\-*/]\s*-?\d+(?:[.,]\d+)?)+"
        r"(?![\w.])"
    )

    try:
        normalized_expression = normalize_expression(expression)
        expected_tokens = tokenize(normalized_expression)
    except (TypeError, ValueError):
        return False

    if not expected_tokens:
        return False

    candidates = re.findall(expression_pattern, message)

    if len(candidates) != 1:
        return False

    try:
        actual_tokens = tokenize(normalize_expression(candidates[0]))
    except (TypeError, ValueError):
        return False

    return expected_tokens == actual_tokens


def _contextual_operation_is_grounded(
    operator: str,
    operand: float,
    message: str,
) -> bool:
    """Check whether an operation and operand are explicit in the message."""
    operation_patterns = {
        "+": (
            r"\b(?:add|addition|plus|sum|increase by|somar|some|mais|"
            r"adicionar|adicione)\b"
        ),
        "-": (
            r"\b(?:subtract|subtraction|minus|take away|decrease by|"
            r"subtrair|subtraia|menos|diminuir)\b"
        ),
        "*": (
            r"\b(?:multiply|multiplication|multiplied|times|product|"
            r"vezes|multiplicar|multiplique)\b"
        ),
        "/": (r"\b(?:divide|division|divided|dividir|divida|dividido|" r"divisão)\b"),
    }

    detected_operations = []

    for candidate_operator, pattern in operation_patterns.items():
        for match in re.finditer(pattern, message, flags=re.IGNORECASE):
            detected_operations.append((candidate_operator, match))

    # Reject messages that do not clearly specify exactly one operation.
    if len(detected_operations) != 1:
        return False

    detected_operator, operation_match = detected_operations[0]
    if detected_operator != operator:
        return False

    number_words = {
        "zero": 0,
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
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "um": 1,
        "uma": 1,
        "dois": 2,
        "duas": 2,
        "três": 3,
        "tres": 3,
        "quatro": 4,
        "cinco": 5,
        "seis": 6,
        "sete": 7,
        "oito": 8,
        "nove": 9,
        "dez": 10,
        "onze": 11,
        "doze": 12,
        "treze": 13,
        "catorze": 14,
        "quatorze": 14,
        "quinze": 15,
        "dezesseis": 16,
        "dezassete": 17,
        "dezessete": 17,
        "dezoito": 18,
        "dezenove": 19,
        "vinte": 20,
    }

    compound_number_pattern = r"twenty-(?:one|two|three|four|five|six|seven|eight|nine)"
    number_word_pattern = "|".join(
        [
            compound_number_pattern,
            *(re.escape(word) for word in sorted(number_words, key=len, reverse=True)),
        ]
    )
    number_pattern = (
        rf"(?<!\w)-?\d+(?:[.,]\d+)?(?!\w)|" rf"\b(?:{number_word_pattern})\b"
    )

    # Only consider numbers appearing after the operation wording.
    remaining_message = message[operation_match.end() :]
    number_matches = list(
        re.finditer(number_pattern, remaining_message, flags=re.IGNORECASE)
    )

    # Multiple possible operands are ambiguous, so fail closed.
    if len(number_matches) != 1:
        return False

    operand_text = number_matches[0].group(0).lower()

    try:
        if operand_text in number_words:
            mentioned_operand = float(number_words[operand_text])
        elif operand_text.startswith("twenty-"):
            unit = operand_text.removeprefix("twenty-")
            mentioned_operand = float(20 + number_words[unit])
        else:
            mentioned_operand = float(operand_text.replace(",", "."))
    except (KeyError, ValueError):
        return False

    return math.isclose(
        mentioned_operand,
        float(operand),
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _interpret_with_llm(message: str) -> dict[str, Any]:
    """Interpret a request and validate the provider's structured response."""
    prompt = (
        f"{MATH_AGENT_PROMPT}\n"
        "Interpret the following user message. Return only the JSON object.\n"
        f"User message: {json.dumps(message, ensure_ascii=True)}"
    )

    try:
        response = get_provider().generate(prompt)
        interpretation = json.loads(response)
    except Exception:
        return _failed_interpretation()

    if not isinstance(interpretation, dict):
        return _failed_interpretation()

    intent = interpretation.get("intent")

    if intent in {"not_math", "ambiguous"}:
        return {"intent": intent}

    if intent == "expression":
        expression = interpretation.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return _failed_interpretation()

        try:
            tokens = tokenize(expression)
        except ValueError:
            return _failed_interpretation()

        if not any(token in {"+", "-", "*", "/"} for token in tokens):
            return _failed_interpretation()

        if not _expression_is_grounded(expression, message):
            return _failed_interpretation()

        return {"intent": "expression", "expression": expression}

    if intent == "contextual_operation":
        operator = interpretation.get("operator")
        operand = interpretation.get("operand")

        if operator not in {"+", "-", "*", "/"}:
            return _failed_interpretation()

        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            return _failed_interpretation()

        if not math.isfinite(operand):
            return _failed_interpretation()

        if not _contextual_operation_is_grounded(
            operator,
            float(operand),
            message,
        ):
            return _failed_interpretation()

        return {
            "intent": "contextual_operation",
            "operator": operator,
            "operand": float(operand),
        }

    return _failed_interpretation()


def _process_llm_interpretation(
    interpretation: dict[str, Any],
    memory: Memory,
) -> MathResult:
    """Validate and execute a structured interpretation."""
    intent = interpretation["intent"]

    if intent == "not_math":
        return MathResult(handled=False)

    if intent == "ambiguous":
        return MathResult(
            handled=True,
            needs_clarification=True,
        )

    if intent == "interpretation_failed":
        return MathResult(
            handled=True,
            interpretation_failed=True,
        )

    if intent == "expression":
        result = solve(interpretation["expression"])
        memory.save_result(result)
        return MathResult(handled=True, result=result)

    if intent == "contextual_operation":
        last_result = memory.get_last_result()

        if last_result is None:
            return MathResult(
                handled=True,
                result=None,
                needs_context=True,
            )

        result = solve_with_context(
            last_result,
            interpretation["operator"],
            interpretation["operand"],
        )
        memory.save_result(result)
        return MathResult(handled=True, result=result)

    return MathResult(
        handled=True,
        interpretation_failed=True,
    )


def process_math_message(message: str, memory: Memory) -> MathResult:
    """Process a mathematical request using deterministic rules first."""
    expression = extract_expression(message)

    if expression is not None:
        result = solve(expression)
        memory.save_result(result)
        return MathResult(handled=True, result=result)

    if is_invalid_mathematical_expression(message):
        raise ValueError("Invalid mathematical expression.")

    intent = parse_intent(message)

    if requires_context(expression, intent):
        last_result = memory.get_last_result()

        if last_result is None:
            return MathResult(
                handled=True,
                result=None,
                needs_context=True,
            )

        operation, value = intent
        result = solve_with_context(last_result, operation, value)
        memory.save_result(result)
        return MathResult(handled=True, result=result)

    interpretation = _interpret_with_llm(message)
    return _process_llm_interpretation(interpretation, memory)
