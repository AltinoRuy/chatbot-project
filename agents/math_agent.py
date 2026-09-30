from dataclasses import dataclass

from core.context_detector import requires_context
from core.expression_extractor import (
    extract_expression,
    is_invalid_mathematical_expression,
)
from core.intent_parser import parse_intent
from core.tools import add, divide, multiply, subtract
from services.memory import Memory

MATH_AGENT_PROMPT = """
You are the Math Agent.

Your responsibilities are:
- Identify and process mathematical requests.
- Interpret complete mathematical expressions and contextual operations.
- Use the available mathematical tools for every calculation.
- Treat tool results as the only source of truth.
- Use the previous mathematical result when a contextual operation requires it.
- Return structured mathematical results to the orchestrator.

You must not:
- Perform calculations using language-model reasoning.
- Generate conversational responses for the user.
- Rewrite or modify authoritative mathematical results.
- Perform tasks outside mathematical processing.

Mathematical operations must always be executed through the appropriate tool.
"""


@dataclass
class MathResult:
    """Structured result returned by the Math Agent.

    Attributes:
        handled: Whether the message was identified as a mathematical task.
        result: Authoritative mathematical result, if available.
        needs_context: Whether the task requires a previous mathematical
            result that is not currently available.
    """

    handled: bool
    result: float | None = None
    needs_context: bool = False


def calculate(
    left: float,
    operator: str,
    right: float,
) -> float:
    """Calculates a mathematical operation using the appropriate tool.

    Args:
        left: First operand.
        operator: Mathematical operator.
        right: Second operand.

    Returns:
        The result of the mathematical operation.

    Raises:
        ValueError: If the operator is not supported.
    """
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
    """Tokenizes a mathematical expression.

    A minus sign is treated as a negative sign when it appears at the
    beginning of the expression or immediately after an operator or an
    opening parenthesis. Otherwise, it is treated as the subtraction
    operator.

    Parentheses are preserved as individual tokens.

    Args:
        expression: Mathematical expression to tokenize.

    Returns:
        A list containing numbers, operators, and parentheses.

    Raises:
        ValueError: If the expression contains an invalid token.
    """
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
    """Parses and evaluates mathematical expressions.

    The parser implements standard mathematical precedence:

    1. Parentheses.
    2. Multiplication and division.
    3. Addition and subtraction.

    Every binary operation is delegated to ``calculate`` so that the
    mathematical tools remain the source of truth.
    """

    def __init__(self, tokens: list[str]) -> None:
        """Initializes the expression parser.

        Args:
            tokens: Tokenized mathematical expression.
        """
        self.tokens = tokens
        self.position = 0

    def parse(self) -> float:
        """Parses the complete token sequence.

        Returns:
            The calculated result.

        Raises:
            ValueError: If the expression is syntactically invalid.
        """
        if not self.tokens:
            raise ValueError("Invalid mathematical expression.")

        result = self._parse_expression()

        if self.position != len(self.tokens):
            raise ValueError("Invalid mathematical expression.")

        return result

    def _parse_expression(self) -> float:
        """Parses addition and subtraction."""
        result = self._parse_term()

        while self._match("+") or self._match("-"):
            operator = self.tokens[self.position - 1]
            right = self._parse_term()
            result = calculate(result, operator, right)

        return result

    def _parse_term(self) -> float:
        """Parses multiplication and division."""
        result = self._parse_factor()

        while self._match("*") or self._match("/"):
            operator = self.tokens[self.position - 1]
            right = self._parse_factor()
            result = calculate(result, operator, right)

        return result

    def _parse_factor(self) -> float:
        """Parses numbers and parenthesized expressions."""
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
        """Consumes a token if it matches the expected value.

        Args:
            token: Token to match.

        Returns:
            True if the current token matched and was consumed.
        """
        if self.position >= len(self.tokens):
            return False

        if self.tokens[self.position] != token:
            return False

        self.position += 1
        return True


def evaluate_tokens(tokens: list[str]) -> float:
    """Evaluates tokenized mathematical expressions.

    Mathematical precedence and parentheses are handled by the expression
    parser. All binary calculations are delegated to the mathematical tools
    through ``calculate``.

    Args:
        tokens: Tokenized mathematical expression.

    Returns:
        The calculated result.

    Raises:
        ValueError: If the token sequence is invalid.
    """
    parser = _ExpressionParser(tokens)

    return parser.parse()


def solve(expression: str) -> float:
    """Solves a normalized mathematical expression.

    Args:
        expression: Normalized mathematical expression.

    Returns:
        The calculated result.

    Raises:
        ValueError: If the expression is invalid.
    """
    tokens = tokenize(expression)

    if not tokens:
        raise ValueError("Invalid mathematical expression.")

    return evaluate_tokens(tokens)


def solve_with_context(
    last_result: float,
    operation: str,
    value: float,
) -> float:
    """Calculates an operation using the previous result as the first operand.

    Args:
        last_result: Result from the previous calculation.
        operation: Mathematical operator.
        value: Number to use in the operation.

    Returns:
        The result of the mathematical operation.
    """
    return calculate(last_result, operation, value)


def process_math_message(
    message: str,
    memory: Memory,
) -> MathResult:
    """Processes a user message as a mathematical task.

    This is the public entry point of the Math Agent. It decides whether the
    message is a complete expression, a contextual operation, or not a
    mathematical request.

    Args:
        message: User message.
        memory: Memory instance used to store and retrieve mathematical results.

    Returns:
        A structured MathResult describing the outcome.

    Raises:
        ValueError: If the mathematical expression is invalid.
    """
    expression = extract_expression(message)

    if expression is not None:
        result = solve(expression)
        memory.save_result(result)

        return MathResult(
            handled=True,
            result=result,
        )

    if is_invalid_mathematical_expression(message):
        raise ValueError("Invalid mathematical expression.")

    intent = parse_intent(message)

    if not requires_context(expression, intent):
        return MathResult(handled=False)

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

    return MathResult(
        handled=True,
        result=result,
    )
