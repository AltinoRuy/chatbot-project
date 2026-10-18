import json
from unittest.mock import Mock, patch

import pytest

from agents.math_agent import (
    MathResult,
    _contextual_operation_is_grounded,
    _expression_is_grounded,
    calculate,
    process_math_message,
    solve,
    solve_with_context,
)
from core.tools import DivisionByZeroError
from services.memory import Memory


@pytest.mark.parametrize(
    ("left", "operator", "right", "expected"),
    [
        (5, "+", 4, 9),
        (5, "-", 4, 1),
        (5, "*", 4, 20),
        (8, "/", 2, 4),
        (-5, "+", 3, -2),
        (2.5, "*", 2, 5),
    ],
)
def test_calculate_supported_operations(
    left: float,
    operator: str,
    right: float,
    expected: float,
) -> None:
    assert calculate(left, operator, right) == expected


def test_calculate_rejects_unsupported_operator() -> None:
    with pytest.raises(ValueError, match="Unsupported operator"):
        calculate(5, "%", 2)


def test_calculate_division_by_zero() -> None:
    with pytest.raises(DivisionByZeroError):
        calculate(5, "/", 0)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("5+4", 9),
        ("2+3*4", 14),
        ("(2+3)*4", 20),
        ("((18+6)/3)*4-5", 27),
        ("-5+3", -2),
        ("2.5*2", 5),
    ],
)
def test_solve_supported_expressions(
    expression: str,
    expected: float,
) -> None:
    assert solve(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "5++",
        "2*/3",
        "abc",
        "2..5+1",
        "5+",
        "(2+3",
        "2+3)",
        "2+()",
    ],
)
def test_solve_rejects_invalid_expressions(expression: str) -> None:
    with pytest.raises(ValueError):
        solve(expression)


def test_solve_with_previous_context() -> None:
    assert solve_with_context(9, "/", 2) == 4.5


def test_process_complete_expression_saves_result() -> None:
    memory = Memory()

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value="5+4",
        ),
        patch(
            "agents.math_agent.solve",
            return_value=9.0,
        ) as mock_solve,
    ):
        result = process_math_message("Quanto é 5 + 4?", memory)

    assert result == MathResult(handled=True, result=9.0)
    assert memory.get_last_result() == 9.0
    mock_solve.assert_called_once_with("5+4")


def test_process_contextual_operation_uses_previous_result() -> None:
    memory = Memory()
    memory.save_result(9.0)

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=False,
        ),
        patch(
            "agents.math_agent.parse_intent",
            return_value=("/", 2.0),
        ),
        patch(
            "agents.math_agent.solve_with_context",
            return_value=4.5,
        ) as mock_solve,
    ):
        result = process_math_message("divide 2", memory)

    assert result == MathResult(handled=True, result=4.5)
    assert memory.get_last_result() == 4.5
    mock_solve.assert_called_once_with(9.0, "/", 2.0)


def test_process_contextual_operation_without_previous_result() -> None:
    memory = Memory()

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=False,
        ),
        patch(
            "agents.math_agent.parse_intent",
            return_value=("/", 2.0),
        ),
    ):
        result = process_math_message("divide 2", memory)

    assert result == MathResult(
        handled=True,
        result=None,
        needs_context=True,
    )
    assert memory.get_last_result() is None


def test_process_invalid_expression_raises_value_error() -> None:
    memory = Memory()

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=True,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Invalid mathematical expression",
        ):
            process_math_message("5++", memory)


def test_process_general_message_uses_llm_interpretation() -> None:
    memory = Memory()
    provider = Mock()
    provider.generate.return_value = json.dumps({"intent": "not_math"})

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=False,
        ),
        patch(
            "agents.math_agent.parse_intent",
            return_value=None,
        ),
        patch(
            "agents.math_agent.get_provider",
            return_value=provider,
        ),
    ):
        result = process_math_message("Hello, how are you?", memory)

    assert result == MathResult(handled=False)
    assert memory.get_last_result() is None
    provider.generate.assert_called_once()

    prompt = provider.generate.call_args.args[0]
    assert "Math Agent" in prompt
    assert "Hello, how are you?" in prompt


def test_llm_expression_is_evaluated_and_saved() -> None:
    memory = Memory()
    provider = Mock()
    provider.generate.return_value = json.dumps(
        {"intent": "expression", "expression": "5+4"}
    )

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=False,
        ),
        patch(
            "agents.math_agent.parse_intent",
            return_value=None,
        ),
        patch(
            "agents.math_agent.get_provider",
            return_value=provider,
        ),
    ):
        result = process_math_message("Quanto é 5 + 4?", memory)

    assert result == MathResult(handled=True, result=9.0)
    assert memory.get_last_result() == 9.0


def test_llm_contextual_operation_uses_previous_result() -> None:
    memory = Memory()
    memory.save_result(9.0)

    provider = Mock()
    provider.generate.return_value = json.dumps(
        {
            "intent": "contextual_operation",
            "operator": "/",
            "operand": 2,
        }
    )

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=False,
        ),
        patch(
            "agents.math_agent.parse_intent",
            return_value=None,
        ),
        patch(
            "agents.math_agent.get_provider",
            return_value=provider,
        ),
    ):
        result = process_math_message(
            "Take the previous result and divide by two",
            memory,
        )

    assert result == MathResult(handled=True, result=4.5)
    assert memory.get_last_result() == 4.5


def test_llm_contextual_operation_without_memory_requests_context() -> None:
    memory = Memory()
    provider = Mock()
    provider.generate.return_value = json.dumps(
        {
            "intent": "contextual_operation",
            "operator": "/",
            "operand": 2,
        }
    )

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=False,
        ),
        patch(
            "agents.math_agent.parse_intent",
            return_value=None,
        ),
        patch(
            "agents.math_agent.get_provider",
            return_value=provider,
        ),
    ):
        result = process_math_message(
            "Take the previous result and divide by two",
            memory,
        )

    assert result == MathResult(
        handled=True,
        result=None,
        needs_context=True,
    )
    assert memory.get_last_result() is None


@pytest.mark.parametrize(
    "response",
    [
        "not valid json",
        "[]",
        '{"intent": "unknown"}',
        '{"intent": "expression"}',
        '{"intent": "contextual_operation", "operator": "%", "operand": 2}',
        '{"intent": "contextual_operation", "operator": "/", "operand": true}',
    ],
)
def test_llm_rejects_invalid_interpretations(response: str) -> None:
    memory = Memory()
    memory.save_result(9.0)
    provider = Mock()
    provider.generate.return_value = response

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=False,
        ),
        patch(
            "agents.math_agent.parse_intent",
            return_value=None,
        ),
        patch(
            "agents.math_agent.get_provider",
            return_value=provider,
        ),
    ):
        result = process_math_message("Help me understand this", memory)

    assert result.handled is True
    assert result.interpretation_failed is True
    assert memory.get_last_result() == 9.0


def test_llm_ambiguous_intent_requests_clarification() -> None:
    memory = Memory()
    provider = Mock()
    provider.generate.return_value = json.dumps({"intent": "ambiguous"})

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=False,
        ),
        patch(
            "agents.math_agent.parse_intent",
            return_value=None,
        ),
        patch(
            "agents.math_agent.get_provider",
            return_value=provider,
        ),
    ):
        result = process_math_message("Do the calculation", memory)

    assert result == MathResult(
        handled=True,
        needs_clarification=True,
    )
    assert memory.get_last_result() is None


def test_llm_provider_exception_does_not_change_memory() -> None:
    memory = Memory()
    memory.save_result(9.0)

    provider = Mock()
    provider.generate.side_effect = RuntimeError("Provider unavailable")

    with (
        patch(
            "agents.math_agent.extract_expression",
            return_value=None,
        ),
        patch(
            "agents.math_agent.is_invalid_mathematical_expression",
            return_value=False,
        ),
        patch(
            "agents.math_agent.parse_intent",
            return_value=None,
        ),
        patch(
            "agents.math_agent.get_provider",
            return_value=provider,
        ),
    ):
        result = process_math_message("Help me understand this", memory)

    assert result.handled is True
    assert result.interpretation_failed is True
    assert memory.get_last_result() == 9.0


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("5+4", "Quanto é 5 + 4?"),
        ("10/2", "Calculate 10 / 2"),
        ("2*3", "What is 2 * 3?"),
    ],
)
def test_expression_grounding_accepts_explicit_expression(
    expression: str,
    message: str,
) -> None:
    assert _expression_is_grounded(expression, message) is True


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("5+7", "Quanto é 5 + 4?"),
        ("5+4", "Quanto é 15 + 4?"),
        ("20/2", "Calculate 10 / 2"),
    ],
)
def test_expression_grounding_rejects_unmentioned_expression(
    expression: str,
    message: str,
) -> None:
    assert _expression_is_grounded(expression, message) is False


@pytest.mark.parametrize(
    ("message", "operator", "operand", "expected"),
    [
        ("divide by two", "/", 2, True),
        ("divide by three", "/", 2, False),
        ("multiply by two", "*", 2, True),
        ("multiply by two", "/", 2, False),
        ("divide by twenty-five", "/", 25, True),
        ("divide by twenty-five", "/", 20, False),
        ("divide by two or multiply by three", "/", 2, False),
        ("divide by 2", "/", 2, True),
        ("dividir por três", "/", 3, True),
    ],
)
def test_contextual_operation_grounding(
    message: str,
    operator: str,
    operand: float,
    expected: bool,
) -> None:
    assert _contextual_operation_is_grounded(operator, operand, message) is expected


def test_contextual_operation_grounding_rejects_wrong_operator() -> None:
    assert (
        _contextual_operation_is_grounded(
            "/",
            2,
            "multiply by two",
        )
        is False
    )


def test_contextual_operation_grounding_rejects_multiple_operands() -> None:
    assert (
        _contextual_operation_is_grounded(
            "/",
            2,
            "divide by two and then by three",
        )
        is False
    )
