from unittest.mock import patch

import pytest

from agents.math_agent import (
    MathResult,
    calculate,
    process_math_message,
    solve,
    solve_with_context,
)
from services.memory import Memory


def test_calculate_addition() -> None:
    assert calculate(5, "+", 4) == 9


def test_calculate_division() -> None:
    assert calculate(10, "/", 2) == 5


def test_solve_complete_expression() -> None:
    assert solve("8*3") == 24


def test_solve_with_previous_context() -> None:
    assert solve_with_context(9, "/", 2) == 4.5


def test_invalid_operator_raises_value_error() -> None:
    with pytest.raises(ValueError):
        calculate(5, "%", 2)


def test_calculate_multiplication() -> None:
    assert calculate(6, "*", 5) == 30


def test_solve_invalid_expression_raises_value_error() -> None:
    with pytest.raises(ValueError):
        solve("banana")


def test_calculate_subtraction() -> None:
    assert calculate(10, "-", 3) == 7


def test_process_math_message_with_complete_expression() -> None:
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
        result = process_math_message("5 + 4", memory)

    assert result == MathResult(
        handled=True,
        result=9.0,
        needs_context=False,
    )
    assert memory.get_last_result() == 9.0
    mock_solve.assert_called_once_with("5 + 4")


def test_process_math_message_with_previous_context() -> None:
    memory = Memory()
    memory.save_result(9.0)

    intent = ("/", 2.0)

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
            return_value=intent,
        ),
        patch(
            "agents.math_agent.solve_with_context",
            return_value=4.5,
        ) as mock_solve_with_context,
    ):
        result = process_math_message("divide 2", memory)

    assert result == MathResult(
        handled=True,
        result=4.5,
        needs_context=False,
    )
    assert memory.get_last_result() == 4.5
    mock_solve_with_context.assert_called_once_with(
        9.0,
        "/",
        2.0,
    )


def test_process_math_message_without_previous_context() -> None:
    memory = Memory()

    intent = ("/", 2.0)

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
            return_value=intent,
        ),
    ):
        result = process_math_message("divide 2", memory)

    assert result == MathResult(
        handled=True,
        result=None,
        needs_context=True,
    )
    assert memory.get_last_result() is None


def test_process_math_message_with_general_message() -> None:
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
            return_value=None,
        ),
    ):
        result = process_math_message("hello", memory)

    assert result == MathResult(handled=False)
    assert memory.get_last_result() is None


def test_process_math_message_with_invalid_expression() -> None:
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
        with pytest.raises(ValueError, match="Invalid mathematical expression."):
            process_math_message("5++", memory)
