import pytest

from agents.math_agent import calculate, solve, solve_with_context


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
