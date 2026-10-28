import pytest

from core.tools import (
    DivisionByZeroError,
    add,
    divide,
    multiply,
    subtract,
)


def test_add_returns_sum() -> None:
    assert add(5, 4) == 9


def test_subtract_returns_difference() -> None:
    assert subtract(10, 3) == 7


def test_multiply_returns_product() -> None:
    assert multiply(6, 5) == 30


def test_divide_returns_quotient() -> None:
    assert divide(10, 2) == 5


def test_divide_by_zero_raises_division_by_zero_error() -> None:
    with pytest.raises(
        DivisionByZeroError,
        match="Cannot divide by zero",
    ):
        divide(10, 0)
