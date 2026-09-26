from core.expression_extractor import (
    extract_expression,
    is_context_operation,
    normalize_expression,
)


def test_extract_addition() -> None:
    assert extract_expression("5+4") == "5+4"


def test_extract_division_with_spaces() -> None:
    assert extract_expression("10 / 2") == "10/2"


def test_extract_decimal_numbers() -> None:
    assert extract_expression("5.5 * 2") == "5.5*2"


def test_extract_invalid_expression() -> None:
    assert extract_expression("hello") is None


def test_extract_missing_operator() -> None:
    assert extract_expression("42") is None


def test_normalize_portuguese_words() -> None:
    assert normalize_expression("cinco mais quatro") == "5+4"


def test_context_operation() -> None:
    assert is_context_operation("+5") is True
    assert is_context_operation("menos 5") is True
    assert is_context_operation("5+5") is False
