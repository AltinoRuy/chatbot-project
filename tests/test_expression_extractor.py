from core.expression_extractor import extract_expression


def test_extract_addition() -> None:
    assert extract_expression("5+4") == (5.0, "+", 4.0)


def test_extract_division_with_spaces() -> None:
    assert extract_expression("10 / 2") == (10.0, "/", 2.0)


def test_extract_decimal_numbers() -> None:
    assert extract_expression("5.5 * 2") == (5.5, "*", 2.0)


def test_returns_none_for_words() -> None:
    assert extract_expression("banana + banana") is None


def test_returns_none_for_incomplete_expression() -> None:
    assert extract_expression("3 +") is None
