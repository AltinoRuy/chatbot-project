from core.expression_extractor import (
    extract_expression,
    is_context_operation,
    is_invalid_mathematical_expression,
)


def test_extract_simple_expression() -> None:
    assert extract_expression("5+4") == "5+4"


def test_extract_expression_with_spaces() -> None:
    assert extract_expression("5 + 4") == "5+4"


def test_extract_decimal_expression() -> None:
    assert extract_expression("5.5 * 2") == "5.5*2"


def test_extract_negative_expression() -> None:
    assert extract_expression("-5 + 2") == "-5+2"


def test_extract_number_words_portuguese() -> None:
    assert extract_expression("cinco mais quatro") == "5+4"


def test_extract_number_words_english() -> None:
    assert extract_expression("five plus four") == "5+4"


def test_extract_number_words_spanish() -> None:
    assert extract_expression("cinco más cuatro") == "5+4"


def test_extract_expression_inside_portuguese_sentence() -> None:
    assert extract_expression("Quanto é 5 + 4?") == "5+4"


def test_extract_expression_inside_english_sentence() -> None:
    assert extract_expression("How much is 5 + 4?") == "5+4"


def test_extract_expression_inside_spanish_sentence() -> None:
    assert extract_expression("¿Cuánto es 5 + 4?") == "5+4"


def test_returns_none_for_non_math_text() -> None:
    assert extract_expression("I have 5 apples and 4 bananas") is None


def test_detect_context_operation() -> None:
    assert is_context_operation("+5") is True


def test_detect_context_division() -> None:
    assert is_context_operation("/2") is True


def test_context_operation_rejects_complete_expression() -> None:
    assert is_context_operation("5+4") is False


def test_detect_invalid_expression() -> None:
    assert is_invalid_mathematical_expression("5+") is True


def test_detect_invalid_operator_sequence() -> None:
    assert is_invalid_mathematical_expression("2//3") is True


def test_detect_invalid_expression_with_words() -> None:
    assert is_invalid_mathematical_expression("banana+banana") is True


def test_detect_invalid_multiplication_with_words() -> None:
    assert is_invalid_mathematical_expression("banana*banana") is True


def test_detect_invalid_division_with_words() -> None:
    assert is_invalid_mathematical_expression("banana/banana") is True


def test_detect_invalid_subtraction_with_words() -> None:
    assert is_invalid_mathematical_expression("banana-banana") is True


def test_ignore_non_math_sentence() -> None:
    assert is_invalid_mathematical_expression("hello world") is False


def test_ignore_sentence_with_numbers_without_operator() -> None:
    assert is_invalid_mathematical_expression("I have 5 apples and 4 bananas") is False


def test_ignore_plain_word() -> None:
    assert is_invalid_mathematical_expression("banana") is False
