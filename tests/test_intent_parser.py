from core.intent_parser import parse_intent


def test_returns_none_for_empty_value() -> None:
    result = parse_intent("")

    assert result is None


def test_returns_none_for_unknown_message() -> None:
    result = parse_intent("hello")

    assert result is None


def test_returns_none_for_invalid_operator() -> None:
    result = parse_intent("modulo 5")

    assert result is None


def test_returns_none_for_incomplete_expression() -> None:
    result = parse_intent("3 +")

    assert result is None


def test_returns_none_for_non_numeric_value() -> None:
    result = parse_intent("add banana")

    assert result is None


def test_returns_none_for_invalid_contextual_operation() -> None:
    result = parse_intent("banana")

    assert result is None


def test_parse_add_intent() -> None:
    result = parse_intent("mais 5")

    assert result == ("+", 5.0)


def test_parse_divide_intent() -> None:
    result = parse_intent("divide 2")

    assert result == ("/", 2.0)


def test_parse_shorthand_intent() -> None:
    result = parse_intent("+5")

    assert result == ("+", 5.0)


def test_parse_subtraction_shorthand_intent() -> None:
    result = parse_intent("-5")

    assert result == ("-", 5.0)


def test_parse_number_word() -> None:
    result = parse_intent("mais cinco")

    assert result == ("+", 5.0)


def test_parse_subtraction_with_number_word() -> None:
    result = parse_intent("menos cinco")

    assert result == ("-", 5.0)


def test_parse_multiplication_intent() -> None:
    result = parse_intent("vezes 2")

    assert result == ("*", 2.0)


def test_parse_division_with_number_word() -> None:
    result = parse_intent("dividido por dois")

    assert result == ("/", 2.0)


def test_parse_add_natural_language_intent() -> None:
    result = parse_intent("adicione 5")

    assert result == ("+", 5.0)


def test_parse_add_infinitive_intent() -> None:
    result = parse_intent("adicionar 5")

    assert result == ("+", 5.0)


def test_parse_sum_natural_language_intent() -> None:
    result = parse_intent("some 7")

    assert result == ("+", 7.0)


def test_parse_sum_infinitive_intent() -> None:
    result = parse_intent("somar 7")

    assert result == ("+", 7.0)


def test_parse_subtract_natural_language_intent() -> None:
    result = parse_intent("subtraia 3")

    assert result == ("-", 3.0)


def test_parse_subtract_infinitive_intent() -> None:
    result = parse_intent("subtrair 3")

    assert result == ("-", 3.0)


def test_parse_remove_natural_language_intent() -> None:
    result = parse_intent("tire 2")

    assert result == ("-", 2.0)


def test_parse_remove_infinitive_intent() -> None:
    result = parse_intent("tirar 2")

    assert result == ("-", 2.0)


def test_parse_multiply_natural_language_intent() -> None:
    result = parse_intent("multiplique por 2")

    assert result == ("*", 2.0)


def test_parse_multiply_infinitive_intent() -> None:
    result = parse_intent("multiplicar por 4")

    assert result == ("*", 4.0)


def test_parse_divide_natural_language_intent() -> None:
    result = parse_intent("divida por 4")

    assert result == ("/", 4.0)


def test_parse_divide_without_por() -> None:
    result = parse_intent("divida 4")

    assert result == ("/", 4.0)


def test_parse_divide_infinitive_intent() -> None:
    result = parse_intent("divide por 4")

    assert result == ("/", 4.0)


def test_parse_add_number_word() -> None:
    result = parse_intent("some dois")

    assert result == ("+", 2.0)


def test_parse_subtract_number_word() -> None:
    result = parse_intent("subtraia três")

    assert result == ("-", 3.0)


def test_parse_multiply_number_word() -> None:
    result = parse_intent("multiplique por quatro")

    assert result == ("*", 4.0)


def test_parse_divide_number_word() -> None:
    result = parse_intent("divida por cinco")

    assert result == ("/", 5.0)


def test_parse_decimal_value() -> None:
    result = parse_intent("mais 2.5")

    assert result == ("+", 2.5)


def test_parse_negative_value() -> None:
    result = parse_intent("menos -5")

    assert result == ("-", -5.0)


def test_returns_none_for_unknown_operator_word() -> None:
    result = parse_intent("modulo 5")

    assert result is None


def test_returns_none_for_invalid_number_word() -> None:
    result = parse_intent("mais banana")

    assert result is None


def test_returns_none_for_invalid_division_value() -> None:
    result = parse_intent("divide banana")

    assert result is None


def test_returns_none_for_invalid_multiplication_value() -> None:
    result = parse_intent("multiplique por banana")

    assert result is None


def test_parse_context_prefix() -> None:
    result = parse_intent("agora subtraia 2")

    assert result == ("-", 2.0)


def test_parse_context_prefix_with_por() -> None:
    result = parse_intent("agora subtraia por 2")

    assert result == ("-", 2.0)


def test_parse_english_contextual_multiplication_with_that() -> None:
    result = parse_intent("Multiply that by 8.")

    assert result == ("*", 8.0)


def test_parse_english_contextual_multiplication_with_prefix() -> None:
    result = parse_intent("Now multiply by 8.")

    assert result == ("*", 8.0)


def test_parse_english_contextual_division_with_prefix() -> None:
    result = parse_intent("Then divide by 7.")

    assert result == ("/", 7.0)


def test_parse_spanish_contextual_division_with_result_reference() -> None:
    result = parse_intent("Ahora divide el resultado por 7.")

    assert result == ("/", 7.0)


def test_parse_english_contextual_division_with_result_reference() -> None:
    result = parse_intent("Divide the result by 7.")

    assert result == ("/", 7.0)
