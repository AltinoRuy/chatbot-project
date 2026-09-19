from unittest.mock import Mock, patch

from core.intent_parser import parse_intent


def test_returns_none_for_empty_value() -> None:
    provider = Mock()
    provider.generate.return_value = "+|"

    with patch(
        "core.intent_parser.get_provider",
        return_value=provider,
    ):
        result = parse_intent("add")

    assert result is None


def test_parse_add_intent() -> None:
    provider = Mock()
    provider.generate.return_value = "+|5"

    with patch(
        "core.intent_parser.get_provider",
        return_value=provider,
    ):
        result = parse_intent("add 5")

    assert result == ("+", 5.0)


def test_parse_divide_intent() -> None:
    provider = Mock()
    provider.generate.return_value = "/|2"

    with patch(
        "core.intent_parser.get_provider",
        return_value=provider,
    ):
        result = parse_intent("divide 2")

    assert result == ("/", 2.0)


def test_returns_none_for_invalid_intent() -> None:
    provider = Mock()
    provider.generate.return_value = "none"

    with patch(
        "core.intent_parser.get_provider",
        return_value=provider,
    ):
        result = parse_intent("hello")

    assert result is None


def test_returns_none_for_invalid_provider_format() -> None:
    provider = Mock()
    provider.generate.return_value = "invalid response"

    with patch(
        "core.intent_parser.get_provider",
        return_value=provider,
    ):
        result = parse_intent("something")

    assert result is None


def test_returns_none_for_invalid_operator() -> None:
    provider = Mock()
    provider.generate.return_value = "%|5"

    with patch(
        "core.intent_parser.get_provider",
        return_value=provider,
    ):
        result = parse_intent("modulo 5")

    assert result is None


def test_does_not_call_provider_for_incomplete_expression() -> None:
    provider = Mock()

    with patch(
        "core.intent_parser.get_provider",
        return_value=provider,
    ):
        result = parse_intent("3 +")

    assert result is None
    provider.generate.assert_not_called()


def test_returns_none_for_non_numeric_value() -> None:
    provider = Mock()
    provider.generate.return_value = "+|banana"

    with patch(
        "core.intent_parser.get_provider",
        return_value=provider,
    ):
        result = parse_intent("add banana")

    assert result is None
