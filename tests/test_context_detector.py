from core.context_detector import requires_context


def test_requires_context_when_intent_exists() -> None:
    expression = None
    intent = ("/", 2.0)

    assert requires_context(expression, intent) is True


def test_does_not_require_context_for_expression() -> None:
    expression = (5.0, "+", 4.0)
    intent = None

    assert requires_context(expression, intent) is False


def test_does_not_require_context_when_both_are_none() -> None:
    expression = None
    intent = None

    assert requires_context(expression, intent) is False
