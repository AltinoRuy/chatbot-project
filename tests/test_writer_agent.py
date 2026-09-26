from unittest.mock import Mock, patch

from agents.writer_agent import write_response


def test_write_response_returns_provider_response() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_falls_back_when_placeholder_is_missing() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is 9."

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_falls_back_when_response_contains_extra_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = (
        "You can't divide 2 by itself. {{RESULT}}"
    )

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        result = write_response("divide 2", 4.5)

    assert result == "The result is 4.5."


def test_write_response_handles_context_requirement() -> None:
    provider = Mock()
    provider.generate.return_value = "There is no previous result."

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        result = write_response("divide 2", None, True)

    assert result == "There is no previous result."


def test_write_response_handles_non_math_message() -> None:
    provider = Mock()
    provider.generate.return_value = "Hello! How can I help you?"

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        result = write_response("Hello", None, False)

    assert result == "Hello! How can I help you?"


def test_writer_prompt_protects_authoritative_result() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        write_response("5 + 4", 9.0)

    prompt = provider.generate.call_args.args[0]

    assert "The mathematical operation has already been calculated." in prompt
    assert "The authoritative result is provided by the application." in prompt
    assert "Do not calculate anything." in prompt
    assert (
        "Do not interpret, reinterpret, or modify the mathematical operation."
        in prompt
    )
    assert "The sentence MUST contain the exact placeholder {{RESULT}}." in prompt