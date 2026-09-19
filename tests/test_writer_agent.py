from unittest.mock import Mock, patch

from agents.writer_agent import write_response


def test_write_response_returns_provider_response() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is 9."

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_without_previous_context() -> None:
    provider = Mock()
    provider.generate.return_value = "Please perform a calculation first."

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        result = write_response(
            "divide 2",
            None,
            True,
        )

    assert result == "Please perform a calculation first."


def test_write_response_for_general_message() -> None:
    provider = Mock()
    provider.generate.return_value = "Hello! How can I help you?"

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        result = write_response("hello")

    assert result == "Hello! How can I help you?"


def test_writer_sends_authoritative_result_to_provider() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is 9."

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        write_response("5 + 4", 9.0)

    provider.generate.assert_called_once()

    prompt = provider.generate.call_args.args[0]

    assert "Final result: 9.0" in prompt


def test_writer_prompt_forbids_calculations() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is 9."

    with patch(
        "agents.writer_agent.get_provider",
        return_value=provider,
    ):
        write_response("5 + 4", 9.0)

    prompt = provider.generate.call_args.args[0]

    assert (
        "Never perform, repeat, verify, or infer any mathematical calculation."
        in prompt
    )
    assert "Never change the provided value." in prompt
