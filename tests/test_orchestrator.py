from unittest.mock import Mock, patch

from agents.math_agent import MathDomainError, MathResult
from core.tools import DivisionByZeroError
from llm.providers.fallback_provider import ProviderFallbackError
from services.memory import Memory
from services.orchestrator import process_message


def test_process_message_with_general_message() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=MathResult(
                handled=False,
            ),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="Hello! How can I help you?",
        ) as mock_write_response,
    ):
        result = process_message("hello", memory)

    assert result == "Hello! How can I help you?"
    mock_process_math.assert_called_once_with("hello", memory)
    mock_write_response.assert_called_once_with("hello")


def test_process_message_without_previous_context() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=MathResult(
                handled=True,
                result=None,
                needs_context=True,
            ),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="Please perform a calculation first.",
        ) as mock_write_response,
    ):
        result = process_message("divide 2", memory)

    assert result == "Please perform a calculation first."
    mock_process_math.assert_called_once_with("divide 2", memory)
    mock_write_response.assert_called_once_with(
        "divide 2",
        None,
        True,
    )


def test_process_message_with_previous_context() -> None:
    memory = Memory()
    memory.save_result(9.0)

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=MathResult(
                handled=True,
                result=4.5,
            ),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="The result is 4.5.",
        ) as mock_write_response,
    ):
        result = process_message("divide 2", memory)

    assert result == "The result is 4.5."
    mock_process_math.assert_called_once_with("divide 2", memory)
    mock_write_response.assert_called_once_with(
        "divide 2",
        4.5,
    )


def test_process_message_with_complete_expression() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=MathResult(
                handled=True,
                result=9.0,
            ),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="The result is 9.",
        ) as mock_write_response,
    ):
        result = process_message("5 + 4", memory)

    assert result == "The result is 9."
    mock_process_math.assert_called_once_with("5 + 4", memory)
    mock_write_response.assert_called_once_with(
        "5 + 4",
        9.0,
    )


def test_process_message_with_division_by_zero_error() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=DivisionByZeroError(
                "Cannot divide by zero",
            ),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="It is not possible to divide by zero.",
        ) as mock_write_response,
    ):
        result = process_message("10 / 0", memory)

    assert result == "It is not possible to divide by zero."
    mock_process_math.assert_called_once_with("10 / 0", memory)
    mock_write_response.assert_called_once_with(
        "10 / 0",
        error="Cannot divide by zero",
    )


def test_process_message_with_math_error() -> None:
    memory = Memory()
    error_message = "Invalid mathematical expression."
    expected_response = "Could you rephrase your mathematical expression?"

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=MathDomainError(error_message),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value=expected_response,
        ) as mock_write_response,
    ):
        result = process_message("5++", memory)

    assert result == expected_response
    mock_process_math.assert_called_once_with("5++", memory)
    mock_write_response.assert_called_once_with(
        "5++",
        error=error_message,
    )


def test_process_message_with_english_expression() -> None:
    memory = Memory()

    math_result = MathResult(
        handled=True,
        result=9.0,
    )

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=math_result,
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="The result is 9.",
        ) as mock_write_response,
    ):
        result = process_message("How much is 5 + 4?", memory)

    assert result == "The result is 9."
    mock_process_math.assert_called_once_with(
        "How much is 5 + 4?",
        memory,
    )
    mock_write_response.assert_called_once_with(
        "How much is 5 + 4?",
        9.0,
    )


def test_process_message_with_spanish_expression() -> None:
    memory = Memory()

    math_result = MathResult(
        handled=True,
        result=9.0,
    )

    message = "¿Cuánto es 5 + 4?"

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=math_result,
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="El resultado es 9.",
        ) as mock_write_response,
    ):
        result = process_message(message, memory)

    assert result == "El resultado es 9."
    mock_process_math.assert_called_once_with(
        message,
        memory,
    )
    mock_write_response.assert_called_once_with(
        message,
        9.0,
    )


def test_process_message_with_invalid_word_expression() -> None:
    memory = Memory()
    error_message = "Invalid mathematical expression."
    expected_response = "Could you rephrase your mathematical expression?"

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=MathDomainError(error_message),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value=expected_response,
        ) as mock_write_response,
    ):
        result = process_message("banana+banana", memory)

    assert result == expected_response
    mock_process_math.assert_called_once_with(
        "banana+banana",
        memory,
    )
    mock_write_response.assert_called_once_with(
        "banana+banana",
        error=error_message,
    )


def test_process_message_with_ambiguous_math_intent() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=MathResult(
                handled=True,
                needs_clarification=True,
            ),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="Could you clarify your calculation?",
        ) as mock_write_response,
    ):
        result = process_message("calculate something", memory)

    assert result == "Could you clarify your calculation?"
    mock_process_math.assert_called_once_with(
        "calculate something",
        memory,
    )
    mock_write_response.assert_called_once_with(
        "calculate something",
        response_type="ambiguous",
    )


def test_process_message_with_interpretation_failure() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=MathResult(
                handled=True,
                interpretation_failed=True,
            ),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="Please rephrase your request.",
        ) as mock_write_response,
    ):
        result = process_message("some unclear request", memory)

    assert result == "Please rephrase your request."
    mock_process_math.assert_called_once_with(
        "some unclear request",
        memory,
    )
    mock_write_response.assert_called_once_with(
        "some unclear request",
        response_type="interpretation_failed",
    )


def test_process_message_with_provider_unavailable() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=MathResult(
                handled=True,
                provider_unavailable=True,
            ),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value=(
                "The AI service is currently unavailable. " "Please try again later."
            ),
        ) as mock_write_response,
    ):
        result = process_message("What is five times seven?", memory)

    assert result == (
        "The AI service is currently unavailable. " "Please try again later."
    )
    mock_process_math.assert_called_once_with(
        "What is five times seven?",
        memory,
    )
    mock_write_response.assert_called_once_with(
        "What is five times seven?",
        response_type="provider_unavailable",
    )


def test_process_message_integrates_math_agent_memory_and_writer() -> None:
    memory = Memory()

    with patch(
        "agents.writer_agent.get_provider",
    ) as mock_writer_provider:
        provider = Mock()
        provider.generate.side_effect = [
            "The result is {{RESULT}}.",
            "The result is {{RESULT}}.",
        ]
        mock_writer_provider.return_value = provider

        first_response = process_message("5 + 4", memory)

        assert first_response == "The result is 9."
        assert memory.get_last_result() == 9.0

        second_response = process_message("divide 2", memory)

        assert second_response == "The result is 4.5."
        assert memory.get_last_result() == 4.5

    assert provider.generate.call_count == 2


def test_math_result_and_memory_survive_writer_provider_failure() -> None:
    memory = Memory()
    mock_provider = Mock()
    mock_provider.generate.side_effect = [
        ProviderFallbackError("Writer provider unavailable"),
        "The result is {{RESULT}}.",
    ]

    with (
        patch(
            "agents.math_agent.get_provider",
            return_value=mock_provider,
        ),
        patch(
            "agents.writer_agent.get_provider",
            return_value=mock_provider,
        ),
    ):
        first_response = process_message("5 + 4", memory)

        assert first_response == "The result is 9."
        assert memory.get_last_result() == 9.0

        second_response = process_message("divide 2", memory)

    assert second_response == "The result is 4.5."
    assert memory.get_last_result() == 4.5
    assert mock_provider.generate.call_count == 2


def test_process_message_returns_safe_response_for_unexpected_math_error(
    caplog,
) -> None:
    """Hide unexpected math errors behind a safe application response."""
    memory = Memory()
    secret = "api_key=do-not-expose"

    with patch(
        "services.orchestrator.process_math_message",
        side_effect=RuntimeError(secret),
    ):
        response = process_message("hello", memory)

    assert response == (
        "An unexpected error occurred while processing your message. "
        "Please try again."
    )
    assert "RuntimeError" in caplog.text
    assert secret not in response
    assert secret not in caplog.text


def test_math_result_is_preserved_when_writer_raises_unexpected_error(
    caplog,
) -> None:
    """Preserve the validated math result when response generation crashes."""
    memory = Memory()
    secret = "sk-test-secret-value"

    with patch(
        "services.orchestrator.write_response",
        side_effect=RuntimeError(f"Provider error: {secret}"),
    ):
        response = process_message("5 + 4", memory)

    assert response == "The result is 9."
    assert memory.get_last_result() == 9.0
    assert "RuntimeError" in caplog.text
    assert secret not in response
    assert secret not in caplog.text
