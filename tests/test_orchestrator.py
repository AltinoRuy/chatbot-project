from unittest.mock import patch

from agents.math_agent import MathResult
from services.memory import Memory
from services.orchestrator import process_message


def test_process_message_without_math() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=MathResult(
                handled=False,
                result=None,
            ),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
            return_value="Hello!",
        ) as mock_write_response,
    ):
        result = process_message("Hello", memory)

    assert result == "Hello!"
    mock_process_math.assert_called_once_with("Hello", memory)
    mock_write_response.assert_called_once_with("Hello")


def test_process_message_without_previous_context() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            return_value=MathResult(
                handled=True,
                result=None,
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


def test_process_message_with_math_error() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=ValueError("Invalid mathematical expression."),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
        ) as mock_write_response,
    ):
        result = process_message("5++", memory)

    assert result == "Invalid mathematical expression."
    mock_process_math.assert_called_once_with("5++", memory)
    mock_write_response.assert_not_called()


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
        result = process_message("¿Cuánto es 5 + 4?", memory)

    assert result == "El resultado es 9."
    mock_process_math.assert_called_once_with(
        "¿Cuánto es 5 + 4?",
        memory,
    )
    mock_write_response.assert_called_once_with(
        "¿Cuánto es 5 + 4?",
        9.0,
    )


def test_process_message_with_invalid_word_expression() -> None:
    memory = Memory()

    with (
        patch(
            "services.orchestrator.process_math_message",
            side_effect=ValueError("Invalid mathematical expression."),
        ) as mock_process_math,
        patch(
            "services.orchestrator.write_response",
        ) as mock_write_response,
    ):
        result = process_message("banana+banana", memory)

    assert result == "Invalid mathematical expression."
    mock_process_math.assert_called_once_with(
        "banana+banana",
        memory,
    )
    mock_write_response.assert_not_called()