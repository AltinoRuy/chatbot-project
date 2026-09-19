from unittest.mock import patch

from services.orchestrator import process_message


def test_process_message_with_general_message() -> None:
    with (
        patch(
            "services.orchestrator.extract_expression",
            return_value=None,
        ),
        patch(
            "services.orchestrator.parse_intent",
            return_value=None,
        ),
        patch(
            "services.orchestrator.solve",
        ) as mock_solve,
        patch(
            "services.orchestrator.solve_with_context",
        ) as mock_solve_with_context,
        patch(
            "services.orchestrator.get_last_result",
        ) as mock_get_last_result,
        patch(
            "services.orchestrator.save_result",
        ) as mock_save_result,
        patch(
            "services.orchestrator.write_response",
            return_value="Hello! How can I help you?",
        ) as mock_write_response,
    ):
        result = process_message("hello")

    assert result == "Hello! How can I help you?"
    mock_solve.assert_not_called()
    mock_solve_with_context.assert_not_called()
    mock_get_last_result.assert_not_called()
    mock_save_result.assert_not_called()
    mock_write_response.assert_called_once_with(
        "hello",
        None,
    )


def test_process_message_without_previous_context() -> None:
    intent = ("/", 2.0)

    with (
        patch(
            "services.orchestrator.extract_expression",
            return_value=None,
        ),
        patch(
            "services.orchestrator.parse_intent",
            return_value=intent,
        ),
        patch(
            "services.orchestrator.get_last_result",
            return_value=None,
        ) as mock_get_last_result,
        patch(
            "services.orchestrator.solve_with_context",
        ) as mock_solve_with_context,
        patch(
            "services.orchestrator.write_response",
            return_value="Please perform a calculation first.",
        ) as mock_write_response,
    ):
        result = process_message("divide 2")

    assert result == "Please perform a calculation first."
    mock_get_last_result.assert_called_once()
    mock_solve_with_context.assert_not_called()
    mock_write_response.assert_called_once_with(
        "divide 2",
        None,
        True,
    )


def test_process_message_with_previous_context() -> None:
    intent = ("/", 2.0)

    with (
        patch(
            "services.orchestrator.extract_expression",
            return_value=None,
        ),
        patch(
            "services.orchestrator.parse_intent",
            return_value=intent,
        ),
        patch(
            "services.orchestrator.get_last_result",
            return_value=9.0,
        ) as mock_get_last_result,
        patch(
            "services.orchestrator.solve_with_context",
            return_value=4.5,
        ) as mock_solve_with_context,
        patch(
            "services.orchestrator.save_result",
        ) as mock_save_result,
        patch(
            "services.orchestrator.write_response",
            return_value="The result is 4.5.",
        ) as mock_write_response,
    ):
        result = process_message("divide 2")

    assert result == "The result is 4.5."
    mock_get_last_result.assert_called_once()
    mock_solve_with_context.assert_called_once_with(9.0, "/", 2.0)
    mock_save_result.assert_called_once_with(4.5)
    mock_write_response.assert_called_once_with("divide 2", 4.5)


def test_process_message_with_complete_expression() -> None:
    expression = (5.0, "+", 4.0)

    with (
        patch(
            "services.orchestrator.extract_expression",
            return_value=expression,
        ),
        patch(
            "services.orchestrator.solve",
            return_value=9.0,
        ) as mock_solve,
        patch(
            "services.orchestrator.save_result",
        ) as mock_save_result,
        patch(
            "services.orchestrator.write_response",
            return_value="The result is 9.",
        ) as mock_write_response,
        patch(
            "services.orchestrator.parse_intent",
        ) as mock_parse_intent,
    ):
        result = process_message("5 + 4")

    assert result == "The result is 9."
    mock_solve.assert_called_once_with("5 + 4")
    mock_save_result.assert_called_once_with(9.0)
    mock_write_response.assert_called_once_with("5 + 4", 9.0)
    mock_parse_intent.assert_not_called()
