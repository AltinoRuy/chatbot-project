from unittest.mock import Mock, patch

import pytest

from agents.writer_agent import write_response


def test_write_response_returns_provider_response() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_writer_sends_authoritative_result_to_provider() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        write_response("5 + 4", 9.0)

    provider.generate.assert_called_once()
    prompt = provider.generate.call_args.args[0]

    assert "The authoritative result is 9." in prompt


def test_writer_prompt_forbids_calculations() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        write_response("5 + 4", 9.0)

    prompt = provider.generate.call_args.args[0]

    assert "Do not calculate anything." in prompt
    assert "Do not introduce any additional numbers." in prompt


def test_writer_prompt_protects_authoritative_result() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        write_response("5 + 4", 9.0)

    prompt = provider.generate.call_args.args[0]

    assert "The authoritative result is provided by the application." in prompt
    assert "Do not calculate anything." in prompt
    assert (
        "Do not interpret, reinterpret, or modify the mathematical operation." in prompt
    )
    assert "The sentence MUST contain the exact placeholder {{RESULT}}." in prompt


def test_write_response_formats_integer_result() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_formats_decimal_result() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("10 / 2", 5.0)

    assert result == "The result is 5."


def test_write_response_uses_fallback_when_result_placeholder_is_missing() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is 9."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_uses_fallback_when_response_has_extra_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}, or 10."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_handles_error_response() -> None:
    provider = Mock()
    provider.generate.return_value = (
        "It is not possible to complete the operation: {{ERROR}}."
    )

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "10 / 0",
            error="Cannot divide by zero",
        )

    assert result == (
        "It is not possible to complete the operation: " "Cannot divide by zero."
    )


def test_writer_sends_authoritative_error_to_provider() -> None:
    provider = Mock()
    provider.generate.return_value = (
        "It is not possible to complete the operation: {{ERROR}}."
    )

    with patch("agents.writer_agent.get_provider", return_value=provider):
        write_response("10 / 0", error="Cannot divide by zero")

    prompt = provider.generate.call_args.args[0]

    assert "The authoritative error is: Cannot divide by zero" in prompt


def test_write_response_falls_back_for_invalid_error_response() -> None:
    provider = Mock()
    provider.generate.return_value = "The answer is 10."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "10 / 0",
            error="Cannot divide by zero",
        )

    assert result == "It is not possible to divide by zero."


def test_write_response_handles_context_requirement() -> None:
    provider = Mock()
    provider.generate.return_value = "There is no previous result."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("divide 2", needs_context=True)

    assert result == "There is no previous result."


def test_write_response_uses_fallback_for_blank_context_response() -> None:
    provider = Mock()
    provider.generate.return_value = "   "

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("divide 2", needs_context=True)

    assert result == "There is no previous mathematical result to use."


def test_write_response_uses_fallback_for_context_response_with_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "Your previous result was 10."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("divide 2", needs_context=True)

    assert result == "There is no previous mathematical result to use."


def test_write_response_handles_non_math_response() -> None:
    provider = Mock()
    provider.generate.return_value = "Hello! How can I help you?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Hello")

    assert result == "Hello! How can I help you?"


def test_write_response_uses_fallback_for_non_math_response_with_new_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "The answer is 42."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Hello")

    assert result == "I can help with that, but I will not perform calculations."


def test_write_response_handles_ambiguous_response() -> None:
    provider = Mock()
    provider.generate.return_value = "Could you clarify what you want to calculate?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="ambiguous",
        )

    assert result == "Could you clarify what you want to calculate?"


def test_write_response_uses_fallback_for_ambiguous_response_with_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "Do you mean 5 or 10?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="ambiguous",
        )

    assert result == "Could you clarify what you want to calculate?"


def test_write_response_handles_interpretation_failure() -> None:
    provider = Mock()
    provider.generate.return_value = (
        "I could not confidently interpret the mathematical expression."
    )

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="interpretation_failed",
        )

    assert result == ("I could not confidently interpret the mathematical expression.")


def test_interpretation_failure_falls_back_when_response_has_numbers() -> None:
    provider = Mock()
    provider.generate.return_value = "Try calculating 10 / 2."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="interpretation_failed",
        )

    assert result == "Could you rephrase your mathematical request?"


def test_write_response_uses_portuguese_for_portuguese_message() -> None:
    provider = Mock()
    provider.generate.return_value = "O resultado é {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("quanto é 2 + 2", 4.0)

    assert result == "O resultado é 4."


def test_write_response_uses_english_for_english_message() -> None:
    provider = Mock()
    provider.generate.return_value = "The result is {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("what is 2 + 2", 4.0)

    assert result == "The result is 4."


def test_write_response_rejects_digits_embedded_in_words() -> None:
    provider = Mock()
    provider.generate.return_value = "The answer is abc123 {{RESULT}}."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


def test_write_response_uses_fallback_for_whitespace_non_math_response() -> None:
    provider = Mock()
    provider.generate.return_value = "   "

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Hello")

    assert result == "I can help with that, but I will not perform calculations."


def test_write_response_detects_portuguese_greeting() -> None:
    provider = Mock()
    provider.generate.return_value = "Olá! Como posso ajudar?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Bom dia, tudo bem?")

    assert result == "Olá! Como posso ajudar?"


@pytest.mark.parametrize("provider_response", [None, 42, ["unexpected"]])
def test_write_response_uses_fallback_for_non_string_result_response(
    provider_response: object,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("5 + 4", 9.0)

    assert result == "The result is 9."


@pytest.mark.parametrize("provider_response", [None, 42, ["unexpected"]])
def test_write_response_uses_fallback_for_non_string_error_response(
    provider_response: object,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "10 / 0",
            error="Cannot divide by zero",
        )

    assert result == "It is not possible to divide by zero."


@pytest.mark.parametrize("provider_response", [None, 42, ["unexpected"]])
def test_write_response_uses_fallback_for_non_string_context_response(
    provider_response: object,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("divide 2", needs_context=True)

    assert result == "There is no previous mathematical result to use."


@pytest.mark.parametrize("provider_response", [None, 42, ["unexpected"]])
def test_write_response_uses_fallback_for_non_string_ambiguous_response(
    provider_response: object,
) -> None:
    provider = Mock()
    provider.generate.return_value = provider_response

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate something",
            response_type="ambiguous",
        )

    assert result == "Could you clarify what you want to calculate?"


def test_write_response_uses_portuguese_result_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Resultado inválido 123."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("quanto é 5 + 4", 9.0)

    assert result == "O resultado é 9."


def test_write_response_uses_portuguese_generic_error_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Resposta inválida 123."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calcular uma expressão",
            error="Invalid expression",
        )

    assert result == "Não foi possível concluir a operação: Invalid expression."


def test_write_response_uses_english_generic_error_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Invalid response 123."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "calculate an expression",
            error="Invalid expression",
        )

    assert result == (
        "It was not possible to complete the operation: Invalid expression."
    )


def test_write_response_uses_portuguese_ambiguous_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Você quer 2 ou 3?"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "poderia calcular isso",
            response_type="ambiguous",
        )

    assert result == "Poderia esclarecer o que deseja calcular?"


def test_write_response_uses_portuguese_interpretation_failure_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "Tente calcular 10 / 2."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response(
            "poderia calcular isso",
            response_type="interpretation_failed",
        )

    assert result == "Poderia reformular seu pedido matemático?"


def test_write_response_uses_portuguese_non_math_fallback() -> None:
    provider = Mock()
    provider.generate.return_value = "A resposta é 42."

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("Olá, tudo bem?")

    assert result == "Posso ajudar com isso, mas não vou realizar cálculos."


def test_writer_uses_portuguese_when_language_markers_are_tied() -> None:
    provider = Mock()
    provider.generate.return_value = "Olá!"

    with patch("agents.writer_agent.get_provider", return_value=provider):
        result = write_response("hello olá")

    assert result == "Olá!"
    prompt = provider.generate.call_args.args[0]
    assert "Respond only in Portuguese." in prompt
