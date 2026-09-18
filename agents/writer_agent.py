from llm.provider_factory import get_provider


def write_response(
    message: str,
    result=None,
    needs_context: bool = False,
) -> str:

    if result is not None:
        prompt = (
            "You are a friendly writer agent. "
            "Respond to the user in the same language as the user. "
            "Your only responsibility is to communicate the final result. "
            "The value below was already calculated by another system and is authoritative. "
            "Never perform, repeat, verify, or infer any mathematical calculation. "
            "Never change the provided value. "
            "Do not create an equation. "
            "Do not use numbers from the user message as part of the answer. "
            "Simply communicate the provided final result naturally and briefly.\n\n"
            f"User language sample: {message}\n"
            f"Final result: {result}"
    )

    elif needs_context:
        prompt = (
            "You are a friendly writer agent. "
            "Respond to the user in the same language as the user. "
            "The user requested a mathematical operation that requires "
            "a previous result, but no previous result is available. "
            "Explain this naturally and ask the user to perform a calculation first.\n\n"
            f"User message: {message}"
        )

    else:
        prompt = (
            "You are a friendly writer agent. "
            "Respond to the user in the same language as the user. "
            "Respond naturally to the user's message.\n\n"
            f"User message: {message}"
        )

    provider = get_provider()
    print("WRITER RESULT:", result)
    return provider.generate(prompt)