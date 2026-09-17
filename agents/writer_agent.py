from llm.provider_factory import get_provider


def write_response(message: str, result=None) -> str:
    if result is not None:
        prompt = (
            "You are a friendly writer agent. "
            "Respond to the user in the same language as the user. "
            "When a mathematical result is provided, "
            "your only job is to communicate that result naturally. "
            "Treat the provided mathematical result as the final answer. "
            "Do not reinterpret the mathematical expression. "
            "Do not assign any other meaning to the numbers. "
            "Do not perform mathematical calculations yourself.\n\n"
            f"User message: {message}\n"
            f"Mathematical result: {result}"
        )
    else:
        prompt = (
            "You are a friendly writer agent. "
            "Respond to the user in the same language as the user. "
            "Respond naturally to the user's message.\n\n"
            f"User message: {message}"
        )

    provider = get_provider()

    return provider.generate(prompt)
