import ollama


def write_response(message: str, result=None) -> str:
    if result is not None:
        prompt = (
            f"User message: {message}\n"
            f"Mathematical result: {result}"
        )
    else:
        prompt = (
            f"User message: {message}\n"
            "There is no mathematical result. "
            "Respond naturally to the user's message."
        )

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly writer agent. "
                    "Respond to the user in the same language as the user. "
                    "When a mathematical result is provided, "
                    "your only job is to explain that result naturally. "
                    "Do not reinterpret the user's mathematical expression. "
                    "Do not invent another meaning for the numbers. "
                    "Do not perform mathematical calculations yourself. "
                    "If there is no mathematical result, respond naturally to the user's message."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.message.content