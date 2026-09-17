import ollama


def parse_intent(message: str) -> str:
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Identify the mathematical operation and number requested by the user. "
                    "Return exactly this format: OPERATION|NUMBER. "
                    "Do not put | at the beginning or end. "
                    "Do not use any other characters. "
                    "The operation must be exactly one of: +, -, *, /. "
                    "The number must be only the number. "
                    "Examples: "
                    "subtract 2 -> -|2. "
                    "multiply 3 -> *|3. "
                    "divide 2 -> /|2. "
                    "add 5 -> +|5. "
                    "If there is no mathematical operation, return none."
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    result = response.message.content.strip()
    print("RAW INTENT:", repr(result))

    if result == "none":
        return None

    parts = result.split("|")

    if len(parts) != 2:
        return None

    operation, value = parts

    operation = operation.strip()
    value = value.strip()

    if operation not in ["+", "-", "*", "/"]:
        return None

    if value == "":
        return None

    return operation, float(value)
