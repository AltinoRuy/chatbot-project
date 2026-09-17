_last_result = None


def save_result(result: float) -> None:
    global _last_result
    _last_result = result


def get_last_result():
    return _last_result
