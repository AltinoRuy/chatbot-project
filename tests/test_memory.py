from services.memory import get_last_result, save_result


def test_save_and_get_result() -> None:
    save_result(9.0)

    assert get_last_result() == 9.0


def test_overwrites_previous_result() -> None:
    save_result(9.0)
    save_result(4.5)

    assert get_last_result() == 4.5
