from services.memory import Memory


def test_save_and_get_result() -> None:
    memory = Memory()

    memory.save_result(9.0)

    assert memory.get_last_result() == 9.0


def test_overwrites_previous_result() -> None:
    memory = Memory()

    memory.save_result(9.0)
    memory.save_result(4.5)

    assert memory.get_last_result() == 4.5


def test_new_memory_starts_empty() -> None:
    memory = Memory()

    assert memory.get_last_result() is None


def test_memories_are_independent() -> None:
    first_memory = Memory()
    second_memory = Memory()

    first_memory.save_result(9.0)
    second_memory.save_result(4.5)

    assert first_memory.get_last_result() == 9.0
    assert second_memory.get_last_result() == 4.5
