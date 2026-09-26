class Memory:
    """Stores the latest mathematical result."""

    def __init__(self) -> None:
        """Initializes an empty memory."""
        self._last_result: float | None = None

    def save_result(self, result: float) -> None:
        """Stores the latest mathematical result.

        Args:
            result: Mathematical result to store.
        """
        self._last_result = result

    def get_last_result(self) -> float | None:
        """Returns the latest mathematical result.

        Returns:
            The latest stored result, or None if no result exists.
        """
        return self._last_result