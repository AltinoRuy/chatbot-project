"""Shared error types for language model providers."""


class ProviderConfigurationError(RuntimeError):
    """Raised when a provider cannot initialize due to its own configuration."""
