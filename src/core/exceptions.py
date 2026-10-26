"""Custom application and client exceptions."""


class IRArenaError(Exception):
    """Base exception for all IR-Arena domain and infrastructure errors."""


class TransientAPIError(IRArenaError):
    """Exception raised for transient HTTP status codes (e.g. 429, 5xx) to trigger retry."""
