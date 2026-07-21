"""API error types for OpenHarness."""

from __future__ import annotations


class OpenHarnessApiError(RuntimeError):
    """Base class for upstream API failures."""


class AuthenticationFailure(OpenHarnessApiError):
    """Raised when the upstream service rejects the provided credentials."""


class RateLimitFailure(OpenHarnessApiError):
    """Raised when the upstream service rejects the request due to rate limits."""


class RequestFailure(OpenHarnessApiError):
    """Raised for generic request or transport failures."""
