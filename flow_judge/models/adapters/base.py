from abc import ABC, abstractmethod
from typing import Any


class BaseAPIAdapter(ABC):
    """Base adapter layer for making remote requests to hosted models."""

    def __init__(self, base_url: str):
        """Initialize the BaseAPIAdapter.

        : param: base_url: The baseten deployed model url
        """
        self.base_url = base_url

    @abstractmethod
    def _fetch_response(self, request_body: dict[str, Any]) -> str:
        """Generate a response based on the given request."""
        pass

    @abstractmethod
    def _fetch_batched_response(self, request_bodies: list[dict[str, Any]]) -> list[str]:
        """Generate responses for multiple requests."""
        pass


class AsyncBaseAPIAdapter(ABC):
    """Base adapter layer for making remote requests to hosted models."""

    def __init__(self, base_url: str):
        """Initialize the AsyncBaseAPIAdapter.

        : param: base_url: The baseten deployed model url
        """
        self.base_url = base_url

    @abstractmethod
    async def _async_fetch_response(self, request_body: dict[str, Any]) -> str:
        """Generate a response based on the given request."""
        pass

    @abstractmethod
    async def _async_fetch_batched_response(
        self, request_bodies: list[dict[str, Any]]
    ) -> list[str]:
        """Generate responses for multiple requests."""
        pass
