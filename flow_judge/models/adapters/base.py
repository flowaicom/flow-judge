from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAPIAdapter(ABC):
    """Base adapter layer for making remote requests to hosted models."""
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    @abstractmethod
    def fetch_response(self, request_body: Dict[str, Any]) -> str:
        """Generate a response based on the given request."""
        pass

    @abstractmethod
    def fetch_batched_response(self, request_bodies: list[Dict[str, Any]]) -> list[str]:
        """Generate responses for multiple requests."""
        pass