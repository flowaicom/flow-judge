from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class FlowJudgeError(BaseModel):
    """Represents an error encountered during the Flow Judge evaluation process.

    This class encapsulates detailed error information, including the type of error,
    the specific message, the request ID that caused the error, and other metadata.

    Attributes:
        error_type (str): The type of error encountered (e.g., "TimeoutError").
        error_message (str): A detailed description of the error.
        request_id (str): The ID of the request that caused the error.
        timestamp (datetime): The time when the error occurred.
        retry_count (int): The number of retry attempts made before the error was raised.
        raw_response (Optional[str]): The raw response from Baseten or proxy, if available.

    Note:
        This class is used for both logging and error handling. Ensure that sensitive
        information is not included in the error_message or raw_response fields.
    """

    error_type: str = Field(..., description="Type of the error encountered")
    error_message: str = Field(..., description="Detailed error message")
    request_id: str | None = Field(
        default=None, description="ID of the request that caused the error"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Time when the error occurred"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts made")
    raw_response: str | None = Field(
        None, description="Raw response from Baseten or proxy, if available"
    )

    @field_validator("error_type", "error_message")
    @classmethod
    def check_non_empty_string(cls, v):
        """Placeholder."""
        if not v.strip():
            raise ValueError("Field must not be empty or just whitespace")
        return v


class BasetenAPIError(Exception):
    """Base exception for Baseten API errors."""

    pass


class BasetenRequestError(BasetenAPIError):
    """Exception for request-related errors."""

    pass


class BasetenResponseError(BasetenAPIError):
    """Exception for response-related errors."""

    pass


class BasetenRateLimitError(BasetenAPIError):
    """Exception for rate limit errors."""

    pass
