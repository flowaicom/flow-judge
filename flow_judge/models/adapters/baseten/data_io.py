from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from flow_judge.models.adapters.baseten.errors import FlowJudgeError


class Message(TypedDict):
    """Represents a single request message for the Baseten API.

    Note:
        This class uses TypedDict for strict type checking. Ensure all fields
        are provided when instantiating. The 'id' field is crucial for tracking
        and error reporting throughout the evaluation process.

    Warning:
        Do not include sensitive information in the 'content' field, as it may
        be logged or stored for debugging purposes.
    """

    id: str
    index: int
    prompt: str
    response: str


class BatchResult(BaseModel):
    """Represents the result of a batch evaluation process.

    This class contains both successful outputs and errors encountered during the
    evaluation process, as well as metadata about the batch operation.

    Attributes:
        successful_outputs (List[Message]): List of successful evaluation outputs.
        errors (List[FlowJudgeError]): List of errors encountered during evaluation.
        total_requests (int): Total number of requests processed in the batch.
        success_rate (float): Rate of successful evaluations (0.0 to 1.0).

    Note:
        The success_rate is calculated as (len(successful_outputs) / total_requests).
        Be cautious when interpreting results with a low success rate, as it may
        indicate systemic issues with the evaluation process or input data.
    """

    successful_outputs: list[Message] = Field(
        default_factory=list, description="List of successful evaluation outputs"
    )
    errors: list[FlowJudgeError] = Field(
        default_factory=list, description="List of errors encountered during evaluation"
    )
    total_requests: int = Field(..., description="Total number of requests processed")
    success_rate: float = Field(..., description="Rate of successful evaluations")

    @field_validator("total_requests")
    @classmethod
    def check_positive_total_requests(cls, v):
        """Placeholder."""
        if v < 0:
            raise ValueError("total_requests must be positive")
        return v

    @field_validator("success_rate")
    @classmethod
    def check_success_rate_range(cls, v):
        """Placeholder."""
        if not 0 <= v <= 1:
            raise ValueError("success_rate must be between 0 and 1")
        return v
