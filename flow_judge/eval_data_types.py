import re

from pydantic import BaseModel, Field, field_validator


class EvalInput(BaseModel):
    """Input for evaluation."""

    inputs: list[dict[str, str]] = Field(default_factory=list)
    output: dict[str, str]


class EvalOutput(BaseModel):
    """Output model for evaluation results."""

    feedback: str = Field(..., description="Feedback from the evaluation")
    score: int = Field(..., description="Numeric score from the evaluation")

    @field_validator("score")
    @classmethod
    def score_must_be_non_negative(cls, v):
        """Validate that the score is non-negative."""
        if v < 0:
            raise ValueError("Score must be non-negative")
        return v

    @classmethod
    def parse(cls, response: str) -> "EvalOutput":
        """Parse the evaluation response from the judge."""
        # Compile regex patterns
        feedback_pattern = re.compile(r"<feedback>\s*(.*?)\s*</feedback>", re.DOTALL)
        score_pattern = re.compile(r"<score>\s*(\d+)\s*</score>", re.DOTALL)

        feedback_match = feedback_pattern.search(response)
        score_match = score_pattern.search(response)

        if not feedback_match or not score_match:
            raise ValueError("Failed to parse evaluation response. Response: " + response)

        feedback = feedback_match.group(1).strip()
        score = int(score_match.group(1).strip())

        return cls(feedback=feedback, score=score)
