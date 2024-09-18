import re

from pydantic import BaseModel, Field, field_validator

# Compile regex patterns
FEEDBACK_PATTERN = re.compile(r"<feedback>\s*(.*?)\s*</feedback>", re.DOTALL)
SCORE_PATTERN = re.compile(r"<score>\s*(\d+)\s*</score>", re.DOTALL)


class EvalOutput(BaseModel):
    """Output model for evaluation results."""

    feedback: str = Field(..., description="Feedback from the evaluation")
    score: int = Field(..., description="Numeric score from the evaluation")

    @classmethod
    def parse(cls, response: str) -> "EvalOutput":
        """Parse the evaluation response from the judge."""
        feedback_match = FEEDBACK_PATTERN.search(response)
        score_match = SCORE_PATTERN.search(response)

        if not feedback_match or not score_match:
            raise ValueError("Failed to parse evaluation response. Response: " + response)

        feedback = feedback_match.group(1).strip()
        score = int(score_match.group(1).strip())

        return cls(feedback=feedback, score=score)

    @field_validator("score")
    @classmethod
    def score_must_be_non_negative(cls, v):
        """Validate that the score is non-negative."""
        if v < 0:
            raise ValueError("Score must be non-negative")
        return v
