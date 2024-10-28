import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvalInput(BaseModel):
    """Input for evaluation."""

    inputs: list[dict[str, str]] = Field(default_factory=list)
    output: dict[str, str]


class EvalOutput(BaseModel):
    """Output model for evaluation results."""

    feedback: str = Field(..., description="Feedback from the evaluation")
    score: int | None = Field(..., description="Numeric score from the evaluation")

    @classmethod
    def parse(cls, response: str, fail_on_parse_error: bool = False) -> "EvalOutput":
        """Parse the evaluation response from the judge."""
        try:
            # Compile regex patterns
            feedback_pattern = re.compile(r"<feedback>\s*(.*?)\s*</feedback>", re.DOTALL)
            score_pattern = re.compile(r"<score>\s*(\d+)\s*</score>", re.DOTALL)

            feedback_match = feedback_pattern.search(response)
            score_match = score_pattern.search(response)

            if not feedback_match or not score_match:
                raise ValueError("Failed to parse evaluation response.")

            feedback = feedback_match.group(1).strip()
            score = int(score_match.group(1).strip())

            return cls(feedback=feedback, score=score)
        except Exception as e:
            if fail_on_parse_error:
                raise ValueError(f"Failed to parse evaluation response: {e}") from e
            logger.warning(f"Parsing failed for response: {response}. Error: {e}")
            return EvalOutput(feedback="Error", score=-1)
