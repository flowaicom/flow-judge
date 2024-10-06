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
    score: int = Field(..., description="Numeric score from the evaluation")

    @classmethod
    def parse(cls, response: str, fail_on_parse_error: bool = False) -> "EvalOutput":
        """Parse the evaluation response from the judge."""
        try:
            feedback, score = cls._parse_evaluation_response(response)
            return cls(feedback=feedback, score=score)
        except Exception as e:
            if fail_on_parse_error:
                raise ValueError(f"Failed to parse evaluation response: {e}") from e
            logger.warning(f"Parsing failed for response: {response}. Error: {e}")
            return EvalOutput(feedback="Error", score=-1)

    @staticmethod
    def _parse_evaluation_response(response: str) -> tuple[str, int]:
        # Pattern to match feedback (potentially including the score)
        feedback_pattern = re.compile(r"<feedback>\s*([\s\S]*?)\s*(?:</feedback>)?", re.DOTALL)
        # Pattern to match score, either inside or outside feedback tags
        score_pattern = re.compile(r"<score>\s*(\d+)\s*</score>", re.DOTALL)

        try:
            feedback_match = feedback_pattern.search(response)
            if not feedback_match:
                raise ValueError("Failed to find feedback in the response")

            feedback = feedback_match.group(1).strip()

            # Look for score within the feedback
            score_match = score_pattern.search(feedback)
            if score_match:
                # If score is found within feedback, remove it from feedback
                score = int(score_match.group(1))
                feedback = re.sub(score_pattern, '', feedback).strip()
            else:
                # If score is not in feedback, look for it in the entire response
                score_match = score_pattern.search(response)
                if not score_match:
                    raise ValueError("Failed to find score in the response")
                score = int(score_match.group(1))

            return feedback, score

        except Exception as e:
            logger.error(f"Parsing failed for response: {response}. Error: {str(e)}")
            raise ValueError(f"Failed to parse evaluation response: {str(e)}")
