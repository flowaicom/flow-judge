from pydantic import BaseModel


class RubricItem(BaseModel):
    """Represents an item in the evaluation rubric."""

    score: int
    description: str


class Metric(BaseModel):
    """Represents an evaluation metric."""

    name: str
    criteria: str
    rubric: list[RubricItem]
    required_inputs: list[str] | None = None
    required_output: str

    def print_required_keys(self):
        """Prints the required input and output keys."""
        print(f"Metric: {self.name}")
        print("Required inputs:", ", ".join(self.required_inputs or []))
        print("Required output:", self.required_output)


class CustomMetric(Metric):
    """Represents a custom evaluation metric."""

    def __init__(
        self,
        name: str,
        criteria: str,
        rubric: list[RubricItem],
        required_inputs: list[str],
        required_output: str,
    ):
        """Initialize a custom metric."""
        super().__init__(
            name=name,
            criteria=criteria,
            rubric=rubric,
            required_inputs=required_inputs,
            required_output=required_output,
        )
