import importlib.resources as pkg_resources
import warnings
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, Template, TemplateError
from pydantic import BaseModel, field_validator

from flow_judge import metrics as metrics_module  # Add this import


class RubricItem(BaseModel):
    """Represents an item in the evaluation rubric.

    :param score: The score associated with this rubric item.
    :type score: int
    :param description: A detailed description of the criteria for this score.
    :type description: str

    :raises ValueError: If the score is negative.
    :warns UserWarning: If the score is above 5, as Flow Judge v0.1 is
    trained on 0-5 integer values.
    """

    score: int
    description: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: int):
        """Validate the score of a rubric item.

        :param v: The score value to validate.
        :type v: int
        :return: The validated score.
        :rtype: int
        :raises ValueError: If the score is negative.
        :warns UserWarning: If the score is above 5.
        """
        if v < 0:
            raise ValueError("Score must be non-negative")
        if v > 5:
            warnings.warn(
                "Flow Judge v0.1 has been trained with 0-5 integer values."
                " Scores above 5 may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )
        return v


class Metric(BaseModel):
    """Represents an evaluation metric.

    :param name: The name of the metric.
    :type name: str
    :param description: A detailed description of the metric.
    :type description: str
    :param criteria: The evaluation criteria for this metric.
    :type criteria: str
    :param rubric: A list of RubricItems defining the scoring rubric.
    :type rubric: List[RubricItem]
    :param required_inputs: A list of required input fields for this metric.
    :type required_inputs: List[str]
    :param required_output: The name of the required output field.
    :type required_output: str

    :raises ValueError: If the rubric contains duplicate scores or if
    scores are not in ascending order.
    """

    name: str
    description: str
    criteria: str
    rubric: list[RubricItem]
    required_inputs: list[str]
    required_output: str

    @field_validator("rubric")
    @classmethod
    def check_rubric_scores(cls, values):
        """Validate the rubric scores.

        :param values: The values to validate.
        :type values: dict
        :return: The validated values.
        :rtype: dict
        :raises ValueError: If the rubric contains duplicate scores or if
        scores are not in ascending order.
        """
        rubric = values.get("rubric")
        if rubric:
            scores = [item.score for item in rubric]
            if len(scores) != len(set(scores)):
                raise ValueError("Rubric contains duplicate scores")
            if scores != sorted(scores):
                raise ValueError("Rubric scores are not in ascending order")
        return values

    @classmethod
    def from_yaml(cls, file_path: str) -> "Metric":
        """Load a metric from a YAML file.

        :param file_path: The path to the YAML file.
        :type file_path: str
        :return: A Metric instance.
        :rtype: Metric
        :raises ValueError: If the YAML is invalid or if there's an error
        loading the metric.
        :raises FileNotFoundError: If the specified file is not found.
        """
        try:
            with open(file_path) as file:
                data = yaml.safe_load(file)
            return cls(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {str(e)}") from e
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Metric file not found: {file_path}") from e
        except Exception as e:
            raise ValueError(f"Error loading metric from {file_path}: {str(e)}") from e

    @classmethod
    def load_all_from_directory(cls, directory: str = "_data/metrics") -> dict[str, "Metric"]:
        """Load all metric templates from a directory.

        :param directory: The directory path containing metric YAML files.
        :type directory: str
        :return: A dictionary of metric names to Metric instances.
        :rtype: Dict[str, Metric]
        """
        metrics_dir = Path(directory)
        if not metrics_dir.is_dir():
            # If the directory doesn't exist, use the package data
            metrics_dir = pkg_resources.files(metrics_module) / "_data" / "metrics"

        metrics = {}
        for file in metrics_dir.glob("**/*.yaml"):
            try:
                relative_path = file.relative_to(metrics_dir)
                metric_name = relative_path.with_suffix("").as_posix()
                metrics[metric_name] = cls.from_yaml(file)
            except Exception as e:
                print(f"Error loading metric from {file}: {str(e)}")
        return metrics

    @staticmethod
    def load_prompt_template(template_name: str) -> str:
        """Load a Jinja template from the prompts directory.

        :param template_name: The name of the template file.
        :type template_name: str
        :return: The rendered template as a string.
        :rtype: str
        :raises ValueError: If there's an error in the template.
        """
        template_dir = Path("_data/prompts")
        if not template_dir.is_dir():
            # If the directory doesn't exist, use the package data
            template_dir = pkg_resources.files(metrics_module) / "_data" / "prompts"

        env = Environment(loader=FileSystemLoader(template_dir))
        try:
            template = env.get_template(template_name)
            return template.render()
        except TemplateError as e:
            raise ValueError(f"Error in template {template_name}: {str(e)}") from e

    def _format_inputs(self, eval_input: dict[str, Any]) -> str:
        """Format all inputs with XML tags and concatenate them.

        :param eval_input: A dictionary of input values.
        :type eval_input: Dict[str, Any]
        :return: A string of formatted inputs.
        :rtype: str
        """
        formatted_inputs = []
        for key, value in eval_input.items():
            if key != self.required_output:
                formatted_inputs.append(f"<{key}>\n{str(value)}\n</{key}>")
        return "\n".join(formatted_inputs)

    def validate_inputs(self, eval_input: dict[str, Any]) -> None:
        """Validate that all required inputs are present in the eval_input.

        :param eval_input: A dictionary of input values.
        :type eval_input: Dict[str, Any]
        :raises ValueError: If any required inputs are missing.
        """
        missing_inputs = [
            input_name for input_name in self.required_inputs if input_name not in eval_input
        ]
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {', '.join(missing_inputs)}")

    def format_rubric(self) -> str:
        """Format the rubric for the prompt.

        :return: A string representation of the rubric.
        :rtype: str
        """
        return "\n".join([f"- Score {item.score}: {item.description}" for item in self.rubric])

    def format_prompt(self, eval_input: dict[str, Any]) -> str:
        """Format the prompt for evaluation based on this metric.

        :param eval_input: A dictionary of input values.
        :type eval_input: Dict[str, Any]
        :return: The formatted prompt string.
        :rtype: str
        :raises ValueError: If there's an error rendering the prompt or if
        the prompt exceeds the maximum length.
        """
        self.validate_inputs(eval_input)

        formatted_inputs = self._format_inputs(eval_input)
        formatted_output = (
            f"<{self.required_output}>\n"
            + f"{eval_input.get(self.required_output, '[No output provided]')}"
            + "\n</{self.required_output}>"
        )

        prompt_variables = {
            "INPUTS": formatted_inputs,
            "OUTPUT": formatted_output,
            "EVALUATION_CRITERIA": self.criteria,
            "RUBRIC": self.format_rubric(),
        }

        template_name = "standard.j2" if formatted_inputs else "outputs_only.j2"
        try:
            template = Template(self.load_prompt_template(template_name))
            rendered_prompt = template.render(**prompt_variables)
            if len(rendered_prompt) > 100000:  # Arbitrary limit, adjust as needed
                raise ValueError("Rendered prompt exceeds maximum length")
            return rendered_prompt
        except Exception as e:
            raise ValueError(f"Error rendering prompt: {str(e)}") from e


class CustomMetric(Metric):
    """Represents a custom evaluation metric.

    This class inherits all functionality from the Metric class and can be
    extended to implement custom behavior for specific evaluation needs.
    """

    pass


def list_all_metrics() -> list[str]:
    """List all metric variable names.

    :return: A list of metric names defined as global variables.
    :rtype: List[str]
    """
    return [
        name for name, value in globals().items() if isinstance(value, Metric) and name.isupper()
    ]
