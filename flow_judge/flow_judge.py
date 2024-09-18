import json
import logging
import os
import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

import flow_judge

from .formatting import format_rubric, format_user_prompt, format_vars
from .metrics import CustomMetric, Metric
from .models.models import BaseFlowJudgeModel
from .parsing import EvalOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvalInput(BaseModel):
    """Input model for evaluation."""

    inputs: list[dict[str, str]] = Field(default_factory=list)
    output: str


class FlowJudge:
    """Main class for evaluating AI outputs using specified metrics and models."""

    def __init__(
        self,
        metric: Metric,
        model: BaseFlowJudgeModel,
        output_dir: str | None = "output/",
    ):
        """Initialize FlowJudge with a metric and model."""
        if isinstance(metric, (Metric, CustomMetric)):
            self.metric = metric
        else:
            raise ValueError("Invalid metric type. Use Metric or CustomMetric.")

        if not isinstance(model, BaseFlowJudgeModel):
            raise ValueError("Invalid model type. Use BaseFlowJudgeModel or its subclasses.")

        self.model = model
        self.output_dir = output_dir

    def _format_prompt(self, eval_input: EvalInput) -> str:
        """Format the prompt for a single evaluation input."""
        prompt_variables = {
            "INPUTS": format_vars(eval_input.inputs),
            "OUTPUT": eval_input.output,
            "EVALUATION_CRITERIA": self.metric.criteria,
            "RUBRIC": format_rubric(self.metric.rubric),
        }
        return format_user_prompt(prompt_variables)

    def write_results_to_jsonl(
        self, eval_inputs: list[EvalInput], eval_outputs: list[EvalOutput], output_dir: str
    ):
        """Write evaluation results, inputs, and metadata to separate JSONL files."""
        fmt_metric_name = re.sub(r"\s", "_", re.sub(r"\(|\)", "", self.metric.name.lower()))
        fmt_model_id = self.model.metadata["model_id"].replace("/", "__")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3]
        metadata = {
            "library_version": flow_judge.__version__,
            "timestamp": timestamp,
            **self.model.metadata,
        }

        metric_folder = os.path.join(output_dir, fmt_metric_name)
        os.makedirs(metric_folder, exist_ok=True)

        base_filename = (
            f"{fmt_metric_name}_{fmt_model_id}_{self.model.metadata['model_type']}_{timestamp}"
        )
        metadata_path = os.path.join(metric_folder, f"metadata_{base_filename}.jsonl")
        results_path = os.path.join(metric_folder, f"results_{base_filename}.jsonl")

        # Write metadata file
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

        # Write results file
        with open(results_path, "w", encoding="utf-8") as f:
            for input_data, eval_output in zip(eval_inputs, eval_outputs):
                result = {
                    "sample": input_data.model_dump(),
                    "feedback": eval_output.feedback,
                    "score": eval_output.score,
                }
                f.write(json.dumps(result) + "\n")

    def evaluate(self, eval_input: EvalInput, save_results: bool = False) -> EvalOutput:
        """Evaluate a single input using the specified metric and model."""
        try:
            prompt = self._format_prompt(eval_input)
            response = self.model.generate(prompt)
            eval_output = EvalOutput.parse(response)
            if save_results:
                logger.info(f"Saving results to {self.output_dir}")
                self.write_results_to_jsonl([eval_input], [eval_output], self.output_dir)
            return eval_output
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def batch_evaluate(
        self, eval_inputs: list[EvalInput], use_tqdm: bool = True, save_results: bool = True
    ) -> list[EvalOutput]:
        """Evaluate multiple inputs in batch using the specified metric and model."""
        prompts = [self._format_prompt(eval_input) for eval_input in eval_inputs]
        responses = self.model.batch_generate(prompts, use_tqdm=use_tqdm)
        eval_outputs = [EvalOutput.parse(response) for response in responses]
        if save_results:
            logger.info(f"Saving results to {self.output_dir}")
            self.write_results_to_jsonl(eval_inputs, eval_outputs, self.output_dir)
        return eval_outputs
