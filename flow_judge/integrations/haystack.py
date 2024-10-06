import logging
from typing import Any, List, Dict, Optional

import numpy as np
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import deserialize_type

from flow_judge.flow_judge import EvalInput, EvalOutput, FlowJudge
from flow_judge.metrics.metric import CustomMetric, Metric
from flow_judge.models.base import BaseFlowJudgeModel

logger = logging.getLogger(__name__)

# Based on https://github.com/deepset-ai/haystack/blob/d234c75168dcb49866a6714aa232f37d56f72cab/haystack/components/evaluators/llm_evaluator.py#L354


@component
class HaystackFlowJudge:
    """A component that uses FlowJudge to evaluate inputs."""

    def __init__(
        self,
        metric: Metric | CustomMetric,
        model: BaseFlowJudgeModel,
        output_dir: str = "output/",
        progress_bar: bool = True,
        raise_on_failure: bool = True,
        save_results: bool = True,
        fail_on_parse_error: bool = False,
    ):
        """Construct a new FlowJudge evaluator."""
        if isinstance(metric, (Metric, CustomMetric)):
            self.metric = metric
        else:
            raise ValueError("Invalid metric type. Use Metric or CustomMetric.")

        if not isinstance(model, BaseFlowJudgeModel):
            raise ValueError("Invalid model type. Use BaseFlowJudgeModel or its subclasses.")

        self.model = model
        self.output_dir = output_dir

        self.judge = FlowJudge(metric=self.metric, model=self.model, output_dir=self.output_dir)

        # extract inputs and output from the metric
        self.inputs, self.outputs = self._extract_vars_from_metric(self.metric)
        self.validate_init_parameters(self.inputs, self.outputs)
        self.raise_on_failure = raise_on_failure
        self.progress_bar = progress_bar
        self.save_results = save_results
        self.fail_on_parse_error = fail_on_parse_error

        component.set_input_types(self, **dict(self.inputs))

    @staticmethod
    def _extract_vars_from_metric(
        metric: Metric | CustomMetric,
    ) -> tuple[list[tuple[str, type[list]]], list[str]]:
        """Extract the inputs to the component and its type from the metric.

        It also sets the output of the component.
        """
        eval_inputs_keys: list[str] = metric.required_inputs
        eval_output_key: str = metric.required_output

        inputs = [(key, list[str]) for key in eval_inputs_keys + [eval_output_key]]

        outputs = ["feedback", "score"]

        return inputs, outputs

    @staticmethod
    def validate_init_parameters(inputs: list[tuple[str, type[list]]], outputs: list[str]):
        """Validate the init parameters."""
        # Validate inputs
        if (
            not isinstance(inputs, list)
            or not all(isinstance(_input, tuple) for _input in inputs)
            or not all(
                isinstance(_input[0], str) and _input[1] is not list and len(_input) == 2
                for _input in inputs
            )
        ):
            msg = (
                f"FlowJudge evaluator expects inputs to \
                be a list of tuples. Each tuple must contain an input name and "
                f"type of list but received {inputs}."
            )
            raise ValueError(msg)

        # Validate outputs
        if not isinstance(outputs, list) or not all(isinstance(output, str) for output in outputs):
            msg = f"FlowJudge evaluator expects outputs \
                to be a list of str but received {outputs}."
            raise ValueError(msg)

    @component.output_types(
        results=List[Optional[Dict[str, Any]]],
        metadata=Dict[str, Any],
        score=float,
        individual_scores=List[float],
        error_summary=Dict[str, int],
    )
    def run(self, **inputs) -> Dict[str, Any]:
        """Run the FlowJudge evaluator on the provided inputs."""
        self._validate_input_parameters(dict(self.inputs), inputs)
        eval_inputs: List[EvalInput] = self._prepare_inputs(inputs=inputs, metric=self.metric)
        eval_outputs: List[EvalOutput] = self.judge.batch_evaluate(
            eval_inputs,
            save_results=self.save_results,
            fail_on_parse_error=self.fail_on_parse_error,
        )

        results: List[Optional[Dict[str, Any]]] = []
        parsing_errors = 0
        other_errors = 0

        for eval_output in eval_outputs:
            if eval_output.score != -1:
                result = {
                    "feedback": eval_output.feedback,
                    "score": eval_output.score,
                }
                results.append(result)
            else:
                results.append(None)
                if "parsing error" in eval_output.feedback.lower():
                    parsing_errors += 1
                    if self.fail_on_parse_error:
                        raise ValueError(f"Parsing error encountered: {eval_output.feedback}")
                else:
                    other_errors += 1

        total_errors = parsing_errors + other_errors
        if total_errors > 0:
            error_msg = f"FlowJudge encountered errors in {total_errors} out of {len(eval_outputs)} evaluations."
            error_msg += f"\n- Parsing errors: {parsing_errors}"
            error_msg += f"\n- Other errors: {other_errors}"
            logger.warning(error_msg)

        metadata = self.model.metadata

        valid_scores = [result["score"] for result in results if result is not None]

        if not valid_scores:
            logger.warning("No valid scores were generated. All evaluations failed.")
            score = 0.0
            individual_scores = []
        else:
            score = np.mean(valid_scores)
            individual_scores = [float(s) for s in valid_scores]

        error_summary = {
            "total_evaluations": len(eval_outputs),
            "successful_evaluations": len(valid_scores),
            "parsing_errors": parsing_errors,
            "other_errors": other_errors,
        }

        # Add error information to metadata
        metadata["error_summary"] = error_summary

        return {
            "results": results,
            "metadata": metadata,
            "score": score,
            "individual_scores": individual_scores,
            "error_summary": error_summary,
        }

    @staticmethod
    def _validate_input_parameters(expected: dict[str, Any], received: dict[str, Any]) -> None:
        """Validate the input parameters."""
        # Validate that all expected inputs are present in the received inputs
        for param in expected.keys():
            if param not in received:
                msg = f"FlowJudge evaluator expected input \
                    parameter '{param}' but received only {received.keys()}."
                raise ValueError(msg)

        # Validate that all received inputs are lists
        if not all(isinstance(_input, list) for _input in received.values()):
            msg = (
                "FlowJudge evaluator expects all input values to be lists but received "
                f"{[type(_input) for _input in received.values()]}."
            )
            raise ValueError(msg)

        # Validate that all received inputs are of the same length
        inputs = received.values()
        length = len(next(iter(inputs)))
        if not all(len(_input) == length for _input in inputs):
            msg = (
                f"FlowJudge evaluator expects all input lists\
                    to have the same length but received {inputs} with lengths "
                f"{[len(_input) for _input in inputs]}."
            )
            raise ValueError(msg)

    @staticmethod
    def _prepare_inputs(inputs: dict[str, Any], metric: Metric | CustomMetric) -> list[EvalInput]:
        """Prepare the inputs for the flow judge."""
        eval_inputs = []
        num_samples = len(next(iter(inputs.values())))

        for i in range(num_samples):
            input_list = []
            output_dict = {}
            for key, value_list in inputs.items():
                temp_dict = {}
                if key in metric.required_inputs:
                    temp_dict[key] = value_list[i]
                    input_list.append(temp_dict)
                elif key == metric.required_output:
                    output_dict[key] = value_list[i]

            if not output_dict:
                raise ValueError(f"Required output '{metric.required_output}' not found in inputs.")

            eval_input = EvalInput(inputs=input_list, output=output_dict)
            eval_inputs.append(eval_input)

        return eval_inputs

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            metric=self.metric,
            model=self.model,
            output_dir=self.output_dir,
            progress_bar=self.progress_bar,
            raise_on_failure=self.raise_on_failure,
            save_results=self.save_results,
            fail_on_parse_error=self.fail_on_parse_error,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HaystackFlowJudge":
        """Deserialize this component from a dictionary."""
        data["init_parameters"]["inputs"] = [
            (name, deserialize_type(type_)) for name, type_ in data["init_parameters"]["inputs"]
        ]

        return default_from_dict(cls, data)
