import asyncio
import logging

from flow_judge.eval_data_types import EvalInput, EvalOutput
from flow_judge.metrics import CustomMetric, Metric
from flow_judge.models.base import AsyncBaseFlowJudgeModel, BaseFlowJudgeModel
from flow_judge.utils.prompt_formatter import format_rubric, format_user_prompt, format_vars
from flow_judge.utils.result_writer import write_results_to_disk
from flow_judge.utils.validators import validate_eval_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseFlowJudge:
    """Base class for FlowJudge with common functionality."""

    def __init__(
        self,
        metric: Metric | CustomMetric,
        model: BaseFlowJudgeModel | AsyncBaseFlowJudgeModel,
        output_dir: str | None = "output/",
    ):
        """Initialize BaseFlowJudge with a metric and model."""
        if not isinstance(metric, (Metric, CustomMetric)):
            raise ValueError("Invalid metric type. Use Metric or CustomMetric.")
        self.metric = metric
        self.output_dir = output_dir
        self.model = model

    def _format_prompt(self, eval_input: EvalInput) -> str:
        """Format the prompt for a single evaluation input."""
        prompt_variables = {
            "INPUTS": format_vars(eval_input.inputs),
            "OUTPUT": format_vars([eval_input.output]),
            "EVALUATION_CRITERIA": self.metric.criteria,
            "RUBRIC": format_rubric(self.metric.rubric),
        }
        return format_user_prompt(prompt_variables)

    def _validate_inputs(self, eval_inputs: EvalInput | list[EvalInput]):
        """Validate required inputs and output against the metric."""
        if isinstance(eval_inputs, list):
            for eval_input in eval_inputs:
                validate_eval_input(eval_input, self.metric)
        else:
            validate_eval_input(eval_inputs, self.metric)

    def _save_results(self, eval_inputs: list[EvalInput], eval_outputs: list[EvalOutput]):
        """Save results to disk."""
        logger.info(f"Saving results to {self.output_dir}")
        write_results_to_disk(
            eval_inputs, eval_outputs, self.model.metadata, self.metric.name, self.output_dir
        )


class FlowJudge(BaseFlowJudge):
    """Synchronous FlowJudge class for evaluating AI outputs."""

    def __init__(
        self,
        metric: Metric | CustomMetric,
        model: BaseFlowJudgeModel,
        output_dir: str | None = "output/",
    ):
        """Initialize FlowJudge with a metric and model."""
        super().__init__(metric, model, output_dir)
        if not isinstance(model, BaseFlowJudgeModel):
            raise ValueError("Invalid model type. Use BaseFlowJudgeModel or its subclasses.")

    def evaluate(self, eval_input: EvalInput, save_results: bool = False) -> EvalOutput:
        """Evaluate a single EvalInput object."""
        try:
            self._validate_inputs(eval_input)
            prompt = self._format_prompt(eval_input)
            response = self.model.generate(prompt)
            eval_output = EvalOutput.parse(response)
            if save_results:
                self._save_results([eval_input], [eval_output])
            return eval_output
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        use_tqdm: bool = True,
        save_results: bool = True,
        fail_on_parse_error: bool = False,
    ) -> list[EvalOutput]:
        """Batch evaluate a list of EvalInput objects."""
        self._validate_inputs(eval_inputs)
        prompts = [self._format_prompt(eval_input) for eval_input in eval_inputs]
        responses = self.model.batch_generate(prompts, use_tqdm=use_tqdm)
        eval_outputs = [
            EvalOutput.parse(response, fail_on_parse_error=fail_on_parse_error)
            for response in responses
        ]
        parse_failures = sum(1 for output in eval_outputs if output.score == -1)
        if save_results:
            self._save_results(eval_inputs, eval_outputs)
        if parse_failures > 0:
            logger.warning(f"Number of parsing failures: {parse_failures} out of {len(responses)}")

        return eval_outputs


class AsyncFlowJudge(BaseFlowJudge):
    """Asynchronous FlowJudge class for evaluating AI outputs."""

    def __init__(
        self,
        metric: Metric | CustomMetric,
        model: AsyncBaseFlowJudgeModel,
        output_dir: str | None = "output/",
    ):
        """Initialize AsyncFlowJudge with a metric and model."""
        super().__init__(metric, model, output_dir)
        if not isinstance(model, AsyncBaseFlowJudgeModel):
            raise ValueError("Invalid model type. Use AsyncBaseFlowJudgeModel or its subclasses.")

    async def async_evaluate(self, eval_input: EvalInput, save_results: bool = False) -> EvalOutput:
        """Evaluate a single EvalInput object asynchronously."""
        try:
            self._validate_inputs(eval_input)
            prompt = self._format_prompt(eval_input)
            response = await self.model.async_generate(prompt)
            eval_output = EvalOutput.parse(response)
            if save_results:
                await asyncio.to_thread(self._save_results, [eval_input], [eval_output])
            return eval_output
        except Exception as e:
            logger.error(f"Asynchronous evaluation failed: {e}")
            raise

    async def async_batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        use_tqdm: bool = True,
        save_results: bool = True,
        fail_on_parse_error: bool = False,
    ) -> list[EvalOutput]:
        """Batch evaluate a list of EvalInput objects asynchronously."""
        self._validate_inputs(eval_inputs)
        prompts = [self._format_prompt(eval_input) for eval_input in eval_inputs]
        responses = await self.model.async_batch_generate(prompts, use_tqdm=use_tqdm)
        eval_outputs = [
            EvalOutput.parse(response, fail_on_parse_error=fail_on_parse_error)
            for response in responses
        ]
        parse_failures = sum(1 for output in eval_outputs if output.score == -1)
        if save_results:
            await asyncio.to_thread(self._save_results, eval_inputs, eval_outputs)

        if parse_failures > 0:
            logger.warning(f"Number of parsing failures: {parse_failures} out of {len(responses)}")

        return eval_outputs
