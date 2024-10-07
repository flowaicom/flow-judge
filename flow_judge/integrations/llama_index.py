import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from llama_index.core.evaluation import BaseEvaluator, EvaluationResult

from flow_judge.eval_data_types import EvalInput
from flow_judge.flow_judge import AsyncFlowJudge
from flow_judge.metrics.metric import CustomMetric, Metric
from flow_judge.models.common import AsyncBaseFlowJudgeModel

logger = logging.getLogger(__name__)


class LlamaIndexFlowJudge(BaseEvaluator):
    """LlamaIndexFlowJudge is a custom evaluator for LlamaIndex.

    It uses FlowJudge to evaluate the performance of a rag system.
    """

    def __init__(
        self, metric: Metric | CustomMetric, model: AsyncBaseFlowJudgeModel, output_dir: str = "output/"
    ):
        """Initialize the LlamaIndexFlowJudge."""
        if isinstance(metric, (Metric, CustomMetric)):
            self.metric = metric
        else:
            raise ValueError("Invalid metric type. Use Metric or CustomMetric.")

        if not isinstance(model, AsyncBaseFlowJudgeModel):
            raise ValueError("Invalid model type. Use AsyncBaseFlowJudgeModel or its subclasses.")

        self.model = model
        self.output_dir = output_dir

        self.judge = AsyncFlowJudge(
            metric=self.metric, model=self.model, output_dir=self.output_dir
        )

    def _get_prompts(self):
        """Get the prompts for the flow judge."""
        pass

    def _update_prompts(self):
        """Update the prompts for the flow judge."""
        pass

    # aevaluate naming required
    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        reference: str | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the performance of a model asynchronously."""
        del kwargs  # Unused
        await asyncio.sleep(sleep_time_in_seconds)

        try:
            available_data = self._prepare_available_data(query, response, contexts, reference)
            eval_input = self._create_eval_input(available_data)
            eval_output = await self.judge.async_evaluate(eval_input)
            return EvaluationResult(
                query=query,
                response=response,
                contexts=contexts,
                feedback=eval_output.feedback,
                score=eval_output.score,
                invalid_result=False,
                invalid_reason=None,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                query=query,
                response=response,
                contexts=contexts,
                feedback=None,
                score=None,
                invalid_result=True,
                invalid_reason=str(e),
            )

    def _prepare_available_data(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        reference: str | None = None,
    ):
        available_data = {}
        if query is not None:
            available_data["query"] = query
        if response is not None:
            available_data["response"] = response
        if contexts is not None:
            available_data["contexts"] = "\n\n".join(contexts)
        if reference is not None:
            available_data["reference"] = reference

        if not available_data:
            raise ValueError(
                "At least one of query, response, contexts, or reference must be provided"
            )
        return available_data

    def _create_eval_input(self, available_data: dict[str, Any]):
        inputs = []
        for required_input in self.metric.required_inputs:
            if required_input not in available_data:
                raise ValueError(
                    f"Required input '{required_input}' is not available in this integration"
                )
            inputs.append({required_input: available_data[required_input]})

        if self.metric.required_output not in available_data:
            raise ValueError(
                f"Required output '{self.metric.required_output}' \
                is not available in this integration"
            )
        output = {self.metric.required_output: available_data[self.metric.required_output]}

        return EvalInput(inputs=inputs, output=output)
