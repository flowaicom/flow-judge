from typing import Any, Dict, List, Optional, Union, Sequence
from langchain.evaluation import StringEvaluator
from flow_judge import EvalInput, FlowJudge, AsyncFlowJudge
from flow_judge.models import BaseFlowJudgeModel, AsyncBaseFlowJudgeModel
from flow_judge.metrics import Metric, CustomMetric 
import asyncio

class FlowJudgeLangChainEvaluator(StringEvaluator):

    def __init__(
        self, metric: Metric | CustomMetric, model: BaseFlowJudgeModel | AsyncBaseFlowJudgeModel
    ):
        """Initialize the LlamaIndexFlowJudge."""
        if isinstance(metric, (Metric, CustomMetric)):
            self.metric = metric
        else:
            raise ValueError("Invalid metric type. Use Metric or CustomMetric.")
        
         # Validate model and choose appropriate FlowJudge class
        if isinstance(model, (BaseFlowJudgeModel, AsyncBaseFlowJudgeModel)): 
            self.model = model
        else:
            raise ValueError("The model must be an instance of BaseFlowJudgeModel or AsyncBaseFlowJudgeModel.")
        
        # Determine if the model is async-capable
        self.is_async = hasattr(self.model, 'exec_async') and self.model.exec_async 
        
        # Initialize the appropriate judge based on async capability
        if self.is_async:
            self.judge = AsyncFlowJudge(metric=self.metric, model=self.model)
        else:
            self.judge = FlowJudge(metric=self.metric, model=self.model)

    def _prepare_eval_input(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> EvalInput:
        # Combine all inputs into a single dictionary
        all_inputs = {
            "prediction": prediction,
            "reference": reference,
            "input": input,
            **kwargs
        }

        # Prepare eval_inputs based on metric's required_inputs
        eval_inputs = []
        for req_input in self.metric.required_inputs:
            if req_input in all_inputs:
                value = all_inputs[req_input]
                if isinstance(value, (list, Sequence)) and not isinstance(value, str):
                    eval_inputs.extend([{req_input: v} for v in value])
                else:
                    eval_inputs.append({req_input: value})

        # Prepare the output
        output_key = self.metric.required_output
        output_value = all_inputs.get(output_key, prediction)  # Default to prediction if not specified

        return EvalInput(
            inputs=eval_inputs,
            output={output_key: output_value}
        )

    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        eval_input = self._prepare_eval_input(prediction, reference, input, **kwargs)
        result = self.judge.evaluate(eval_input, save_results=False)
        
        return {
            "score": result.score,
            "reasoning": result.feedback,
        } 
        
    async def _aevaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None, 
        sleep_time_in_seconds: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        await asyncio.sleep(sleep_time_in_seconds)
        eval_input = self._prepare_eval_input(prediction, reference, input, **kwargs)
        result = await self.judge.async_evaluate(eval_input, save_results=False)
        
        return {
            "score": result.score,
            "reasoning": result.feedback,
        }

    @property
    def requires_input(self) -> bool:
        return "input" in self.metric.required_inputs

    @property
    def requires_reference(self) -> bool:
        return "reference" in self.metric.required_inputs

    @property
    def evaluation_name(self) -> str:
        return f"flow_judge_{self.metric.name}"

    def get_required_inputs(self) -> List[str]:
        return self.metric.required_inputs + [self.metric.required_output]