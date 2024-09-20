from importlib.metadata import PackageNotFoundError, version

from flow_judge.eval_data_types import EvalInput, EvalOutput
from flow_judge.flow_judge import AsyncFlowJudge, FlowJudge
from flow_judge.metrics import CustomMetric, Metric, RubricItem, list_all_metrics
from flow_judge.models.base import BaseFlowJudgeModel
from flow_judge.models.model_factory import ModelFactory
from flow_judge.utils.prompt_formatter import format_rubric, format_user_prompt, format_vars

try:
    __version__ = version("flow-judge")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "FlowJudge",
    "AsyncFlowJudge",
    "EvalInput",
    "format_vars",
    "format_rubric",
    "format_user_prompt",
    "RubricItem",
    "Metric",
    "CustomMetric",
    "BaseFlowJudgeModel",
    "ModelFactory",
    "EvalOutput",
]

# Add all metric names to __all__
__all__ += list_all_metrics()
